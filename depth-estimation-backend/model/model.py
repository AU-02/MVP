import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')

class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                if name not in self.shadow:
                    self.shadow[name] = param.data.clone()

                # Handle shape mismatches dynamically
                if self.shadow[name].shape != param.data.shape:
                    print(f"⚠️ Shape mismatch in EMA update for {name}: {self.shadow[name].shape} -> {param.data.shape}")
                    self.shadow[name] = param.data.clone()

                # EMA Update
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                if name in self.shadow:
                    param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

class DDPM(BaseModel, nn.Module):
    def __init__(self, opt):
        nn.Module.__init__(self)
        super(DDPM, self).__init__(opt)
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None
        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.netG)
        self.set_loss()
        self.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            optim_params = list(self.netG.parameters())
            self.optG = torch.optim.Adam(optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()
        self.load_network()
        self.print_network()
        
    def forward(self, x_in):
        return self.p_losses(x_in)

    def feed_data(self, data):
        """Process input data - expects NIR images and depth ground truth"""
        self.data = self.set_device(data)
        
        # Handle different data formats
        if 'tgt_image' in data and 'tgt_depth_gt' in data:
            self.data['input'] = {
                'HR': self.data['tgt_image'],      # NIR image
                'SR': self.data['tgt_depth_gt']    # Depth ground truth
            }
        elif 'HR' in data and 'SR' in data:
            self.data['input'] = {
                'HR': self.data['HR'],
                'SR': self.data['SR']
            }
        else:
            raise KeyError(f"Invalid data keys in self.data: {self.data.keys()}")

        print(f"Fed data - HR shape: {self.data['input']['HR'].shape}, SR shape: {self.data['input']['SR'].shape}")

    def optimize_parameters(self):
        """Single optimization step"""
        self.optG.zero_grad()

        # Forward pass through diffusion model
        loss = self.netG(self.data['input'])
        
        # Get the predicted depth from the model
        if hasattr(self.netG, "module"):
            self.output = self.netG.module.predicted_depth
        else:
            self.output = self.netG.predicted_depth

        # Backward pass
        loss.backward()
        self.optG.step()
        
        # Update EMA
        self.ema_helper.update(self.netG)
        
        # Log loss
        self.log_dict['l_pix'] = loss.item()

    def test(self, continuous=False):
        """CORRECTED: Generate depth map from NIR image during validation/testing"""
        self.netG.eval()
        
        with torch.no_grad():
            input_data = self.data.get('input', {})
            if 'HR' not in input_data:
                raise KeyError("Missing 'HR' (NIR image) in input data")
            
            # Get NIR image (condition)
            nir_image = input_data['HR']
            print(f"Input NIR image shape: {nir_image.shape}")
            
            # CRITICAL FIX 1: Ensure proper shape WITHOUT adding extra dimensions
            # The shape should be [batch_size, channels, height, width]
            if nir_image.dim() == 3:
                # Shape: [channels, height, width] -> [1, channels, height, width]
                nir_image = nir_image.unsqueeze(0)
            
            # CRITICAL FIX 2: Ensure single channel for NIR
            if nir_image.shape[1] != 1:
                nir_image = nir_image[:, :1, :, :]  # Take only first channel
            
            print(f"Processed NIR image shape: {nir_image.shape}")
            
            # CRITICAL FIX 3: Set the correct noise schedule for inference
            # This is crucial - use validation schedule, not training schedule
            if hasattr(self, 'opt') and 'model' in self.opt and 'beta_schedule' in self.opt['model']:
                val_schedule = self.opt['model']['beta_schedule'].get('val', 
                            self.opt['model']['beta_schedule'].get('test',
                            self.opt['model']['beta_schedule']['train']))
                self.set_new_noise_schedule(val_schedule, schedule_phase='val')
            
            # CRITICAL FIX 4: Call p_sample_loop correctly with NIR as condition
            if isinstance(self.netG, nn.DataParallel):
                # The GaussianDiffusion.p_sample_loop expects (condition, continuous)
                self.output = self.netG.module.p_sample_loop(nir_image, continuous)
            else:
                self.output = self.netG.p_sample_loop(nir_image, continuous)
            
            # CRITICAL FIX 5: Ensure output has correct shape and is on correct device
            if self.output.shape != nir_image.shape:
                # Ensure same spatial dimensions
                if self.output.shape[-2:] != nir_image.shape[-2:]:
                    self.output = F.interpolate(self.output, size=nir_image.shape[-2:], 
                                            mode='bilinear', align_corners=False)
            
            # Ensure output is on same device as input
            self.output = self.output.to(nir_image.device)
        
        self.netG.train()

    def test_debug(self, continuous=False):
        
        self.netG.eval()
        
        with torch.no_grad():
            input_data = self.data.get('input', {})
            nir_image = input_data['HR']
            
            # Check model components
            net = self.netG.module if isinstance(self.netG, nn.DataParallel) else self.netG
            
            # Check noise schedule
            if hasattr(net, 'num_timesteps'):
                print(f"Number of timesteps: {net.num_timesteps}")
            if hasattr(net, 'betas'):
                print(f"Beta schedule shape: {net.betas.shape}")
            
            # Check available methods
            available_methods = [m for m in dir(net) if not m.startswith('_')]
            sampling_methods = [m for m in available_methods if 'sample' in m.lower()]
            print(f"Available sampling methods: {sampling_methods}")
            
            # Prepare NIR image
            if nir_image.dim() == 3:
                nir_image = nir_image.unsqueeze(0)
            if nir_image.shape[1] != 1:
                nir_image = nir_image[:, :1, :, :]
            
            try:
                # Test the sampling
                self.output = net.p_sample_loop(nir_image, continuous)
                
            except Exception as e:
                print(f"❌ Sampling failed: {e}")
                import traceback
                traceback.print_exc()
                
                # Generate dummy output for testing
                self.output = torch.randn_like(nir_image)
        
        self.netG.train()

    def sample(self, batch_size=1, continous=False):
        """Sample random depth maps (not conditioned on input)"""
        self.netG.eval()
        with torch.no_grad():
            # For unconditional sampling, we'd need random NIR-like images
            # This is mainly for testing the generation capability
            fake_condition = torch.randn((batch_size, 1, self.opt['model']['diffusion']['image_size'], 
                                        self.opt['model']['diffusion']['image_size'])).to(self.device)
            
            if isinstance(self.netG, nn.DataParallel):
                self.output = self.netG.module.p_sample_loop(fake_condition, continous)
            else:
                self.output = self.netG.p_sample_loop(fake_condition, continous)
        
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_visuals(self, need_LR=True, sample=False):
        """Get current visuals for logging and visualization"""
        out_dict = OrderedDict()
        
        if sample:
            out_dict['SAM'] = self.output.detach().float().cpu()
        else:
            # Predicted depth map
            if self.output is not None:
                out_dict['Predicted'] = self.output.detach().float().cpu()
            else:
                logger.warning("Output is None!")
                out_dict['Predicted'] = torch.zeros_like(self.data['input']['HR']).detach().cpu()
            
            # Ground truth depth
            if 'SR' in self.data['input']:
                out_dict['GT'] = self.data['input']['SR'].detach().float().cpu()
            else:
                logger.warning("'SR' (ground truth) is missing!")
                out_dict['GT'] = torch.zeros_like(self.data['input']['HR']).detach().cpu()
            
            # Input NIR image
            if 'HR' in self.data['input']:
                out_dict['Input'] = self.data['input']['HR'].detach().float().cpu()
            else:
                logger.warning("'HR' (input) is missing!")
                out_dict['Input'] = torch.zeros_like(self.data['input']['SR']).detach().cpu()
        
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__, self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        """Save network weights and optimizer state"""
        gen_path = os.path.join(self.opt['path']['checkpoint'], 'I{}_E{}_depth_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(self.opt['path']['checkpoint'], 'I{}_E{}_depth_opt.pth'.format(iter_step, epoch))
        
        # Network
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        
        # Optimizer
        opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        opt_state['ema_helper'] = self.ema_helper.state_dict()
        torch.save(opt_state, opt_path)
        
        logger.info(f'Saved model in [{gen_path}] and [{opt_path}]')

    def load_network(self):
        """Load pretrained network weights"""
        load_path = self.opt['path'].get('resume_state', None)
        
        if load_path and load_path != 'none':
            logger.info(f'Loading pretrained model from [{load_path}]')
            
            gen_path = f'{load_path}_depth_gen.pth'
            opt_path = f'{load_path}_depth_opt.pth'
            
            # Network
            if os.path.exists(gen_path):
                network = self.netG
                if isinstance(self.netG, nn.DataParallel):
                    network = network.module
                
                # Load with flexibility for architecture changes
                pretrained_dict = torch.load(gen_path, map_location=self.device)
                model_dict = network.state_dict()
                
                # Filter out incompatible keys
                pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                                 if k in model_dict and v.shape == model_dict[k].shape}
                
                # Update current model
                model_dict.update(pretrained_dict)
                network.load_state_dict(model_dict, strict=False)
                
                logger.info(f'Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained model')
            
            # Optimizer and training state
            if self.opt['phase'] == 'train' and os.path.exists(opt_path):
                opt = torch.load(opt_path, map_location=self.device)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt.get('iter', 0)
                self.begin_epoch = opt.get('epoch', 0)
                self.ema_helper.load_state_dict(opt['ema_helper'])
                logger.info(f'Resumed training from epoch {self.begin_epoch}, step {self.begin_step}')
        else:
            logger.info('Starting training from scratch (no pretrained model loaded)')