import os
import torch
import torch.nn as nn

class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(
            'cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.begin_step = 0
        self.begin_epoch = 0

    # Handle nir + depth data
    def feed_data(self, data):
        self.input = data['input'].to(self.device)
        self.HR = data['HR'].to(self.device)

    def optimize_parameters(self):
        pass

    # Return predicted depth and ground truth
    def get_current_visuals(self):
        visuals = {
            'input': self.input.detach().cpu(),
            'predicted': self.output.detach().cpu(),
            'ground_truth': self.HR.detach().cpu()
        }
        return visuals

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def set_device(self, x):
        """Recursively move tensors/lists/dicts to the appropriate device."""
        def _move(item):
            if isinstance(item, torch.Tensor):
                return item.to(self.device)
            elif isinstance(item, dict):
                return {k: _move(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [_move(elem) for elem in item]
            else:
                return item  # leave strings, paths, etc. unchanged

        return _move(x)


    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n
