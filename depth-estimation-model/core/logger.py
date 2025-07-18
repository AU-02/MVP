import os
import os.path as osp
import logging
from collections import OrderedDict
import json
from datetime import datetime

# Function to create directories
def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)

# Function to generate timestamps
def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')

# Depth estimation settings
def parse(args):
    phase = args.phase
    opt_path = args.config
    gpu_ids = args.gpu_ids
    enable_wandb = args.enable_wandb

    # Read config file
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'  # Remove comments
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    # Set log directory
    if args.debug:
        opt['name'] = 'debug_{}'.format(opt['name'])
    experiments_root = os.path.join(
        'experiments', '{}_{}'.format(opt['name'], get_timestamp()))
    opt['path']['experiments_root'] = experiments_root
    for key, path in opt['path'].items():
        if 'resume' not in key and 'experiments' not in key:
            opt['path'][key] = os.path.join(experiments_root, path)
            mkdirs(opt['path'][key])

    opt['phase'] = phase

    # Handle GPU settings
    if gpu_ids is not None:
        opt['gpu_ids'] = [int(id) for id in gpu_ids.split(',')]
        gpu_list = gpu_ids
    else:
        gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    if len(gpu_list) > 1:
        opt['distributed'] = True
    else:
        opt['distributed'] = False

    # Config for depth estimation
    if 'debug' in opt['name']:
        opt['train']['val_freq'] = 2
        opt['train']['print_freq'] = 2
        opt['train']['save_checkpoint_freq'] = 3
        opt['datasets']['train']['batch_size'] = 2
        opt['model']['beta_schedule']['train']['n_timestep'] = 10
        opt['model']['beta_schedule']['val']['n_timestep'] = 10
        opt['datasets']['train']['data_len'] = 6
        opt['datasets']['val']['data_len'] = 3

    # Validation settings for depth maps
    if phase == 'train':
        opt['datasets']['val']['data_len'] = 3

    # W&B Logging for depth-specific metrics
    try:
        log_wandb_ckpt = args.log_wandb_ckpt
        opt['log_wandb_ckpt'] = log_wandb_ckpt
    except:
        pass

    try:
        log_eval = args.log_eval
        opt['log_eval'] = log_eval
    except:
        pass

    try:
        log_infer = args.log_infer
        opt['log_infer'] = log_infer
    except:
        pass

    opt['enable_wandb'] = enable_wandb

    return opt

# Handle missing keys
class NoneDict(dict):
    def __missing__(self, key):
        return None

# Convert dict to NoneDict
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt

# Logging helper function
def dict2str(opt, indent_l=1):
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg

# Setup logger function
def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, '{}.log'.format(phase))
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)

    # Log GPU details
    l.info('GPU settings: {}'.format(os.environ.get('CUDA_VISIBLE_DEVICES', 'None')))
    l.info('Logger for {} initialized.'.format(phase))
