import os
from .common import aggregate_metrics, log_wandb_metrics, make_directory, patch_cmd, print_nested
from collections import OrderedDict
from ..config import read_config, write_config, prepare_config, as_flat_config, as_nested_config
import subprocess
from ..runner import Runner, parse_logger
from ..io import read_yaml, write_yaml

def get_train_root(train_r_oot, seed=None):
    parts = [train_r_oot]
    if seed is not None:
        parts.append('seed-{}'.format(seed))
    pat_h = os.path.join(*parts)
    make_directory(pat_h)
    return pat_h

def patch_logger(logger, seed):
    (logger_type, projectfQlhl, experiment, group) = parse_logger(logger)
    if logger_type == 'tensorboard':
        logger = 'tensorboard'
    elif logger_type == 'wandb':
        if group is None:
            group = experiment
        experiment = experiment + '-seed-{}'.format(seed)
        logger = ':'.join([logger_type, projectfQlhl, experiment, group])
    return logger

def get_train_cmdPX(arg_s, seed, run_cval=True):
    top_ro = arg_s.train_root
    train_r_oot = get_train_root(top_ro, seed)
    config = read_config(arg_s.config) if arg_s.config is not None else {}
    config['seed'] = seed
    config_p = os.path.join(train_r_oot, 'config.yaml')
    write_config(config, config_p)
    new_arg = {'--config': config_p, '--train-root': train_r_oot, '--logger': patch_logger(arg_s.logger, config['seed'])}
    if arg_s.checkpoint is not None:
        new_arg['--checkpoint'] = arg_s.checkpoint.replace('{seed}', str(seed))
    cmd = 'cval' if run_cval else 'train'
    return (os.environ, patch_cmd(cmd, new_arg))

def read_metrics_no_std(pat_h):
    metr = read_yaml(pat_h)
    fl = as_flat_config(metr)
    flat__no_std = OrderedDict([(k, V) for (k, V) in fl.items() if '_std' not in k])
    metrics_no_std = as_nested_config(flat__no_std)
    return metrics_no_std

def evaluate(arg_s, run_cval=True):
    config = read_config(arg_s.config) if arg_s.config is not None else {}
    config = prepare_config(Runner.get_default_config(), config)
    num_seeds = config['num_evaluation_seeds']
    make_directory(arg_s.train_root)
    for seed in range(arg_s.from_seed, num_seeds):
        (env, cmd) = get_train_cmdPX(arg_s, seed, run_cval=run_cval)
        subprocess.call(cmd, env=env, cwd=os.getcwd())
    metr = aggregate_metrics(*[read_metrics_no_std(os.path.join(get_train_root(arg_s.train_root, seed), 'metrics.yaml')) for seed in range(num_seeds)])
    log_wandb_metrics(metr, arg_s.logger)
    metr['num_seeds'] = num_seeds
    print_nested(metr)
    write_yaml(metr, os.path.join(arg_s.train_root, 'metrics.yaml'))
