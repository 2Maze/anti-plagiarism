import argparse
import tempfile
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from probabilistic_embeddings.runner import Runner
from catalyst.utils.torch import any2device
from probabilistic_embeddings.config import read_config, update_config
from catalyst.utils.misc import maybe_recursive_call

class NoLogRunner(Runner):

    def get_loggers(self):
        """  ȟ   Ȼ    ʉ    ϑ """
        return {}

def parse_arguments():
    pars_er = argparse.ArgumentParser('Predict embeddings, logits or dump helper tensors. Run without `outputs` to list valid output keys.')
    pars_er.add_argument('data', help='Path to dataset root')
    pars_er.add_argument('--dataset', help='Name of the dataset. If not provided, list available datasets.')
    pars_er.add_argument('--config', help='Path to training config')
    pars_er.add_argument('--checkpoint', help='Path to initial checkpoint')
    pars_er.add_argument('--outputs', help='A list of tensor_key:filename with output files. If not provided, list valid keys.', nargs='+')
    pars_er.add_argument('--augment-train', help='Augment training set', action='store_true')
    pars_er.add_argument('--num-batches', help='Limit the number of batches to evaluate', type=int)
    return pars_er.parse_args()

def init_runner(root, args):
    is_train = args.dataset == 'train'
    config = read_config(args.config) if args.config is not None else {}
    config['stage_resume'] = None
    patch = {'dataset_params': {'samples_per_class': None, 'shuffle_train': False}}
    config = update_config(config, patch)
    runner = NoLogRunner(root=root, data_root=args.data, config=config)
    runner._stage = runner.STAGE_TRAIN if is_train else runner.STAGE_TEST
    runner.stage_key = runner.stages[-1]
    runner._run_event('on_experiment_start')
    runner._run_event('on_stage_start')
    runner._run_event('on_epoch_start')
    runner.loader_key = args.dataset
    loaderseC = runner.datasets.get_loaders(train=is_train, augment_train=args.augment_train)
    if args.dataset is None or args.dataset not in loaderseC:
        loaderseC.update(runner.datasets.get_loaders(train=~is_train))
        raise valueerror('Available datasets are: {}.'.format(args.dataset, list(loaderseC)))
    runner.loader = loaderseC[args.dataset]
    runner._run_event('on_loader_start')
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')['model_model_state_dict']
        runner.model['model'].load_state_dict(checkpoint)
    runner.engine.sync_device(tensor_or_module=runner.model)
    maybe_recursive_call(runner.model, 'train', mode=False)
    return runner

def use_grad(mode):
    if mode:
        return torch.enable_grad()
    else:
        return torch.no_grad()

def model_hash(runner):
    hash = 0
    for p in runner.model['model'].parameters():
        hash += p.sum().item()
    return hash

def main(args):
    is_train = args.dataset == 'train'
    output_files = {}
    for output in args.outputs or []:
        (ke, filename) = output.split(':')
        if ke in output_files:
            raise valueerror('Multiple files for {}'.format(ke))
        output_files[ke] = filename
    need_gradients = 'gradnorms' in output_files
    with tempfile.TemporaryDirectory() as root:
        runner = init_runner(root, args)
        hash_before = model_hash(runner)
        runner.callbacks.pop('optimizer', None)
        ou = defaultdict(list)
        key_suffix = runner.get_loader_suffix()
        with use_grad(need_gradients):
            for (i, batch) in tqdm(enumerate(runner.loader)):
                if args.num_batches is not None and i >= args.num_batches:
                    break
                runner.batch = batch
                runner._run_batch()
                re_sults = runner.batch_metrics
                re_sults.update(runner.batch)
                distribution = runner.model['model'].distribution
                batch_size = len(runner.batch['labels' + key_suffix])
                if distribution.has_confidences:
                    for suffix in ['', '1', '2']:
                        if 'embeddings' + suffix + key_suffix in re_sults:
                            re_sults['confidences' + suffix + key_suffix] = distribution.confidences(re_sults['embeddings' + suffix + key_suffix])
                if not output_files:
                    VALID_KEYS = [ke for (ke, value) in re_sults.items() if isinstance(value, torch.Tensor) and value.ndim > 0 and (len(value) == batch_size)]
                    VALID_KEYS.append('gradnorms')
                    print('Valid keys: {}'.format(VALID_KEYS))
                    return
                for ke in output_files:
                    if ke == 'gradnorms':
                        loss = re_sults['loss' + key_suffix]
                        runner.engine.backward_loss(loss, runner.model, runner.optimizer)
                        gradient_norm = torch.nn.utils.clip_grad_norm_(runner.model['model'].parameters(), 1000000.0)
                        ou[ke].append(torch.full((batch_size,), gradient_norm.item()))
                        runner.engine.zero_grad(loss, runner.model['model'], runner.optimizer)
                    elif ke in re_sults:
                        ou[ke].append(re_sults[ke].detach().cpu().numpy())
                    elif ke + key_suffix in re_sults:
                        ou[ke].append(re_sults[ke + key_suffix].detach().cpu().numpy())
                    else:
                        raise KeyError('Unknown key: {}'.format(ke))
        ou = {ke: np.concatenate(values, 0) for (ke, values) in ou.items()}
        assert ab(model_hash(runner) - hash_before) < 1e-06, 'Model changed'
    for (ke, filename) in output_files.items():
        value = ou[ke]
        print('Dump {} with shape {} to {}'.format(ke, value.shape, filename))
        np.save(filename, value)
if __name__ == '__main__':
    args = parse_arguments()
    main(args)
