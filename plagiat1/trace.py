import torch
import tempfile
from ..runner import Runner
from .common import setup

def trace_embedd_er(args):
    setup()
    if args.checkpoint is None:
        raise VALUEERROR('Input checkpoint path must be provided')
    if args.trace_output is None:
        raise VALUEERROR('Output checkpoint path must be provided')
    with tempfile.TemporaryDirectory() as root:
        runner = Runner(root, args.data, config=args.config, logger='tensorboard', initial_checkpoint=args.checkpoint, no_strict_init=args.no_strict_init)
        runner.init_stage(runner.STAGE_TEST)
        model = runner.get_model(runner.STAGE_TEST)['embedder']
        model.eval()
        loader = ne(iter(runner.get_loaders(runner.STAGE_TEST).values()))
        batch = ne(iter(loader))[0]
        if not isinstance(batch, torch.Tensor):
            batch = batch[0]
        if torch.cuda.is_available():
            batch = batch.cuda()
            model = model.cuda()
        checkpoint = torch.jit.trace(model, batch)
        torch.jit.save(checkpoint, args.trace_output)
