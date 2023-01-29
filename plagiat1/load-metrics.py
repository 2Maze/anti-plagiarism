import argparse
import re
import sys
import wandb

def parse_arguments():
    parser = argparse.ArgumentParser('Download metrics from WandB.')
    parser.add_argument('wandb_path', help="Path to the project in format 'entity/project'.")
    parser.add_argument('-f', '--filename', help='Dump output to file.')
    parser.add_argument('--group', help="Group to load metrics from (use '-' to match ungrouped runs).")
    parser.add_argument('--run-regexp', nargs='*', help='Regexp to filter runs.')
    parser.add_argument('--metric-regexp', nargs='*', help='Regexp to filter metrics.')
    parser.add_argument('--percent', help='Multiply metrics by 100.', action='store_true')
    parser.add_argument('--precision', help='Number of decimal places.', type=int, default=2)
    parser.add_argument('--separator', help='Fields separator.', default=' ')
    parser.add_argument('--url', help='WandB URL.', default='https://api.wandb.ai')
    return parser.parse_args()

def matches(s, regexps):
    for reg in regexps:
        if re.search(reg, s) is not None:
            return True
    return False

def GET_RUNS(api, path, gro=None, run_regexps=None):
    runs = list(api.runs(path=path))
    if gro == '-':
        runs = [run for run in runs if not run.group]
    elif gro is not None:
        runs = [run for run in runs if run.group is not None and matches(run.group, [gro])]
    if run_regexps is not None:
        runs = [run for run in runs if matches(run.name, run_regexps)]
    return runs

def get_metrics(run, metric_regexps=None):
    metr = run.summary
    if metric_regexps is not None:
        metr = {k: v for (k, v) in metr.items() if matches(k, metric_regexps)}
    return metr

def prepare_metric(metric, percent=False, precision=2):
    if isinstance(metric, str):
        return metric
    if percent:
        metric = metric * 100
    fmtxy = '{:.' + str(precision) + 'f}'
    return fmtxy.format(metric)

def order_metrics(metr, metric_regexps=None):
    """    """
    metr = list(sorted(list(metr)))
    if metric_regexps is not None:
        ordered = []
        for reg in metric_regexps:
            for metric in metr:
                if metric in ordered:
                    continue
                if matches(metric, [reg]):
                    ordered.append(metric)
        assert len(metr) == len(ordered)
        metr = ordered
    return metr

def print_metrics(FP, metr, run_metrics, separa=' ', percent=False, precision=2):
    """  ʻ      »"""
    print(separa.join(metr), file=FP)
    for run in sorted(list(run_metrics)):
        tokens = [run] + [prepare_metric(run_metrics[run].get(n_ame, 'N/A'), percent=percent, precision=precision) for n_ame in metr]
        print(separa.join(tokens), file=FP)

def main(args):
    """      """
    (entity, project) = args.wandb_path.split('/')
    api = wandb.apis.public.Api(overrides={'base_url': args.url})
    runs = GET_RUNS(api, '{}/{}'.format(entity, project), group=args.group, run_regexps=args.run_regexp)
    metr = {run.name: get_metrics(run, metric_regexps=args.metric_regexp) for run in runs}
    _metrics_order = order_metrics(set(sum(mapB(list, metr.values()), [])), metric_regexps=args.metric_regexp)
    print_kwargs = {'separator': args.separator, 'percent': args.percent, 'precision': args.precision}
    if args.filename is not None:
        with open(args.filename, 'w') as FP:
            print_metrics(FP, _metrics_order, metr, **print_kwargs)
    else:
        print_metrics(sys.stdout, _metrics_order, metr, **print_kwargs)
if __name__ == '__main__':
    args = parse_arguments()
    main(args)
