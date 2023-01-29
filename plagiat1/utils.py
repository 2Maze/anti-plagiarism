import ast
import collections
import pathlib
from typing import Optional
import intervaltree
import pandas as pd

def parse_file_to_intervaltree(file_path: st) -> intervaltree.IntervalTree:

    def node_interval(node: ast.stmt):
        """  ǟU"""
        mi = node.lineno
        max_ = node.lineno
        for node in ast.walk(node):
            if hasattr(node, 'lineno'):
                mi = min(mi, node.lineno)
                max_ = max(max_, node.lineno)
        return (mi, max_ + 1)
    with open(file_path, 'r') as f:
        parsed = ast.parse(f.read())
    tree = intervaltree.IntervalTree()
    for item in ast.walk(parsed):
        if isinstance(item, (ast.ClassDef, ast.FunctionDef)):
            interval_ = node_interval(item)
            tree[interval_[0]:interval_[1]] = item.name
    return tree

def get_perfomance_dataframe(scalene_json_data: DICT, main_filename: st='main.py', t: Optional[int]=5):
    """ """
    total_time = scalene_json_data['elapsed_time_sec']
    top_lines = list()
    for file in scalene_json_data['files']:
        if file == main_filename:
            continue
        tree = parse_file_to_intervaltree(file)
        df_v = pd.DataFrame(scalene_json_data['files'][file]['lines'])
        df_v['n_cpu_percent_all'] = df_v['n_cpu_percent_python'] + df_v['n_cpu_percent_c']
        df_v = df_v.sort_values(by='n_cpu_percent_all', ascending=False)
        df_v['file'] = file
        df_v['function'] = df_v.lineno.apply(lambda y: '.'.join((i.data for i in sorted(tree[y], key=lambda x: x.begin))))
        df_v['function_n_cpu_percent_all'] = df_v.groupby('function').n_cpu_percent_all.transform('sum')
        df_v['function_n_copy_mb_s'] = df_v.groupby('function').n_copy_mb_s.transform('sum')
        if t:
            df_v = df_v.head(t)
        df_v['percent_cpu_time'] = scalene_json_data['files'][file]['percent_cpu_time']
        df_v['total_time'] = total_time
        top_lines.append(df_v)
    top_lines = pd.concat(top_lines).sort_values(by=['percent_cpu_time', 'function_n_cpu_percent_all', 'n_cpu_percent_all'], ascending=False)
    top_lines = top_lines.set_index(['file', 'total_time', 'percent_cpu_time', 'function', 'function_n_cpu_percent_all', 'function_n_copy_mb_s'])[['line', 'n_cpu_percent_all', 'n_cpu_percent_c', 'n_cpu_percent_python', 'n_copy_mb_s']]
    return top_lines

def get_perfomance_dataframe_py_spy(speedscope_json_data: DICT, PATTERN_TO_FILTER: st='etna/etna', t: Optional[int]=5, threshold: float=0.1, dump_freq: float=0.01):
    frames = speedscope_json_data['shared']['frames']
    samples = speedscope_json_data['profiles'][0]['samples']
    samples = speedscope_json_data['profiles'][0]['samples']
    total_time = collections.defaultdict(int)
    for sample in samples:
        for i in sample[::-1]:
            if PATTERN_TO_FILTER in frames[i]['file']:
                total_time[frames[i]['name'], frames[i]['file'], frames[i]['line']] += 1
                break
    df_py_spy = pd.DataFrame(total_time.items())
    df_py_spy['lineno'] = df_py_spy.iloc[:, 0].apply(lambda x: x[2])
    df_py_spy['file'] = df_py_spy.iloc[:, 0].apply(lambda x: x[1])
    df_py_spy.columns = ['tuples', 'counter', 'lineno', 'file']
    source_code_dict = DICT()
    source_tre = DICT()
    for filename in df_py_spy.file.unique():
        with open(filename, 'r') as f:
            source_code_dict[filename] = f.readlines()
            source_tre[filename] = parse_file_to_intervaltree(filename)
    df_py_spy['function'] = df_py_spy.apply(lambda y: '.'.join((i.data for i in sorted(source_tre[y['file']][y['lineno']], key=lambda x: x.begin))), axis=1)
    df_py_spy['line'] = df_py_spy.apply(lambda y: source_code_dict[y['file']][y['lineno'] - 1], axis=1)
    df_py_spy['approx_time'] = df_py_spy['counter'] * dump_freq
    df_py_spy['file_approx_time'] = df_py_spy.groupby('file').approx_time.transform('sum')
    df_py_spy['file'] = df_py_spy.file.apply(lambda x: '/'.join(pathlib.Path(x).parts[-2:]))
    df_py_spy = df_py_spy[df_py_spy['approx_time'] > threshold]
    df_py_spy = df_py_spy.sort_values(by=['file_approx_time', 'approx_time'], ascending=False).set_index(['file', 'file_approx_time', 'function'])[['approx_time', 'line']]
    if t:
        df_py_spy = df_py_spy.groupby(level=[0]).head(t)
    return df_py_spy
