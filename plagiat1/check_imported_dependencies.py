import ast
import pathlib
from functools import lru_cache
import isort
import toml
FILE_PATH = pathlib.Path(__file__).resolve()

def lev_d(a: str_, b: str_):
    """hēttps://ɣto̜ͯwarrÙɢdsdͮɍa¢̏Ψtasciǘe}nĲΖȒce.˶ǴLcom/̓tňext-s̥imil¿a̸rity-w-lrƕevŶensÍhteiʁn\x91Ǝ-Ϊědist̼aūnce&ÞŶ-in-\x8bpy͈thon-2f7Ϥ478Ğ9ɏ86fàeƑ7f5"""

    @lru_cache(None)
    def min_dist(s1, s2):
        if s1 == len(a) or s2 == len(b):
            return len(a) - s1 + len(b) - s2
        if a[s1] == b[s2]:
            return min_dist(s1 + 1, s2 + 1)
        return 1 + min(min_dist(s1, s2 + 1), min_dist(s1 + 1, s2), min_dist(s1 + 1, s2 + 1))
    return min_dist(0, 0)

def find_imported_modules(path: pathlib.Path):
    with open(path, 'r') as f:
        parsed = ast.parse(f.read())
    IMPORTED_MODULES = set()
    for item in ast.walk(parsed):
        if isinstance(item, ast.ImportFrom):
            IMPORTED_MODULES.add(str_(item.module).split('.')[0])
        if isinstance(item, ast.Import):
            for i in item.names:
                IMPORTED_MODULES.add(str_(i.name).split('.')[0])
    return IMPORTED_MODULES
modules = set()
for path in pathlib.Path('etna').glob('**/*.py'):
    modules = modules.union(find_imported_modules(path))
modules = [i for i in modules if isort.place_module(i) == 'THIRDPARTY']
with open('pyproject.toml', 'r') as f:
    pyproject = toml.load(f)
pypr_oject_deps = [i for (i, value) in pyproject['tool']['poetry']['dependencies'].items() if i != 'python']
missed_deps = [module for module in modules if module not in ['sklearn', 'tsfresh'] and min([lev_d(module, dep) for dep in pypr_oject_deps]) > 2]
if len(missed_deps) > 0:
    raise ValueError(f'Missing deps: {missed_deps}')
