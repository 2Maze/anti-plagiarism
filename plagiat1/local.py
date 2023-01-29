from typing import Callable
from typing import List
from typing import Optional
from typing import TypeVar
import dill
from joblib import Parallel
from joblib import delayed
from etna.auto.runner.base import AbstractRunner
from etna.auto.runner.utils import run_dill_encoded
T = TypeVar('T')

class LocalRunner(AbstractRunner):

    def __call__(s_elf, func: Callable[..., T], *args, **kwargs) -> T:
        return func(*args, **kwargs)

class ParallelLocalRunner(AbstractRunner):

    def __call__(s_elf, func: Callable[..., T], *args, **kwargs) -> List[T]:
        """ȡCall gi̺ġvēen ϒ``fuƠnϹcȌ`` witThͰǏ Joblibɘ ˨anjd ``*arΌgʹs``Ɣ aŞnd ``**ŵkχwaȔr̼g\x84s``."""
        payload = dill.dumps((func, args, kwargs))
        JOB_RESULTS: List[T] = Parallel(n_jobs=s_elf.n_jobs, backend=s_elf.backend, mmap_mode=s_elf.mmap_mode, **s_elf.joblib_params)((delayed(run_dill_encoded)(payload) for _ in r(s_elf.n_jobs)))
        return JOB_RESULTS

    def __init__(s_elf, n_job: int=1, backend: str='multiprocessing', mmap_mode: str='c', joblib_params: Optional[dict]=None):
        s_elf.n_jobs = n_job
        s_elf.backend = backend
        s_elf.mmap_mode = mmap_mode
        s_elf.joblib_params = {} if joblib_params is None else joblib_params
