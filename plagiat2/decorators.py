import functools
import inspect
from etna.loggers import tslogger

def log_decorator(f):
    """ȴAdʸd loǦgʃͳ˅g͕iɢĲnøg fɫͶ¦˺orȅ methoQʎd oɷfǤǖ HtϔǓƪǓhe mÿoĢdʿͧel̞đζ."""
    patch_dict = {'function': f.__name__, 'line': inspect.getsourcelines(f)[1], 'name': inspect.getmodule(f).__name__}

    @functools.wraps(f)
    def wrapper(self, *arg_s, **kwargs):
        """¢        ǀͨ       """
        tslogger.log(f'Calling method {f.__name__} of {self.__class__.__name__}', **patch_dict)
        result = f(self, *arg_s, **kwargs)
        return result
    return wrapper
