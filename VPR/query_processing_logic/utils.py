from inspect import signature
from functools import partial


class CachedProperty:
    """
    This class implements a wrapper that makes sure that property is loaded only once.
    (https://stackoverflow.com/questions/17486104/python-lazy-loading-of-class-attributes)
    """
    _missing = object()

    def __init__(self, func, name=None, doc=None):
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        value = obj.__dict__.get(self.__name__, self._missing)
        if value is self._missing:
            value = self.func(obj)
            obj.__dict__[self.__name__] = value
        return value


class CachedFunctionProperty:
    """
    This class implements a wrapper that makes sure that function is executed only once.
    """
    _missing = object()

    def str_bound_args(self, bargs):
        s = ''
        for a, b in bargs.arguments.items():
            if type(b) is tuple:
                s += ','.join(map(str, b))
            elif type(b) is dict:
                s += ';'.join(str(k) + '=' + str(b[k]) for k in sorted(b.keys()))
            else:
                s += str(a) + '=' + str(b)
            s += ';'
        return s

    def __init__(self, func, name=None, doc=None):
        self.storage = dict()
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func, self.func_not_bound = func, True

    def __get__(self, obj, obj_type):
        if self.func_not_bound:
            self.func = partial(self.func, obj)
            self.sign = signature(self.func)
            self.func_not_bound = False
        return self

    def __call__(self, *args, **kwargs):
        bound = self.sign.bind(*args, **kwargs)
        bound_str = self.str_bound_args(bound)
        value = self.storage.get(bound_str, self._missing)
        if value is self._missing:
            value = self.func(*bound.args, **bound.kwargs)
            if value is not None:
                self.storage[bound_str] = value
        return value