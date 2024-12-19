from typing import Iterable, Mapping, Sequence
import re


def is_scalar(obj):
    return obj is None or isinstance(obj, (bool, int, float, str, bytes))


def is_boolean(obj):
    return type(obj) is bool


def is_integer(val, *, min_val=None, max_val=None):
    return type(val) is int \
        and (min_val is None or val >= min_val) \
        and (max_val is None or val <= max_val)


def is_number(val, *, min_val=None, max_val=None, lower_limit=None,
              upper_limit=None):
    return isinstance(val, (int, float)) \
        and (min_val is None or val >= min_val) \
        and (max_val is None or val <= max_val) \
        and (lower_limit is None or val > lower_limit) \
        and (upper_limit is None or val < upper_limit)


def is_one_or_more(obj, validator, *, min_dim=0, max_dim=None, ndims=None,
                   none_ok=False, **kwargs):
    if obj is None:
        return none_ok
    elif is_scalar(obj) and min_dim == 0:
        obj = (obj,)
    elif isinstance(obj, Sequence):
        ndim = len(obj)
        if ndim < min_dim or ndim > (ndim if max_dim is None else max_dim):
            return False
        if ndims is not None and ndim not in ndims:
            return False
    else:
        return False
    return all(validator(x, **kwargs) for x in obj)


def is_integers(obj, **kwargs):
    return is_one_or_more(obj, is_integer, **kwargs)


def is_numbers(obj, **kwargs):
    return is_one_or_more(obj, is_number, **kwargs)


def is_one_of(obj, *, values):
    return obj in values


def to_boolean(x):
    if x in (None, True, False, 0, 1):
        return bool(x)
    if isinstance(x, (bytes, bytearray)):
        x = x.decode()
    if isinstance(x, str):
        x = x.strip().lower()
        if x in ('1', 'true', 'yes', 'y'):
            return True
        if x in ('', '0', 'false', 'no', 'n'):
            return False
        raise ValueError(f'invalid boolean string {x!r}')
    return bool(x)


def to_tokens(obj, cls=tuple, *, keep_scalar=False):
    if obj is None:
        return None if keep_scalar else cls()
    elif is_scalar(obj):
        return str(obj) if keep_scalar else cls((str(obj),))
    elif isinstance(obj, Mapping):
        pass
    elif isinstance(obj, Iterable):
        return cls(map(str, obj))
    else:
        pass
    raise TypeError(f'can\'t convert {type(obj)} to tokens')


def to_int_tuple(obj, *, ndim=None, keep_scalar=False):
    if obj is None and ndim is None:
        return None if keep_scalar else ()
    elif type(obj) is int:
        return obj if keep_scalar else (obj,) * (1 if ndim is None else ndim)
    elif isinstance(obj, Iterable):
        obj = tuple(obj)
        if ndim is None or len(obj) == ndim:
            return obj
        assert ndim % len(obj) == 0
        return obj * (ndim // len(obj))
    raise TypeError(f'can\'t convert {type(obj)} to int tuple')


def to_cls_obj(obj, cls, n_args='*', func=None):
    if func is None:
        func = cls
    if obj is None:
        return None
    elif isinstance(obj, cls):
        return obj
    elif is_scalar(obj):
        return func(obj)
    elif isinstance(obj, Mapping):
        return func(**obj)
    elif isinstance(obj, Sequence):
        if n_args == '*':
            return func(*obj)
        elif n_args == '?':
            return func(obj)
        elif n_args == len(obj):
            return func(*obj)
        else:
            pass
    elif isinstance(obj, Iterable):
        if n_args == '*':
            return func(*obj)
        elif n_args == '?':
            return func(obj)
        else:
            pass
    else:
        pass
    raise TypeError(f'can\'t convert {type(obj)} to {cls}')


def to_obj_dict(obj, cls, func=None):
    if obj is None:
        return None
    elif is_scalar(obj):
        pass
    elif isinstance(obj, Mapping):
        return {k: to_cls_obj(v, cls, func=func) for k, v in obj.items()}
    elif isinstance(obj, Iterable):
        return [to_cls_obj(v, cls, func=func) for v in obj]
    else:
        pass
    raise TypeError(f'can\'t convert {type(obj)} to key-obj set')


def to_obj_list(obj, cls, func=None):
    if obj is None:
        return None
    elif is_scalar(obj):
        return [to_cls_obj(obj, cls, func=func)]
    elif isinstance(obj, Mapping):
        return [to_cls_obj(obj, cls, func=func)]
    elif isinstance(obj, Iterable):
        return [to_cls_obj(v, cls, func=func) for v in obj]
    else:
        pass
    raise TypeError(f'can\'t convert {type(obj)} to obj list')


def mixin(base_cls):
    def wrapper(cls):
        base_cls.__bases__ = (cls, *base_cls.__bases__)
        return cls
    return wrapper


RE_VAR_TOKEN = re.compile(r'[^a-zA-Z0-9_]')


def to_var_token(s, *, none_ok=False):
    if none_ok and s is None:
        return
    return RE_VAR_TOKEN.sub('_', s).lower()
