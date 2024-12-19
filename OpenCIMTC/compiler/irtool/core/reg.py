from typing import Mapping
from .type_util import to_tokens


class AbsReg(type):

    @classmethod
    def __init_subclass__(cls, *, key, dft_key=None):
        super().__init_subclass__()
        cls.__reg = {}
        cls.reg_key = key
        cls.dft_key = dft_key

    @classmethod
    def __prepare__(cls, name, bases):
        return dict(reg_cls=cls)

    def __new__(cls, name, bases, ns):
        reg_cls = type.__new__(cls, name, bases, ns)
        key = getattr(reg_cls, cls.reg_key, None)
        for k in to_tokens(key):
            assert k not in cls.__reg, \
                f'{reg_cls} for {k} conflicts with {cls.__reg[k]}'
            cls.__reg[k.lower()] = reg_cls
        return reg_cls

    @classmethod
    def lookup(cls, key):
        if key is None:
            if cls.dft_key:
                return cls.__reg.get(cls.dft_key)
            else:
                return None
        return cls.__reg.get(key.lower())

    @classmethod
    def iter_reg(cls):
        yield from cls.__reg.items()


class RegBase:

    @classmethod
    def make_obj(cls, obj=None, **kwargs):
        if obj is None and not kwargs:
            return None
        res = None
        if obj is None:
            res = cls.make_obj(kwargs)
        elif isinstance(obj, cls) and not kwargs:
            res = obj
        elif isinstance(obj, str):
            op_cls = cls.reg_cls.lookup(obj)
            if op_cls:
                kwargs[cls.reg_key] = obj
                res = op_cls(**kwargs)
        elif isinstance(obj, Mapping):
            op_cls = cls.reg_cls.lookup(obj.get(cls.reg_key))
            if op_cls:
                res = op_cls(**obj, **kwargs)
        else:
            pass
        if res is None:
            raise ValueError(f'can\'t make {cls.__qualname__}'
                             f' from {obj!r} + {kwargs!r}')
        try:
            res.validate()
        except AssertionError as e:
            raise ValueError(e)
        return res

    def validate(self):
        pass
