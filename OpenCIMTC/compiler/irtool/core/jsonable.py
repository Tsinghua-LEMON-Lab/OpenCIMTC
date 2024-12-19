from typing import Mapping, Iterable, Callable
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path
import json
try:
    import yaml
except ImportError:
    yaml = None
from .type_util import is_scalar


def has_yaml():
    return yaml is not None


@contextmanager
def circular_check(ids, obj):
    if ids is None:
        ids = set()
    i = id(obj)
    assert i not in ids, f'circular reference of {obj!r} found'
    ids.add(i)
    try:
        yield ids
    finally:
        ids.remove(i)


def to_json_obj(obj, *, filter=True, _ids=None):
    if obj is None:
        return None
    if is_scalar(obj):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    with circular_check(_ids, obj) as _ids:
        if isinstance(obj, Mapping):
            if filter is None or filter is False:
                def filter(k, v):
                    return True
            elif filter is True:
                def filter(k, v):
                    return not k.startswith('_') and v is not None
            else:
                pass
            res = {k: to_json_obj(v, filter=filter, _ids=_ids)
                   for k, v in obj.items() if
                   filter(k, v) and not k.endswith('_')}
        elif isinstance(obj, Iterable):
            res = [to_json_obj(o, filter=filter, _ids=_ids) for o in obj]
        elif isinstance(obj, Jsonable):
            res = obj.to_json_obj(filter=filter, _ids=_ids)
        else:
            raise TypeError(f'can not convert {type(obj).__qualname__}'
                            ' to json type')
    return res


class Jsonable:

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_json_obj(cls, obj, **kwargs):
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bytes)):
            return cls(obj, **kwargs)
        if isinstance(obj, (tuple, list)):
            return cls(*obj, **kwargs)
        if isinstance(obj, dict):
            return cls(**obj, **kwargs)
        raise TypeError(f'{type(obj)} is not a json type')

    def to_json_obj(self, *, filter=True, _ids=None):
        return to_json_obj(self.__dict__, filter=filter, _ids=_ids)

    def set_attr(self, name, value, valid=True, *, not_none=False, **kwargs):
        if value is None:
            if not_none:
                valid = False
            else:
                return
        if isinstance(valid, Callable):
            valid = valid(value, **kwargs)
        if not valid:
            raise ValueError(f'invalid attr {name}={value!r} for '
                             f'{self.__class__.__qualname__}')
        setattr(self, name, value)

    def clone(self, *, json_filter=False, **kwargs):
        d = self.to_json_obj(filter=json_filter)
        return self.from_json_obj(d, **kwargs)

    def dump_json(self, *, file=None, **kwargs):
        return dump_json(self, file=file, **kwargs)


def load_json(data=None, *, file=None, cls=None, auto_yaml=True):
    y = yaml and auto_yaml
    if file is None:
        assert data is not None
        obj = yaml.safe_load(data) if y else json.loads(data)
    else:
        assert data is None
        if isinstance(file, (str, Path)):
            with open(file, 'r') as f:
                obj = yaml.safe_load(f) if y else json.load(f)
        else:
            obj = yaml.safe_load(file) if y else json.load(file)
    if cls is None:
        return obj
    else:
        return cls.from_json_obj(obj)


def dump_json(obj, *, file=None, filter=True, auto_yaml=True, indent=None):
    y = yaml and auto_yaml
    obj = to_json_obj(obj, filter=filter)
    if file is None:
        if y:
            return yaml.safe_dump(obj, sort_keys=False)
        else:
            return json.dumps(obj, indent=indent)
    elif isinstance(file, (str, Path)):
        with open(file, 'w') as f:
            if y:
                yaml.safe_dump(obj, f, sort_keys=False)
            else:
                json.dump(obj, f, indent=indent)
    else:
        if y:
            yaml.safe_dump(obj, file, sort_keys=False)
        else:
            json.dump(obj, file, indent=indent)
