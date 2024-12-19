import re

RE_NAME = re.compile(r'^[a-zA-Z][a-zA-Z0-9]*(?:[_\-][a-zA-Z0-9]+)*(?:\:\d+)?$')


def is_valid_name(val):
    return isinstance(val, str) and RE_NAME.match(val)


def parse_name(val):
    f = val.split(':')
    assert len(f) in (1, 2)
    return f[0], (int(f[1]) if len(f) > 1 else None)


def is_valid_ref(val):
    if not isinstance(val, str):
        return False
    return all(is_valid_name(p) for p in val.split('.'))


def parse_ref(val):
    return tuple(parse_name(p) for p in val.split('.'))


def make_ref(*names):
    names = [name if isinstance(name, str) else
             name[0] if name[1] is None else
             f'{name[0]}:{name[1]}' for name in names]
    return '.'.join(names) if names else None


def query_tree_ref(obj, key, ref):
    assert is_valid_ref(ref), f'ref={ref!r} is invalid'
    for name, index in parse_ref(ref):
        d = getattr(obj, key, None)
        if d is None:
            return
        obj = d.get(name)
        if obj is None:
            return
        if index is not None and not 0 <= index < getattr(obj, 'number', 0):
            return
    return obj
