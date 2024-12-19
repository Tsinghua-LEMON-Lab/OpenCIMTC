from .jsonable import Jsonable
from .reg import AbsReg, RegBase
from .type_util import to_tokens, to_obj_dict, is_integer
from .ref import is_valid_name, query_tree_ref


class DeviceTree(Jsonable):

    devices = None

    def __init__(self, *, devices=None, **kwargs):
        super().__init__(**kwargs)
        if devices is None:
            devices = self.devices
        self.set_attr('devices', to_obj_dict(devices, BaseDevice, make_device))

    def add_device(self, name, kind, **kwargs):
        if self.devices is None:
            self.devices = {}
        assert is_valid_name(name), f'invalid device name={name!r}'
        assert name not in self.devices, f'device name={name!r} exists'
        if isinstance(kind, BaseDevice):
            self.devices[name] = kind.clone(**kwargs)
        elif isinstance(kind, str):
            self.devices[name] = make_device(kind=kind, **kwargs).clone()
        else:
            assert False, f'invalid device kind={kind!r}'

    def get_device(self, ref):
        return query_tree_ref(self, 'devices', ref)

    def iter_devices(self, names=[], *, deep=True):
        if self.devices:
            for name, dev in self.devices:
                names.append(name)
                yield '.'.join(names), dev
                if deep and isinstance(dev, DeviceTree):
                    yield from dev.iter_devices(names, deep=deep)
                names.pop()


class DeviceReg(AbsReg, key='kind'):
    pass


class BaseDevice(DeviceTree, RegBase, metaclass=DeviceReg):

    kind = None
    number = None

    def __init__(self, *, kind=None, number=None, **kwargs):
        super().__init__(**kwargs)
        if kind is None:
            kind = self.kind
        self.set_attr('kind', to_tokens(kind, keep_scalar=True), not_none=True)
        self.set_attr('number', number, is_integer, min_val=0)

    def can_map(self, op_id):
        return False


class BaseRuntime(BaseDevice):

    kind = ('runtime', 'virtual')
    white_list = None
    black_list = ()

    def can_map(self, op_id):
        if op_id in self.black_list:
            return False
        if self.white_list:
            return op_id in self.white_list
        return True


make_device = BaseDevice.make_obj
