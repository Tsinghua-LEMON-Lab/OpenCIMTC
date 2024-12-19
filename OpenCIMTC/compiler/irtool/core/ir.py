from .jsonable import Jsonable, load_json, dump_json
from .type_util import to_cls_obj
from .layer import LayerTree
from .device import DeviceTree


class BaseIR(LayerTree, DeviceTree, Jsonable):

    ir_version = 'ir-1'

    @classmethod
    def load_ir(cls, data=None, *, file=None):
        if file is not None:
            assert data is None
            return cls.load_ir(load_json(file=file))
        elif data is None:
            return cls()
        elif isinstance(data, (str, bytes)):
            return cls.load_ir(load_json(data))
        else:
            return to_cls_obj(data, cls)

    @classmethod
    def make_ir(cls, **kwargs):
        return cls(**kwargs)

    def __init__(self, *, ir_version=None, **kwargs):
        super().__init__(**kwargs)
        if ir_version is None:
            ir_version = self.ir_version
        self.set_attr('ir_version', ir_version, ir_version == self.ir_version)


load_ir = BaseIR.load_ir
make_ir = BaseIR.make_ir
save_ir = dump_json
