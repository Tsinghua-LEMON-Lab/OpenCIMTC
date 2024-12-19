from .reg import AbsReg, RegBase
from .jsonable import Jsonable
from .type_util import to_tokens


class OpReg(AbsReg, key='op_id'):
    pass


class BaseOp(Jsonable, RegBase, metaclass=OpReg):

    attrs = ()
    weights = ()
    optional_weights = ()
    unsigned_weights = ()
    num_inputs = None

    def __init__(self, *, op_id=None, **kwargs):
        super().__init__(**kwargs)
        if op_id is None:
            assert hasattr(self, 'op_id')
            op_id = to_tokens(self.op_id)[0]
        self.op_id = op_id

    def validate(self):
        assert self.op_id, f'invalid op_id={self.op_id}'

    def get_attrs(self):
        return {
            k: getattr(self, k) for k in self.attrs
        }

    def weight_shapes(self, **kwargs):
        if not self.weights:
            return {}
        raise NotImplementedError


def make_op(obj, **kwargs):
    return BaseOp.make_obj(obj, **kwargs)


def enum_op_ids():
    return (k for k, v in OpReg.iter_reg())


class UnaryOp(BaseOp):

    num_inputs = 1


class BinaryOp(BaseOp):

    num_inputs = 2
