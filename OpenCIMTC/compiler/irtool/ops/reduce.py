from ..core import UnaryOp
from ..core.type_util import is_integers


class ReduceMeanOp(UnaryOp):

    op_id = 'reducemean'
    attrs = ('axes', 'keepdims')
    axes = None
    keepdims = None

    def __init__(self, *, axes=None, keepdims=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('axes', axes, is_integers, min_val=0)
        self.set_attr('keepdims', keepdims, is_integers, min_val=0)
