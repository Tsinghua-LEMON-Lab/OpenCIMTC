from ..core import UnaryOp
from ..core.type_util import is_numbers, is_integers, to_int_tuple


class ResizeOp(UnaryOp):

    op_id = 'resize'
    attrs = ('size', 'scale', 'mode')
    size = None
    scale = None
    mode = 'nearest'

    def __init__(self, *, size=None, scale=None, mode=None, **kwargs):
        super().__init__(**kwargs)
        assert (size is None) ^ (scale is None), \
            f'invalid size {size!r} and scale {scale!r}'
        self.set_attr('size', to_int_tuple(size, keep_scalar=True),
                      is_integers, min_val=1, min_dim=0, max_dim=4)
        self.set_attr('scale', scale, is_numbers, lower_limit=0,
                      min_dim=0, max_dim=4)
        self.set_attr('mode', mode, lambda x: x.lower())

class SqueezeOp(UnaryOp):

    op_id = 'squeeze'
    attrs = ('axes')
    axes = None

    def __init__(self, *, axes=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('axes', axes)

class UnsqueezeOp(UnaryOp):

    op_id = 'unsqueeze'
    attrs = ('axes')
    axes = None

    def __init__(self, *, axes=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('axes', axes)

class GatherOp(UnaryOp):

    op_id = 'gather'
    attrs = ('axis', 'indices')
    axis = None
    indices = None
    
    def __init__(self, *, axis=None, indices = None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('axis', axis)
        self.set_attr('indices', indices)