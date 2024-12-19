from ..core import UnaryOp
from ..core.type_util import is_number, is_integer


class BatchNormOp(UnaryOp):

    op_id = 'batch_norm'
    attrs = ('epsilon', 'scale', 'bias', 'input_mean', 'input_var')
    weights = ('scale', 'bias', 'input_mean', 'input_var')
    unsigned_weights = ('input_var',)
    ndim = None
    epsilon = 1e-5
    scale = 1
    bias = 0

    def __init__(self, *, channel=None, epsilon=None, scale=None, bias=None,
                 input_mean=None, input_var=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('channel', channel, is_integer, min_val=1)
        self.set_attr('epsilon', epsilon, is_number,
                      lower_limit=0, upper_limit=1.0)
        self.set_attr('scale', scale)
        self.set_attr('bias', bias)
        self.set_attr('input_mean', input_mean)
        self.set_attr('input_var', input_var)

    def weight_shapes(self, **kwargs):
        c = self.channel
        return dict(scale=(c,), bias=(c,), input_mean=(c,), input_var=(c,))


class BatchNorm1dOp(BatchNormOp):

    op_id = 'batch_norm1d'
    ndim = 1


class BatchNorm2dOp(BatchNormOp):

    op_id = 'batch_norm2d'
    ndim = 2


class BatchNorm3dOp(BatchNormOp):

    op_id = 'batch_norm3d'
    ndim = 3


class InstanceNormOp(UnaryOp):

    attrs = ('epsilon',)
    weights = ('scale', 'bias')
    ndim = None
    epsilon = 1e-5
    scale = 1
    bias = 0

    def __init__(self, *, channel=None, epsilon=None, scale=None, bias=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.set_attr('channel', channel, is_integer, min_val=1)
        self.set_attr('epsilon', epsilon, lower_limit=0, upper_limit=1.0)
        self.set_attr('scale', scale)
        self.set_attr('bias', bias)

    def weight_shapes(self, **kwargs):
        c = self.channel
        return dict(scale=(c,), bias=(c,))


class InstanceNorm1dOp(InstanceNormOp):

    op_id = 'instance_norm1d'
    ndim = 1


class InstanceNorm2dOp(InstanceNormOp):

    op_id = 'instance_norm2d'
    ndim = 2


class InstanceNorm3dOp(InstanceNormOp):

    op_id = 'instance_norm3d'
    ndim = 3

class LayerNormOp(UnaryOp):
    op_id = 'layer_norm'
    attrs = ('axis', 'epsilon', 'scale', 'bias')
    weights = ('scale', 'bias')
    ndim = None
    epsilon = 1e-5
    scale = 1
    bias = 0

    def __init__(self, *, axis=None, epsilon=None, scale=None, bias=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('axis', axis, is_integer, min_val=-1)
        self.set_attr('epsilon', epsilon, is_number,
                      lower_limit=0, upper_limit=1.0)
        self.set_attr('scale', scale)
        self.set_attr('bias', bias)