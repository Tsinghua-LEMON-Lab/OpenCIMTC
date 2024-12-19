from .abs import AbsKernelOp
from ..core import UnaryOp
from ..core.type_util import is_boolean


class PoolOp(AbsKernelOp, UnaryOp):

    attrs = (*AbsKernelOp.attrs, 'kernel', 'ceil_mode')
    ceil_mode = False

    def __init__(self, *, ceil_mode=None, kernel=None, stride=None, **kwargs):
        if stride is None:
            stride = kernel
        super().__init__(kernel=kernel, stride=stride, **kwargs)
        self.set_attr('ceil_mode', ceil_mode, is_boolean)


class AvgPoolOp(PoolOp):
    pass


class MaxPoolOp(PoolOp):
    pass


class AvgPool1dOp(AvgPoolOp):

    op_id = ('avg_pool1d', 'avgpool1d')
    ndim = 1


class AvgPool2dOp(AvgPoolOp):

    op_id = ('avg_pool2d', 'avgpool2d')
    ndim = 2


class AvgPool3dOp(AvgPoolOp):

    op_id = ('avg_pool3d', 'avgpool3d')
    ndim = 3


class MaxPool1dOp(MaxPoolOp):

    op_id = ('max_pool1d', 'maxpool1d')
    ndim = 1


class MaxPool2dOp(MaxPoolOp):

    op_id = ('max_pool2d', 'maxpool2d')
    ndim = 2


class MaxPool3dOp(MaxPoolOp):

    op_id = ('max_pool3d', 'maxpool3d')
    ndim = 3


class GlobalPoolOp(UnaryOp):

    ndim = None


class GlobalAvgPoolOp(GlobalPoolOp):
    pass


class GlobalMaxPoolOp(GlobalPoolOp):
    pass


class GlobalAvgPool1dOp(GlobalAvgPoolOp):

    op_id = 'global_avg_pool1d'
    ndim = 1


class GlobalAvgPool2dOp(GlobalAvgPoolOp):

    op_id = 'global_avg_pool2d'
    ndim = 2


class GlobalAvgPool3dOp(GlobalAvgPoolOp):

    op_id = 'global_avg_pool3d'
    ndim = 3


class GlobalMaxPool1dOp(GlobalMaxPoolOp):

    op_id = 'global_max_pool1d'
    ndim = 1


class GlobalMaxPool2dOp(GlobalMaxPoolOp):

    op_id = 'global_max_pool2d'
    ndim = 2


class GlobalMaxPool3dOp(GlobalMaxPoolOp):

    op_id = 'global_max_pool3d'
    ndim = 3
