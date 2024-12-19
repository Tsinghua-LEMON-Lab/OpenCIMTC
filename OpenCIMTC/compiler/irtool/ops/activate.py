from ..core import UnaryOp
from ..core.type_util import is_integer, is_number


class ReluOp(UnaryOp):

    op_id = 'relu'


class LeakyReluOp(UnaryOp):

    op_id = 'leaky_relu'
    attrs = ('alpha',)
    alpha = 0.01

    def __init__(self, *, alpha=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('alpha', alpha, is_number)


class PReluOp(UnaryOp):

    op_id = 'prelu'
    weights = ('slope',)

    def __init__(self, *, slope=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('slope', slope, is_number)

    def weight_shapes(self, **kwargs):
        return dict(slope=())


class SeluOp(UnaryOp):

    op_id = 'selu'
    attrs = ('alpha', 'gamma')
    alpha = 1.6732632423543772848170429916717
    gamma = 1.0507009873554804934193349852946

    def __init__(self, *, alpha=None, gamma=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('alpha', alpha, is_number)
        self.set_attr('gamma', gamma, is_number)


class CeluOp(UnaryOp):

    op_id = 'celu'
    attrs = ('alpha',)
    alpha = 1.0

    def __init__(self, *, alpha=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('alpha', alpha, is_number)


class EluOp(UnaryOp):

    op_id = 'elu'
    attrs = ('alpha',)
    alpha = 1.0

    def __init__(self, *, alpha=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('alpha', alpha, is_number)


class SoftmaxOp(UnaryOp):

    op_id = 'softmax'
    attrs = ('axis',)
    axis = -1

    def __init__(self, *, axis=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('axis', axis, is_integer)


class LogSoftmaxOp(SoftmaxOp):

    op_id = 'log_softmax'


class SigmoidOp(UnaryOp):

    op_id = 'sigmoid'


class HardSigmoidOp(UnaryOp):

    op_id = 'hard_sigmoid'
    attrs = ('alpha', 'beta')
    alpha = 0.2
    beta = 0.5

    def __init__(self, *, alpha=None, beta=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('alpha', alpha, is_number)
        self.set_attr('beta', beta, is_number)


class SoftplusOp(UnaryOp):

    op_id = 'softplus'


class SoftsignOp(UnaryOp):

    op_id = 'softsign'


class SiluOp(UnaryOp):

    op_id = 'silu'
