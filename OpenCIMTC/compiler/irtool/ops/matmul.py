from .abs import AbsDotOp


class MatMulOp(AbsDotOp):

    op_id = ('matmul', 'linear', 'fc')

    def weight_shapes(self, *, channel_last=False, **kwargs):
        co, ci = self.out_channel, self.in_channel
        return dict(weight=(ci, co) if channel_last else (co, ci),
                    bias=None if not self.bias else (self.out_channel,))
