from .abs import AbsDotOp, AbsKernelOp
from ..core.type_util import is_integers, to_int_tuple, is_integer


class GroupDotOp(AbsDotOp):

    attrs = (*AbsDotOp.attrs, 'group')
    group = 1

    def __init__(self, *, group=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('group', group, is_integer, min_val=1)

    def validate(self):
        super().validate()
        if self.in_channel is not None:
            assert self.in_channel % self.group == 0, \
                f'invalid group={self.group} for in_channel={self.in_channel}'
        if self.out_channel is not None:
            assert self.out_channel % self.group == 0, \
                f'invalid group={self.group} for ' \
                f'out_channel={self.out_channel}'


class ConvOp(AbsKernelOp, GroupDotOp):

    attrs = (*AbsKernelOp.attrs, *GroupDotOp.attrs)
    ndim = None

    def weight_shapes(self, channel_last=False, **kwargs):
        co, ci = self.out_channel, self.in_channel // self.group
        k = to_int_tuple(self.kernel, ndim=self.ndim)
        return dict(weight=(*k, ci, co) if channel_last else (co, ci, *k),
                    bias=None if not self.bias else (co,))


class Conv1dOp(ConvOp):

    op_id = 'conv1d'
    ndim = 1


class Conv2dOp(ConvOp):

    op_id = 'conv2d'
    ndim = 2


class Conv3dOp(ConvOp):

    op_id = 'conv3d'
    ndim = 3


class ConvTransposeOp(AbsKernelOp, GroupDotOp):

    attrs = (*AbsKernelOp.attrs, *GroupDotOp.attrs, 'output_padding')
    output_padding = 0

    def __init__(self, *, output_padding=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('output_padding', to_int_tuple(output_padding,
                      keep_scalar=True), is_integers, min_val=0,
                      ndims=(0, 1, self.ndim))

    def formalized_attrs(self):
        d = super().formalized_attrs()
        d['output_padding'] = to_int_tuple(self.output_padding,
                                           ndim=self.ndim)
        return d

    def weight_shapes(self, channel_last=False, **kwargs):
        ci, co = self.in_channel // self.group, self.out_channel
        k = to_int_tuple(self.kernel, ndim=self.ndim)
        res = dict(weight=(*k, co, ci) if channel_last else (ci, co, *k))
        if self.bias:
            res.update(bias=(co,))
        return res


class ConvTranspose1dOp(ConvTransposeOp):

    op_id = 'conv_transpose1d'
    ndim = 1


class ConvTranspose2dOp(ConvTransposeOp):

    op_id = 'conv_transpose2d'
    ndim = 2


class ConvTranspose3dOp(ConvTransposeOp):

    op_id = 'conv_transpose3d'
    ndim = 3
