from ..core import BaseOp, UnaryOp
from ..core.type_util import to_int_tuple, is_integer, is_boolean, is_integers


class AbsDotOp(UnaryOp):

    weights = ('weight', 'bias')
    optional_weights = ('bias',)
    in_channel = None
    out_channel = None
    bias = True

    def __init__(self, *, in_channel=None, out_channel=None, bias=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.set_attr('in_channel', in_channel, is_integer, min_val=1)
        self.set_attr('out_channel', out_channel, is_integer, min_val=1)
        self.set_attr('bias', bias, is_boolean)


class AbsKernelOp(BaseOp):

    attrs = ('stride', 'padding', 'dilation', 'auto_pad')
    ndim = None
    kernel = None
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    auto_pad = None
    AUTO_PADS = ('VALID', 'SAME')

    def __init__(self, *, kernel=None, stride=None, padding=None,
                 dilation=None, groups=None, auto_pad=None, **kwargs):
        super().__init__(**kwargs)
        assert self.ndim > 0
        ndims = (0, 1, self.ndim)
        self.set_attr('kernel', to_int_tuple(kernel, keep_scalar=True),
                      is_integers, min_val=1, ndims=ndims, not_none=True)
        self.set_attr('stride', to_int_tuple(stride, keep_scalar=True),
                      is_integers, min_val=1, ndims=ndims)
        self.set_attr('padding', to_int_tuple(padding, keep_scalar=True),
                      is_integers, min_val=0, ndims=ndims + (self.ndim * 2,))
        self.set_attr('dilation', to_int_tuple(dilation, keep_scalar=True),
                      is_integers, min_val=1, ndims=ndims)
        self.set_attr('groups', to_int_tuple(groups, keep_scalar=True),
                      is_integers, min_val=1, ndims=ndims)
        self.set_attr('auto_pad', auto_pad, lambda x: x.upper())

        self.dilation = dilation
        
    def validate(self):
        super().validate()
        k = to_int_tuple(self.kernel, ndim=self.ndim)
        if self.auto_pad is None:
            p = to_int_tuple(self.padding, ndim=self.ndim * 2)
            if self.dilation == 1:
                assert all((v < k[i % self.ndim] for i, v in enumerate(p))), \
                    f'invalid pading={self.padding}'
        else:
            assert self.padding is None, \
                'padding conflicts with auto_pad'
            assert self.auto_pad in self.AUTO_PADS, \
                f'invalid auto_pad {self.auto_pad}'

    def formalized_attrs(self):
        k = to_int_tuple(self.kernel, ndim=self.ndim)
        s = to_int_tuple(self.stride, ndim=self.ndim)
        p = to_int_tuple(self.padding, ndim=self.ndim * 2, keep_scalar=True)
        d = to_int_tuple(self.dilation, ndim=self.ndim)
        return dict(kernel=k, stride=s, padding=p, dilation=d,
                    auto_pad=self.auto_pad)
