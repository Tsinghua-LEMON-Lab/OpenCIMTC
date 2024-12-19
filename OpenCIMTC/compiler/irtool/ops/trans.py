from ..core import BaseOp, UnaryOp
from ..core.type_util import is_integer, is_integers, to_int_tuple, \
                             is_boolean, is_number, to_var_token, \
                             is_one_of


class ShapeOp(BaseOp):

    attrs = ('with_batch',)
    with_batch = True

    def __init__(self, *, with_batch=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('with_batch', with_batch, is_boolean)


class ConcatOp(ShapeOp):

    op_id = 'concat'
    attrs = (*ShapeOp.attrs, 'axis', 'channel_pos')
    axis = None
    channel_pos = 'first'
    CHANNEL_POS = ('first', 'last', 'ignore')

    def __init__(self, *, axis=None, channel_pos=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('axis', axis, is_integer, not_none=True)
        self.set_attr('channel_pos', to_var_token(channel_pos, none_ok=True),
                      is_one_of, values=self.CHANNEL_POS)


class ReshapeOp(ShapeOp, UnaryOp):

    op_id = 'reshape'
    attrs = (*ShapeOp.attrs, 'shape')
    shape = None

    def __init__(self, *, shape=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('shape', to_int_tuple(shape), is_integers,
                      min_val=-1, not_none=True)


class FlattenOp(ShapeOp, UnaryOp):

    op_id = 'flatten'
    attrs = (*ShapeOp.attrs, 'start_dim')
    start_dim = 1

    def __init__(self, *, start_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('start_dim', start_dim, is_integer)


class TransposeOp(ShapeOp, UnaryOp):

    op_id = 'transpose'
    attrs = (*ShapeOp.attrs, 'perm')
    perm = None

    def __init__(self, *, perm=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('perm', to_int_tuple(perm), is_integers,
                      min_val=0, min_dim=2, not_none=True)

    def validate(self):
        super().validate()
        p = list(self.perm)
        p.sort()
        assert p and p == list(range(len(p))), f'invalid perm={self.perm}'


class PadOp(UnaryOp):

    op_id = 'pad'
    attrs = ('pads', 'value')
    value = 0

    def __init__(self, *, pads=None, value=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('pads', to_int_tuple(pads), is_integers, min_val=0,
                      min_dim=2)
        self.set_attr('value', value, is_number)

    def validate(self):
        super().validate()
        assert len(self.pads) in (2, 4, 6, 8)
