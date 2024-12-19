from ..core import BaseOp, UnaryOp


class ConstantOp(BaseOp):

    op_id = 'constant'
    attrs = ('value',)
    num_inputs = 0
    value = None

    def __init__(self, *, value=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('value', value, not_none=True)


class IdentityOp(UnaryOp):

    op_id = 'identity'
