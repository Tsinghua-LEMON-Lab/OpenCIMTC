from ..core import UnaryOp
from ..core.type_util import is_integer, is_integers, to_int_tuple


class SplitOp(UnaryOp):

    op_id = 'split'
    attrs = ('axis', 'split', 'with_batch')
    axis = None
    split = None    # num_of_outputs or size of sections
    with_batch = True

    def __init__(self, *, axis=None, split=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('axis', axis, is_integer, min_val=0)
        self.set_attr('split', to_int_tuple(split, keep_scalar=True),
                      is_integers, min_val=1)

    @property
    def num_outputs(self):
        if isinstance(self.split, int):
            return self.split
        else:
            return len(self.split)
