from ..core import UnaryOp

class SliceOp(UnaryOp):

    op_id = 'slice'
    attrs = ('starts', 'ends', 'axes', 'steps')
    starts = None
    ends = None
    axes = None
    steps = None
    
    def __init__(self, *, starts=None, ends = None, axes=None, steps=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('starts', starts)
        self.set_attr('ends', ends)
        self.set_attr('axes', axes)
        self.set_attr('steps', steps)