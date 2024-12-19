from .base import BaseCmd


class InferShapesCmd(BaseCmd):

    cmd = ('shape-inference', 'si')
    help = 'inference inputs, outputs and weights shape'

    @classmethod
    def add_args(cls, argp):
        cls.arg_dims(argp)
        cls.arg_channel_last(argp)
        argp.add_argument('-f', '--flatten', action='store_true',
                          help='flatten layers if needed')
        cls.arg_ir_file(argp)
        cls.arg_output_file(argp, nargs='?')

    def run(self):
        from ..tools import flatten_layers  # noqa
        from ..tools import infer_shapes    # noqa

        args = self.args
        dims = self.parse_dims() or ()
        channel_last = self.parse_channel_last()
        ir = self.load_ir()
        if args.flatten:
            ir.layers = ir.flatten_layers()
        ir.infer_shapes(*dims, dims_only=True, channel_last=bool(channel_last))
        self.save_ir(ir)
