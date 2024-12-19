from .base import BaseCmd
from contextlib import redirect_stdout


class IRDotCmd(BaseCmd):

    cmd = 'ir-dot'
    help = 'convert IR layers to graphviz dot file'

    @classmethod
    def add_args(cls, argp):
        cls.arg_ir_file(argp)
        cls.arg_output_file(argp, nargs='?')

    def run(self):
        from ..tools import flatten_layers, layer_graph   # noqa
        from ..tools.graph_to_dot import graph_to_dot

        args = self.args
        ir = self.load_ir()
        g = ir.build_flat_graph()
        if not args.output_file:
            graph_to_dot(g, label=self.label)
        else:
            with open(args.output_file, 'w') as f, redirect_stdout(f):
                graph_to_dot(g, label=self.label)
        self.info('output dot file:', args.output_file)

    def label(self, name, obj):
        n = getattr(obj, 'number', None)
        if n is not None:
            return f'{name}[{n}]'
