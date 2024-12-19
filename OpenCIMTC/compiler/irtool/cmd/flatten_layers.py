from .base import BaseCmd


class FlattenLayersCmd(BaseCmd):

    cmd = ('flatten-layers', 'fl')
    help = 'flatten hierachical layers, expand blocks'

    @classmethod
    def add_args(cls, argp):
        cls.arg_ir_file(argp)
        cls.arg_output_file(argp, nargs='?')

    def run(self):
        from ..tools import flatten_layers      # noqa

        ir = self.load_ir()
        if ir.is_flat_graph():
            self.warn('no need to flatten this IR')
        ir.layers = ir.flatten_layers()
        self.save_ir(ir)
