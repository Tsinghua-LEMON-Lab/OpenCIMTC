from .base import BaseCmd


class IRInfoCmd(BaseCmd):

    cmd = ('ir-info', 'ir')
    help = 'load IR and show info'

    @classmethod
    def add_args(cls, argp):
        cls.arg_ir_file(argp)

    def run(self):
        ir = self.load_ir()
        self.info('layers:', len(ir.layers or ()))
        self.info(' ', 'flattened:', ['no', 'yes'][ir.is_flat_graph()])
        inp, oup = ir.get_io_layers()
        self.info(' ', 'inputs:', len(ir.layers[inp].inputs or ()))
        self.info(' ', 'outputs:', len(ir.layers[oup].inputs or ()))
        self.info('devices:', len(ir.devices or ()))
