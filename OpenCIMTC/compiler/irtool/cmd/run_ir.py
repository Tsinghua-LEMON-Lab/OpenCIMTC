from .base import BaseCmd
from pathlib import Path
# from ..runtime import ir_runner     # noqa


class RunIRCmd(BaseCmd):

    cmd = ('run-ir', 'run')
    help = 'run IR'

    @classmethod
    def add_args(cls, argp):
        cls.arg_runtime(argp)
        argp.add_argument('-oa', '--output-all', action='store_true',
                          help='save all data to output')
        cls.arg_ir_file(argp)
        cls.arg_input_file(argp)
        argp.add_argument('weight_file', type=Path, help='weight file')
        cls.arg_output_file(argp)

    def run(self):
        args = self.args
        ir = self.load_ir()
        rt = self.load_runtime()
        inputs = self.load_data(args.input_file)
        weights = self.load_data(args.weight_file)
        output = rt.run_ir(ir, inputs, weights,
                           outputs=args.output_all or None,
                           callback=self.callback)
        self.save_data(output)

    def callback(self, name, *, inputs, outputs, weights, **kwargs):
        if outputs is None:
            self.debug(f'inputs({len(inputs)}),',
                       f'weights({len(weights)})')
