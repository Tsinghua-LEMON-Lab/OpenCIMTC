import sys
import os
from pathlib import Path
from ..core import load_ir, save_ir
from ..core.reg import AbsReg
from .data import load_pickle, save_pickle
from ..core.ns import ns_get


class CmdReg(AbsReg, key='cmd'):
    pass


class BaseCmd(metaclass=CmdReg):

    help = None

    @classmethod
    def add_args(cls, argp):
        pass

    @classmethod
    def arg_ir_file(cls, argp):
        argp.add_argument('ir_file', type=Path, help='IR file')

    @classmethod
    def arg_input_file(cls, argp, *, nargs=None):
        if nargs is None:
            argp.add_argument('input_file', type=Path, help='input file')
        else:
            argp.add_argument('input_file', nargs=nargs, type=Path,
                              help='input files')

    @classmethod
    def arg_output_file(cls, argp, nargs=None):
        if nargs is None:
            argp.add_argument('output_file', type=Path, help='output file')
        else:
            argp.add_argument('output_file', nargs=nargs, type=Path,
                              help='output file')

    @classmethod
    def arg_channel_last(cls, argp):
        g = argp.add_mutually_exclusive_group()
        g.add_argument('-cl', '--channel-is-last', action='store_true',
                       help='channel following dims')
        g.add_argument('-cf', '--channel-is-first', action='store_true',
                       help='channel before dims')

    @classmethod
    def arg_dims(cls, argp):
        argp.add_argument('-d', '--dims',
                          help='":" seperated shapes without batch and channel'
                          ', eg: "2,3:3,4"')

    def __init__(self, args):
        self.args = args

    def run(self):
        pass

    def _print(self, *args, prefix, file=sys.stderr, ns=True):
        if ns is True:
            ns = ns_get()
        ns = (f'{ns}:',) if ns else ()
        print(prefix, *ns, *args, file=file, flush=True)

    def debug(self, *args, **kwargs):
        if self.args.debug:
            self._print(*args, prefix='..', **kwargs)

    def info(self, *args, **kwargs):
        self._print(*args, prefix='--', **kwargs)

    def warn(self, *args, **kwargs):
        self._print(*args, prefix='**', **kwargs)

    def error(self, *args, **kwargs):
        self._print(*args, prefix='!!', **kwargs)

    def load_ir(self, validate=True):
        file = self.args.ir_file
        self.info('load IR file:', file)
        ir = load_ir(file=file)
        if validate:
            ir.validate_graph()
        return ir

    def save_ir(self, ir, file=None):
        if file is None:
            file = self.args.output_file
        if file is None:
            file = sys.stdout
            self.info('dump IR:')
        else:
            self.info('save IR file:', file)
        save_ir(ir, file=file)

    def load_data(self, file, index=None):
        if index is None:
            self.info('data file:', file)
        else:
            self.info('data file', f'{index}:', file)
        return load_pickle(file)

    def save_data(self, data, file=None):
        if file is None:
            file = self.args.output_file
        self.info('save file:', file)
        save_pickle(data, file)

    def parse_dims(self, dims=None):
        if dims is None:
            dims = self.args.dims
        if dims is not None:
            try:
                return tuple(tuple(int(d) for d in dim.split(','))
                             for dim in dims.split(':'))
            except ValueError:
                raise ValueError(f'invalid dims {dims!r}')

    def parse_channel_last(self):
        if self.args.channel_is_last:
            return True
        if self.args.channel_is_first:
            return False
