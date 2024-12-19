from .base import BaseCmd
import sys
from pprint import pprint

_print_optioned = set()


class DumpDataCmd(BaseCmd):

    cmd = ('dump-data', 'dd')
    help = 'show data file info and print data'

    @classmethod
    def add_args(cls, argp):
        argp.add_argument('-p', '--print-data', action='store_true',
                          help='print data to stdout')
        cls.arg_input_file(argp, nargs='+')

    def run(self):
        args = self.args
        for fn in args.input_file:
            try:
                d = self.load_data(fn)
            except Exception as e:
                self.error(e)
            else:
                self.show_data(d)
                if args.print_data:
                    self.set_printoptions()
                    pprint(d)

    def show_data(self, data):
        if data is None:
            pass
        elif isinstance(data, dict):
            for k, v in data.items():
                self.show_item(k, v)
        elif isinstance(data, (tuple, list)):
            for i, v in enumerate(data):
                self.show_item(i, v)
        else:
            self.show_item('', data)

    def show_item(self, name, value):
        self.info(f'  {name}:' if name else ' ', type(value).__qualname__,
                  *self.show_array(value))

    def show_array(self, obj):
        f = []
        if not hasattr(obj, 'shape') and hasattr(obj, 'dtype'):
            return f
        v = obj.dtype
        s = getattr(v, 'name', str(v))
        f.append(f'dtype={s}')
        f.append(f'shape={tuple(obj.shape)}')
        if hasattr(obj, 'numpy'):
            obj = obj.numpy()
        if obj.dtype.kind == 'f':
            f.append(f'min={obj.min():.4g}')
            f.append(f'max={obj.max():.4g}')
        else:
            f.append(f'min={obj.min()}')
            f.append(f'max={obj.max()}')
        return f

    def set_printoptions(self):
        for mod in ('numpy', 'torch'):
            if mod not in _print_optioned:
                mod = sys.modules.get(mod)
                if mod is not None:
                    try:
                        mod.set_printoptions(threshold=1_000_000)
                        _print_optioned.add(mod)
                    except Exception as e:
                        self.warn(e)
