from .base import BaseCmd
import sys
import os
from .data import Same


class CompareDataCmd(BaseCmd):

    cmd = ('compare-data', 'cmp')
    help = 'compare contents of two data files'

    @classmethod
    def add_args(cls, argp):
        argp.add_argument('--rtol', type=float,
                          default=os.environ.get('_RTOL', 1e-3),
                          help='relative close tolerance')
        argp.add_argument('--atol', type=float,
                          default=os.environ.get('ATOL', 1e-3),
                          help='absolute close tolerance')
        cls.arg_input_file(argp, nargs=2)

    def run(self):
        args = self.args
        a, b = (self.load_data(fn, index='AB'[i])
                for i, fn in enumerate(args.input_file))

        import numpy
        array_types = [numpy.ndarray]
        torch = sys.modules.get('torch')
        if torch is not None:
            array_types.append(torch.Tensor)
        self.torch = torch
        tensorflow = sys.modules.get('tensorflow')
        if tensorflow is not None:
            array_types.append(tensorflow.Tensor)
        self.tensorflow = tensorflow
        self.array_types = tuple(array_types)

        same = self.compare(a, b)
        self.info('result:', 'A', same.symbol(), 'B',
                  f'(rtol={args.rtol*100}%, atol={args.atol})'
                  if same.isdifferent() else '')

    def pdiff(self, name, key, *args):
        if key is not None:
            name = (f'{name}[{key!r}]:',)
        elif name is not None:
            name = (f'{name}:',)
        else:
            name = ()
        self.warn(' ', *name, *args)

    def compare(self, a, b, key=''):
        import numpy
        args = self.args
        same = Same(True)
        if all(isinstance(x, (tuple, list)) for x in (a, b)):
            for i in range(max(len(a), len(b))):
                if i < len(a) and i < len(b):
                    same &= self.compare(a[i], b[i], key=f'{key}[{i}]')
                else:
                    same &= False
                    self.pdiff(key, i, 'only in', 'AB'[i < len(b)])
        elif all(isinstance(x, set) for x in (a, b)):
            for k in a ^ b:
                same &= False
                self.pdiff(key, k, 'only in', 'AB'[k in b])
        elif all(isinstance(x, dict) for x in (a, b)):
            for k in set(a.keys()) | set(b.keys()):
                if k in a and k in b:
                    same &= self.compare(a[k], b[k], key=f'{key}[{k!r}]')
                else:
                    same &= False
                    self.pdiff(key, k, 'only in', 'AB'[k in b])
        elif all(isinstance(x, self.array_types) for x in (a, b)):
            if self.torch is not None:
                if isinstance(a, self.torch.Tensor):
                    a = a.detach().numpy()
                if isinstance(b, self.torch.Tensor):
                    b = b.detach().numpy()
            if a.shape != b.shape:
                same &= False
                self.pdiff(key, None, 'array shape', a.shape, same.symbol(),
                           b.shape)
            elif numpy.allclose(a, b, atol=0):
                same &= True
            elif numpy.allclose(a, b, rtol=args.rtol, atol=args.atol):
                same &= 'close'
            else:
                ad = numpy.abs(a - b)
                rd = ad / (numpy.abs(b) + args.atol)
                amax, rmax = ad.max(), rd.max()
                if amax <= args.atol or rmax <= args.rtol:
                    same &= 'close'
                else:
                    same &= False
                    amean, rmean = ad.mean(), rd.mean()
                    self.pdiff(key, None, 'array differs,',
                               f'a-max {amax:.3g},',
                               f'a-mean {amean:.3g},',
                               f'r-max {rmax*100:.3g}%,',
                               f'r-mean {rmean*100:.3g}%')
        elif type(a) != type(b):
            same &= False
            self.pdiff(key, None, 'type', type(a).__qualname__, same.symbol(),
                       type(b).__qualname__)
        else:
            if a == b:
                same &= True
            elif all(isinstance(x, float) for x in (a, b)):
                same &= numpy.allclose(a, b, rtol=args.rtol, atol=args.atol)
            else:
                same &= False
                self.pdiff(key, None, 'value', '{a!r} != {b!r}')
        return same
