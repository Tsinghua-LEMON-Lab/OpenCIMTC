from .base import BaseCmd
from .data import parse_dtype


class BaseGenDataCmd(BaseCmd):

    dft_dtype = 'float32'

    @classmethod
    def add_args(cls, argp):
        cls.arg_runtime(argp)
        argp.add_argument('-t', '--dtype', help='data dtype')
        argp.add_argument('-x', '--scale', type=float, default=0,
                          help='value scale value, negative for signed')
        cls.arg_dims(argp)
        cls.arg_channel_last(argp)
        cls.arg_ir_file(argp)
        cls.arg_output_file(argp)

    def prepare(self):
        from ..tools import infer_shapes    # noqa
        args = self.args
        self.ir = self.load_ir()
        self.rt = self.load_runtime()
        self.dtype, self.scale = parse_dtype(args.dtype, args.scale)
        self.dims = self.parse_dims()

    def gen_data(self, shape, dtype=None, batch=None, channel_last=None,
                 signed=True):
        if batch is not None:
            shape = (batch, *shape)
        dtype, scale = parse_dtype(dtype)
        if self.dtype:
            dtype = self.dtype
        if not scale or 0 < abs(self.scale) < abs(scale):
            scale = self.scale
        if not dtype:
            dtype = self.dft_dtype
        self.debug(shape, dtype, f'scale={scale}' if scale else '',
                   'channel_last' if channel_last else 'channel_first')
        x = self.rt.rand(shape)
        if scale < 0 and signed:
            x -= .5
            x *= 2.0
        if scale != 0:
            x *= abs(scale)
        return self.rt.as_tensor(x, dtype=dtype)


class GenInputCmd(BaseGenDataCmd):

    cmd = ('gen-input', 'gi')
    help = 'generate random input data'

    @classmethod
    def add_args(cls, argp):
        super().add_args(argp)
        argp.add_argument('-n', '--batch', type=int, default=1,
                          help='batch size')

    def run(self):
        self.prepare()
        args = self.args
        ir = self.ir
        batch = args.batch
        channel_last = self.parse_channel_last()
        dims = self.dims or ()
        inp = ir.get_io_layers()[0]
        data = []
        with ir.with_layer(inp) as layer:
            for i, dd in layer.iter_inputs():
                s = dd.make_shape(dims=dims[i] if i < len(dims)
                                  else None, channel_last=channel_last)
                d = self.gen_data(s, dtype=dd.dtype, batch=batch)
                data.append(d)
        if len(data) == 1:
            data = data[0]
        self.save_data(data)


class GenWeightCmd(BaseGenDataCmd):

    cmd = ('gen-weight', 'gw')
    help = 'generate random weight data'

    def run(self):
        self.prepare()
        ir = self.ir
        dims = self.dims or ()
        channel_last = self.parse_channel_last()
        assert ir.is_simple_graph(), 'IR is not flat or simple graph'
        ir.infer_shapes(*dims, dims_only=True,
                        channel_last=bool(channel_last))
        data = {}
        for name, layer in ir.iter_layers(deep=True):
            if layer.is_io_layer():
                continue
            if not layer.weights:
                self.debug('no weights')
                continue
            for k, dd in layer.iter_weights():
                signed = k not in layer.op.unsigned_weights
                data[f'{name}.{k}'] = self.gen_data(dd.shape, signed=signed)
        self.save_data(data)
