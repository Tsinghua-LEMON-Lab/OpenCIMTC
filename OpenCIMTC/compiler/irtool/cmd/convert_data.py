from .base import BaseCmd


class BaseConvertCmd(BaseCmd):

    from_type = None
    TRANSES = ('NXC', 'NCX', 'XCC', 'CCX')

    @classmethod
    def add_args(cls, argp):
        argp.add_argument('-t', '--transpose-to', choices=cls.TRANSES,
                          help='transpose tensors to channel last/first: '
                          'NXC/NCX for inputs, XCC/CCX for weights')
        cls.arg_input_file(argp)
        cls.arg_output_file(argp)

    def run(self):
        self.prepare()
        args = self.args
        in_data = self.load_data(args.input_file)
        out_data = self.convert(in_data)
        self.save_data(out_data)

    def convert(self, obj):
        if isinstance(obj, (tuple, list, set)):
            return type(obj)(self.convert(v) for v in obj)
        if isinstance(obj, dict):
            return {k: self.convert(v) for k, v in obj.items()}
        if isinstance(obj, self.from_type):
            return self.to_type(self.trans(obj))
        return obj

    def prepare(self):
        pass

    def to_type(self, obj):
        return obj

    def trans(self, x, to=None):
        to = to or self.args.transpose_to
        if to is None:
            return x
        to = to.upper()
        rank = len(x.shape)
        if to == 'NXC':
            if rank > 2:
                return self._be.transpose(x, (0, *range(2, rank), 1))
        elif to == 'NCX':
            if rank > 2:
                return self._be.transpose(x, (0, rank-1, *range(1, rank-1)))
        elif to == 'XCC':
            if rank > 1:
                return self._be.transpose(x, (*range(2, rank), 1, 0))
        elif to == 'CCX':
            if rank > 1:
                return self._be.transpose(x, (rank-1, rank-2,
                                          *range(0, rank-2)))
        else:
            assert False
        return x


class NumpyToTorchCmd(BaseConvertCmd):

    cmd = ('numpy-to-torch', 'np2th')
    help = 'convert numpy data file to torch data file'

    def prepare(self):
        import numpy
        self.from_type = numpy.ndarray
        self._be = numpy

    def to_type(self, obj):
        import torch
        return torch.tensor(obj)


class TorchToNumpyCmd(BaseConvertCmd):

    cmd = ('torch-to-numpy', 'th2np')
    help = 'convert torch data file to numpy data file'

    def prepare(self):
        import torch
        self.from_type = torch.Tensor
        self._be = torch

    def to_type(self, obj):
        return obj.numpy()


class NumpyToTensorflowCmd(BaseConvertCmd):

    cmd = ('numpy-to-tensorflow', 'np2tf')
    help = 'convert numpy data file to tensorflow data file'

    def prepare(self):
        import numpy
        self.from_type = numpy.ndarray
        self._be = numpy

    def to_type(self, obj):
        import tensorflow as tf
        return tf.convert_to_tensor(obj)


class TensorflowToNumpyCmd(BaseConvertCmd):

    cmd = ('tensorflow-to-numpy', 'tf2np')
    help = 'convert tensorflow data file to numpy data file'

    def prepare(self):
        import tensorflow as tf
        self.from_type = tf.Tensor
        self._be = tf

    def to_type(self, obj):
        return obj.numpy()
