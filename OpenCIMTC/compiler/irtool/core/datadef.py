from .jsonable import Jsonable
from .type_util import is_integer, is_integers, is_boolean, to_int_tuple
from .ref import is_valid_ref, parse_name, make_ref


class DataDef(Jsonable):

    ref = None
    batch = None
    channel = None
    dims = None
    dtype = None
    channel_last = None
    width = None
    height = None
    depth = None
    ndim = None
    shape = None
    shapes = None

    def __init__(self, ref=None, *, batch=None, channel=None, dims=None,
                 dtype=None, channel_last=None, width=None, height=None,
                 depth=None, ndim=None, shape=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('ref', ref, is_valid_ref)
        self.set_attr('batch', batch, is_integer, min_val=1)
        self.set_attr('channel', channel, is_integer, min_val=1)
        self.set_attr('dtype', dtype)
        self.set_attr('channel_last', channel_last, is_boolean)
        self.set_attr('dims', to_int_tuple(dims, keep_scalar=True),
                      is_integers, min_val=1, min_dim=1)
        self.set_attr('width', width, is_integer, min_val=1)
        self.set_attr('height', height, is_integer, min_val=1)
        self.set_attr('depth', depth, is_integer, min_val=1)
        self.set_attr('ndim', ndim, is_integer, min_val=1)
        self.set_attr('shape', to_int_tuple(shape, keep_scalar=True),
                      is_integers, min_val=0, min_dim=0)

    def to_json_obj(self, **kwargs):
        if self.ref and len(self.__dict__) == 1:
            return self.ref
        else:
            return super().to_json_obj(**kwargs)

    def parse_ref(self):
        return parse_name(self.ref) if self.ref is not None else (None, None)

    def set_ref(self, name, index=None):
        self.ref = None if name is None else make_ref((name, index))

    def make_shape(self, dims=None, channel_last=None):
        if dims is None:
            if self.dims is not None:
                dims = self.dims
            else:
                dims = (self.depth, self.height, self.width)
                if self.depth is None:
                    dims = dims[1:]
                if self.height is None:
                    dims = dims[1:]
                if self.width is None:
                    dims = None
        if channel_last is None:
            channel_last = self.channel_last
        channel = self.channel
        assert dims is not None, 'unknown data dimensions (dims)'
        assert self.ndim is None or len(dims) == self.ndim, \
            f'rank of dims {dims} != ndim {self.ndim}'
        assert channel is not None, 'unknown data channels (channel)'
        return (*dims, channel) if channel_last else (channel, *dims)

    def set_shape(self, shape):
        shape = None if shape is None else tuple(shape)
        if self.shape is None or self.shape == shape:
            pass
        elif not self.shapes:
            self.shapes = [self.shape]
        else:
            self.shapes.append(self.shape)
        self.shape = shape


make_datadef = DataDef
