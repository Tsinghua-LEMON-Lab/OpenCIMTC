from ..core import BaseDevice, Jsonable
from ..core.type_util import to_cls_obj, is_integer, is_boolean


class RramProfile(Jsonable):

    in_channel = None
    out_channel = None
    in_bits = None
    out_bits = None
    weight_bits = None
    signed = None

    def __init__(self, *, in_channel=None, out_channel=None, in_bits=None,
                 out_bits=None, weight_bits=None, signed=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('in_channel', in_channel, is_integer, min_val=1,
                      not_none=True)
        self.set_attr('out_channel', out_channel, is_integer, min_val=1,
                      not_none=True)
        self.set_attr('in_bits', in_bits, is_integer, min_val=1,
                      not_none=True)
        self.set_attr('out_bits', out_bits, is_integer, min_val=1,
                      not_none=True)
        self.set_attr('weight_bits', weight_bits, is_integer,
                      min_val=1, not_none=True)
        self.set_attr('signed', signed, is_boolean, not_none=True)


class RramDevice(BaseDevice):

    kind = 'rram'
    profile = None

    def __init__(self, *, profile=None, **kwargs):
        super().__init__(**kwargs)
        p = dict(self.profile)
        if profile is not None:
            p.update(profile)
        self.profile = RramProfile(**p)


class Rram144kDevice(RramDevice):

    kind = 'rram-144k'

    profile = {
        'in_channel': 576,   # 1152/2
        'out_channel': 128,  # 128
        'in_bits': 2,        # [-1, 1] sint2
        'out_bits': 4,       # [-7, 7] sint4
        'weight_bits': 4,    # [-7, 7] sint4
        'signed': True,
    }
