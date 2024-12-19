from unittest import TestCase, main
from .rram import RramDevice
from ..core import make_device


class TestReRamDevice(TestCase):

    def test_rram(self):
        self.assertRaises(ValueError, RramDevice)
        self.assertRaises(ValueError, make_device, 'rram')
        dev = make_device('rram', profile=dict(in_channel=4, out_channel=3,
                          in_bits=2, out_bits=3, weight_bits=2, signed=True),
                          number=4)
        self.assertEqual(dev.profile.out_channel, 3)
        self.assertEqual(dev.number, 4)

    def test_rram144k(self):
        dev = make_device('rram-144k')
        self.assertEqual(dev.profile.in_channel, 576)
        self.assertEqual(dev.profile.weight_bits, 4)
        self.assertIs(dev.profile.signed, True)


if __name__ == '__main__':
    main()
