from unittest import TestCase, main
from .conv import Conv3dOp, Conv2dOp
from ..core import make_op


class TestConvOp(TestCase):

    def test_conv1d(self):
        op = make_op('conv1d', in_channel=3, out_channel=4, kernel=3)
        self.assertEqual(op.op_id, 'conv1d')
        self.assertEqual(op.in_channel, 3)
        self.assertEqual(op.out_channel, 4)
        self.assertEqual(op.kernel, 3)
        self.assertEqual(op.stride, 1)
        self.assertEqual(op.padding, 0)
        self.assertEqual(op.dilation, 1)
        self.assertEqual(op.group, 1)
        self.assertIs(op.bias, True)

    def test_conv2d(self):
        op = Conv2dOp(in_channel=4, out_channel=2, kernel=3, stride=1,
                      padding=1, dilation=2)
        self.assertEqual(op.op_id, 'conv2d')
        self.assertEqual(op.dilation, 2)
        self.assertEqual(op.kernel, 3)
        self.assertEqual(op.stride, 1)
        self.assertEqual(op.padding, 1)

    def test_conv3d(self):
        op = Conv3dOp(in_channel=3, out_channel=4, kernel=3)
        self.assertEqual(op.op_id, 'conv3d')
        self.assertEqual(op.in_channel, 3)
        self.assertEqual(op.out_channel, 4)
        self.assertEqual(op.kernel, 3)
        self.assertEqual(op.stride, 1)
        self.assertEqual(op.padding, 0)
        self.assertEqual(op.dilation, 1)
        self.assertEqual(op.group, 1)
        self.assertIs(op.bias, True)

        self.assertRaises(ValueError, Conv3dOp, in_channel=1, out_channel=2,
                          kernel=3, padding=[1, 2])

    def test_convtranspose2d(self):
        op = make_op('conv_transpose2d', in_channel=4, out_channel=3, kernel=3,
                     stride=(1, 2), padding=(2, 1), dilation=(1, 2),
                     output_padding=(1, 0))
        self.assertEqual(op.op_id, 'conv_transpose2d')
        self.assertEqual(op.formalized_attrs(), dict(kernel=(3, 3),
                         stride=(1, 2), padding=(2, 1, 2, 1),
                         dilation=(1, 2), output_padding=(1, 0),
                         auto_pad=None))


if __name__ == '__main__':
    main()
