from unittest import TestCase, main
from .trans import ConcatOp, ReshapeOp, FlattenOp, TransposeOp
from ..core import make_op


class TestTransOps(TestCase):

    def test_concat(self):
        op = make_op('concat', axis=-3)
        self.assertTrue(isinstance(op, ConcatOp))
        self.assertEqual(op.axis, -3)
        self.assertIs(op.channel_pos, "first")

    def test_reshape(self):
        op = make_op('reshape', shape=[3, 4])
        self.assertTrue(isinstance(op, ReshapeOp))
        self.assertEqual(op.shape, (3, 4))

    def test_flatten(self):
        op = make_op('flatten')
        self.assertTrue(isinstance(op, FlattenOp))

    def test_transpose(self):
        op = make_op('transpose', perm=[0, 2, 3, 1])
        self.assertTrue(isinstance(op, TransposeOp))
        self.assertEqual(op.perm, (0, 2, 3, 1))

    def test_pad(self):
        op = make_op('pad', pads=[0, 0])
        self.assertEqual(op.pads, (0, 0))
        self.assertEqual(op.value, 0)
        op = make_op('pad', pads=[1, 0, 0, 1], value=0.1)
        self.assertEqual(op.pads, (1, 0, 0, 1))
        self.assertEqual(op.value, 0.1)
        self.assertRaises(ValueError, make_op, 'pad', pads=0)
        self.assertRaises(ValueError, make_op, 'pad', pads=[0, 1, 2])


if __name__ == '__main__':
    main()
