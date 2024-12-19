from unittest import TestCase, main
from .matmul import MatMulOp
from ..core import make_op


class TestMatMul(TestCase):

    def test_matmul(self):
        op = MatMulOp(in_channel=2, out_channel=3, bias=False)
        self.assertEqual(op.op_id, 'matmul')
        self.assertEqual(op.in_channel, 2)
        self.assertEqual(op.out_channel, 3)
        self.assertIs(op.bias, False)

        op = make_op('matmul', in_channel=4, out_channel=6, bias=True)
        self.assertTrue(isinstance(op, MatMulOp))
        self.assertEqual(op.op_id, 'matmul')
        self.assertEqual(op.in_channel, 4)
        self.assertEqual(op.out_channel, 6)
        self.assertIs(op.bias, True)


if __name__ == '__main__':
    main()
