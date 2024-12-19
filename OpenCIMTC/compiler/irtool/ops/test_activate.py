from unittest import TestCase, main
from .activate import ReluOp, SoftmaxOp
from ..core import make_op


class TestActivateOps(TestCase):

    def test_relu(self):
        op = make_op('relu')
        self.assertTrue(isinstance(op, ReluOp))

    def test_softmax(self):
        op = make_op('softmax')
        self.assertTrue(isinstance(op, SoftmaxOp))


if __name__ == '__main__':
    main()
