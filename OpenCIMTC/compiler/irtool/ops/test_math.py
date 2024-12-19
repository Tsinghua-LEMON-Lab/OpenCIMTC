from unittest import TestCase, main
from .math import AddOp, SubOp, MulOp, DivOp
from ..core import make_op


class TestMathOps(TestCase):

    def test_add(self):
        op = make_op('add')
        self.assertTrue(isinstance(op, AddOp))

    def test_sub(self):
        op = make_op('sub')
        self.assertTrue(isinstance(op, SubOp))

    def test_mul(self):
        op = make_op('mul')
        self.assertTrue(isinstance(op, MulOp))

    def test_div(self):
        op = make_op('div')
        self.assertTrue(isinstance(op, DivOp))


if __name__ == '__main__':
    main()
