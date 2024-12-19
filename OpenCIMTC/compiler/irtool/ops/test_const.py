from unittest import TestCase, main
from .const import ConstantOp, IdentityOp
from ..core import make_op


class TestConstOps(TestCase):

    def test_constant(self):
        op = make_op('constant', value=1.2)
        self.assertTrue(isinstance(op, ConstantOp))
        self.assertEqual(op.value, 1.2)
        self.assertEqual(op.attrs, ('value',))

    def test_identity(self):
        op = make_op('identity')
        self.assertTrue(isinstance(op, IdentityOp))


if __name__ == '__main__':
    main()
