from unittest import TestCase, main
from .pool import PoolOp, GlobalPoolOp
from ..core import make_op


class TestPoolOps(TestCase):

    def test_poolings(self):
        for method in ('avg', 'max'):
            for ndim in (1, 2, 3):
                op = make_op(f'{method}_pool{ndim}d', kernel=2)
                self.assertTrue(isinstance(op, PoolOp))
                self.assertEqual(op.ndim, ndim)
                self.assertEqual(op.kernel, 2)
                self.assertEqual(op.stride, 2)
                op = make_op(f'{method}_pool{ndim}d', kernel=2, stride=1)
                self.assertEqual(op.kernel, 2)
                self.assertEqual(op.stride, 1)

        for method in ('avg', 'max'):
            for ndim in (1, 2, 3):
                op = make_op(f'global_{method}_pool{ndim}d', kernel=2)
                self.assertTrue(isinstance(op, GlobalPoolOp))
                self.assertEqual(op.ndim, ndim)


if __name__ == '__main__':
    main()
