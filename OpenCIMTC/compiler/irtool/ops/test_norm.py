from unittest import TestCase, main
from .norm import BatchNormOp
from ..core import make_op


class TestNormOps(TestCase):

    def test_batch_norms(self):
        for ndim in (1, 2, 3):
            op = make_op(f'batch_norm{ndim}d', channel=4)
            self.assertTrue(isinstance(op, BatchNormOp))
            self.assertEqual(op.ndim, ndim)
            self.assertEqual(op.channel, 4)
            self.assertEqual(op.epsilon, 1e-5)


if __name__ == '__main__':
    main()
