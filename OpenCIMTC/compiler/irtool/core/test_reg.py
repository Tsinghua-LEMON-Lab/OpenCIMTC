from unittest import TestCase, main
from .reg import AbsReg, RegBase


class TestReg(TestCase):

    def test_reg(self):
        class AReg(AbsReg, key='k'):
            pass

        class A(RegBase, metaclass=AReg):
            def __init__(self, k):
                self.k = k

        class A1(A):
            k = 'a1'

        class BReg(AbsReg, key='k'):
            pass

        class B(RegBase, metaclass=BReg):
            pass

        class B1(B):
            k = 'a1'

        self.assertIsNone(AReg.lookup('x'))
        self.assertIsNone(AReg.lookup(None))
        self.assertIs(AReg.lookup('a1'), A1)
        self.assertIs(BReg.lookup('a1'), B1)

        a = A.make_obj('a1')
        self.assertIs(type(a), A1)
        self.assertEqual(a.k, 'a1')
        a = A.make_obj({'k': 'a1'})
        self.assertIs(type(a), A1)
        self.assertEqual(a.k, 'a1')


if __name__ == '__main__':
    main()
