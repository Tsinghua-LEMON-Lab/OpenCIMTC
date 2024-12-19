from unittest import TestCase, main
from .ref import is_valid_name, parse_name, is_valid_ref, parse_ref, make_ref


class TestRef(TestCase):

    def test_is_valid_name(self):
        self.assertFalse(is_valid_name(None))
        self.assertFalse(is_valid_name(''))
        self.assertFalse(is_valid_name(1))
        self.assertFalse(is_valid_name('1'))
        self.assertFalse(is_valid_name('_'))
        self.assertFalse(is_valid_name('-'))
        self.assertTrue(is_valid_name('a'))
        self.assertTrue(is_valid_name('a1'))
        self.assertFalse(is_valid_name('a-'))
        self.assertFalse(is_valid_name('a_'))
        self.assertFalse(is_valid_name('_a'))
        self.assertFalse(is_valid_name('_1'))
        self.assertTrue(is_valid_name('a:1'))
        self.assertFalse(is_valid_name(':1'))
        self.assertFalse(is_valid_name(':a'))
        self.assertFalse(is_valid_name('a:a'))
        self.assertFalse(is_valid_name('_:1'))
        self.assertFalse(is_valid_name('-:1'))
        self.assertFalse(is_valid_name('a:-1'))
        self.assertFalse(is_valid_name('a:_1'))
        self.assertTrue(is_valid_name('a1-b2_c3-d4'))
        self.assertTrue(is_valid_name('a1-b2_c3-d4:999'))

    def test_parse_name(self):
        self.assertEqual(parse_name('a1'), ('a1', None))
        self.assertEqual(parse_name('a:1'), ('a', 1))
        self.assertEqual(parse_name('a1-b2_c3-d4:5'), ('a1-b2_c3-d4', 5))

    def test_is_valid_ref(self):
        self.assertTrue(is_valid_ref('a.b1:2.c-2.d_3-4:5'))

    def test_parse_ref(self):
        self.assertEqual(parse_ref('a.b1:2.c-2.d_3-4:5'),
                         (('a', None), ('b1', 2), ('c-2', None), ('d_3-4', 5)))

    def test_make_ref(self):
        self.assertIsNone(make_ref())
        self.assertIsNone(make_ref(*[]))
        self.assertEqual(make_ref('a', ('b', 3), 'c', ('d', None)),
                         'a.b:3.c.d')


if __name__ == '__main__':
    main()
