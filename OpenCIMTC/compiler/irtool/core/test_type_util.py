from unittest import TestCase, main
from .type_util import (is_scalar, to_tokens, to_cls_obj, to_obj_dict,
                        to_int_tuple, to_obj_list, is_integers, is_integer,
                        is_boolean, to_boolean)


class TestTypeUtil(TestCase):

    def test_is_scalar(self):
        self.assertTrue(is_scalar(None))
        self.assertTrue(is_scalar(True))
        self.assertTrue(is_scalar(False))
        self.assertTrue(is_scalar(1))
        self.assertTrue(is_scalar(-.5))
        self.assertTrue(is_scalar('xx'))
        self.assertTrue(is_scalar(b'\x03'))

        self.assertFalse(is_scalar(object()))
        self.assertFalse(is_scalar([]))
        self.assertFalse(is_scalar(()))
        self.assertFalse(is_scalar({}))

    def test_is_integer(self):
        self.assertIs(is_integer(0), True)
        self.assertIs(is_integer(None), False)
        self.assertIs(is_integer(1.0), False)
        self.assertIs(is_integer('0'), False)
        self.assertIs(is_integer([]), False)
        self.assertIs(is_integer(0, min_val=0), True)
        self.assertIs(is_integer(0, min_val=1), False)
        self.assertIs(is_integer(0, min_val=-1), True)
        self.assertIs(is_integer(0, max_val=0), True)
        self.assertIs(is_integer(0, max_val=1), True)
        self.assertIs(is_integer(0, max_val=-1), False)
        self.assertIs(is_integer(0, min_val=0, max_val=0), True)
        self.assertIs(is_integer(0, min_val=-1, max_val=1), True)
        self.assertIs(is_integer(0, min_val=1, max_val=-1), False)

    def test_is_boolean(self):
        self.assertIs(is_boolean(None), False)
        self.assertIs(is_boolean(True), True)
        self.assertIs(is_boolean(False), True)
        self.assertIs(is_boolean(0), False)
        self.assertIs(is_boolean(1), False)
        self.assertIs(is_boolean('true'), False)
        self.assertIs(is_boolean('false'), False)

    def test_is_integers(self):
        self.assertFalse(is_integers(None))
        self.assertTrue(is_integers(None, none_ok=True))
        self.assertFalse(is_integers(False))
        self.assertFalse(is_integers(1.0))
        self.assertTrue(is_integers(()))
        # self.assertFalse(is_integers((), min_dim=1))
        self.assertTrue(is_integers(0))
        self.assertTrue(is_integers(0, min_val=0))
        self.assertTrue(is_integers(0, max_val=0))
        self.assertFalse(is_integers(0, min_val=1))
        self.assertFalse(is_integers(0, max_val=-1))
        self.assertTrue(is_integers([0]))
        self.assertTrue(is_integers([0], min_dim=1))
        self.assertTrue(is_integers([0], max_dim=1))
        self.assertFalse(is_integers([0], min_dim=2))
        self.assertFalse(is_integers([0], max_dim=0))

    def test_to_boolean(self):
        for x in (None, False, 0, 0.0, '0', '', 'false', 'False', 'FALSE',
                  'no', 'N', [], {}):
            self.assertIs(to_boolean(x), False)
        for x in (1, 1.0, '1', True, 'Y', 'yes', 'true', [0], {None: None}):
            self.assertIs(to_boolean(x), True)
        self.assertRaises(ValueError, to_boolean, 'ok')

    def test_to_tokens(self):
        self.assertEqual(to_tokens(None), ())
        self.assertEqual(to_tokens(None, list), [])
        self.assertEqual(to_tokens('foo bar'), ('foo bar',))
        self.assertEqual(to_tokens('foo bar'.split()), ('foo', 'bar'))
        self.assertEqual(to_tokens(iter([1, 2, 3])), ('1', '2', '3'))

        self.assertRaises(TypeError, to_tokens, {})
        self.assertRaises(TypeError, to_tokens, object())

    def test_to_int_tuple(self):
        self.assertEqual(to_int_tuple(None), ())
        self.assertRaises(TypeError, to_int_tuple, None, ndim=1)
        self.assertEqual(to_int_tuple(1), (1,))
        self.assertEqual(to_int_tuple(1, ndim=3), (1, 1, 1))
        self.assertEqual(to_int_tuple([1, 2, 3]), (1, 2, 3))
        self.assertEqual(to_int_tuple((1, 2, 3), ndim=3), (1, 2, 3))
        self.assertEqual(to_int_tuple([1, 2, 3], ndim=6), (1, 2, 3, 1, 2, 3))
        self.assertRaises(AssertionError, to_int_tuple, [1, 2], ndim=1)
        self.assertRaises(AssertionError, to_int_tuple, [1, 2], ndim=3)
        self.assertEqual(to_int_tuple(1, keep_scalar=True), 1)
        self.assertEqual(to_int_tuple([1, 2], keep_scalar=True), (1, 2))

    def test_to_cls_obj(self):
        class Foo:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
        f = Foo()
        self.assertIs(to_cls_obj(None, Foo), None)
        self.assertIs(to_cls_obj(f, Foo), f)
        self.assertEqual(to_cls_obj('1', float), 1.0)
        self.assertEqual(to_cls_obj('1 2 3'.split(), str,
                         func=lambda *x: ','.join(map(str, x))), '1,2,3')
        a = (False, 1, {'x': None})
        d = {'a': 1, 'b': [2, None, {'c': b'\x03'}]}
        self.assertEqual(to_cls_obj(a, Foo).args, a)
        self.assertEqual(to_cls_obj(d, Foo).kwargs, d)
        self.assertEqual(to_cls_obj((1, False), list, n_args='?'), [1, False])
        self.assertEqual(to_cls_obj(range(3), list, n_args='?'), [0, 1, 2])

    def test_to_obj_dict(self):
        class Foo:
            def __init__(self, a=None, b=None):
                self.a = a
                self.b = b

            def __eq__(self, o):
                return self.__dict__ == o.__dict__

        self.assertEqual(to_obj_dict(None, Foo), None)
        self.assertEqual(to_obj_dict({}, Foo), {})
        self.assertEqual(to_obj_dict([], Foo), [])
        self.assertRaises(TypeError, to_obj_dict, 'a', Foo)
        self.assertRaises(TypeError, to_obj_dict, True, Foo)
        self.assertEqual(to_obj_dict({'a': {'b': 'xx'}, 'x': None}, Foo),
                         {'a': Foo(b='xx'), 'x': None})
        self.assertEqual(to_obj_dict([{'b': 3.14}, {'a': [2.18]}], Foo),
                         [Foo(b=3.14), Foo(a=[2.18])])

    def test_obj_to_list(self):
        class Foo:
            def __init__(self, a=9, b=None):
                self.a = a
                self.b = b

            def __eq__(self, other):
                return self.__dict__ == other.__dict__

        self.assertIsNone(to_obj_list(None, str))
        self.assertEqual(to_obj_list(1, str), ['1'])
        self.assertEqual(to_obj_list(dict(a=1), Foo), [Foo(1)])
        self.assertEqual(to_obj_list(({'b': 3},
                         {'a': [False, .4], 'b': {}}), Foo),
                         [Foo(9, 3), Foo([False, .4], {})])


if __name__ == '__main__':
    main()
