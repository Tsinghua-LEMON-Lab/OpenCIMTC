from unittest import TestCase, main
from .jsonable import to_json_obj, Jsonable, has_yaml, load_json, dump_json
from io import StringIO
from tempfile import TemporaryDirectory
from pathlib import Path


class TestToJsonObj(TestCase):

    def test_scala(self):
        self.assertIs(to_json_obj(None), None)
        self.assertEqual(to_json_obj(1), 1)
        self.assertEqual(to_json_obj(1.5), 1.5)
        self.assertEqual(to_json_obj('foo'), 'foo')
        self.assertIs(to_json_obj(True), True)
        self.assertIs(to_json_obj(False), False)
        self.assertEqual(to_json_obj(b'\0\1\2'), b'\0\1\2')

    def test_array(self):
        d1 = [None, 1, 1.5, True, False, 'foo', b'\0\1\2']
        self.assertEqual(to_json_obj(d1), d1)
        self.assertEqual(to_json_obj(tuple(d1)), d1)
        d2 = [d1, *d1, (d1)]
        self.assertEqual(to_json_obj(d2), d2)

    def test_object(self):
        d1 = dict(a=[1, '2', False], b=1, c=1.5, d=True, e='foo', f=b'\0\1\2')
        self.assertEqual(to_json_obj(d1), d1)
        d2 = dict(d1, _g='gg', h=d1, i=None, j=[], k={})
        self.assertEqual(to_json_obj(d2), dict(d1, h=d1, j=[], k={}))
        self.assertEqual(to_json_obj(d2, filter=False), d2)

    def test_jsonable_obj(self):
        class Foo(Jsonable):
            pass
        d1 = dict(a=1, b='b')
        d2 = dict(_c=[1.5, False, b'\xcc'], d=None)
        d3 = {**d1, **d2}
        f1 = Foo(**d3)
        self.assertEqual(to_json_obj(f1), d1)
        self.assertEqual(to_json_obj(f1, filter=False), d3)
        self.assertEqual(to_json_obj([f1]), [d1])
        self.assertEqual(to_json_obj(dict(f=f1)), dict(f=d1))
        f2 = Foo(f=f1, x=dict(y=f1), z=None, _q=1)
        d4 = dict(f=d1, x=dict(y=d1))
        d5 = dict(f=d3, x=dict(y=d3), z=None, _q=1)
        self.assertEqual(to_json_obj(f2), d4)
        self.assertEqual(f2.to_json_obj(), d4)
        self.assertEqual(f2.to_json_obj(filter=False), d5)

    def test_unjsonable_obj(self):
        class X:
            pass
        x = X()
        x.a = 1
        self.assertRaises(TypeError, to_json_obj, x)
        self.assertRaises(TypeError, to_json_obj, [x])
        self.assertRaises(TypeError, to_json_obj, {'x': x})
        self.assertEqual(to_json_obj({'_x': x}), {})

    def test_from_json_obj(self):
        class Foo(Jsonable):
            def __init__(self, *args, **kwargs):
                self.a = list(args)
                self.k = kwargs
        self.assertIs(Foo.from_json_obj(None), None)
        self.assertIs(type(Foo.from_json_obj(1)), Foo)
        self.assertEqual(Foo.from_json_obj(1).__dict__, {'a': [1], 'k': {}})
        self.assertEqual(Foo.from_json_obj([.5, None]).__dict__,
                         {'a': [.5, None], 'k': {}})
        self.assertEqual(Foo.from_json_obj({'x': [True, {'y': []}]}).__dict__,
                         {'a': [], 'k': {'x': [True, {'y': []}]}})

    def test_circular(self):
        class Foo(Jsonable):
            pass
        d1 = {}
        d1['a'] = d1
        self.assertRaises(AssertionError, to_json_obj, d1)
        d2 = []
        d2.append(d2)
        self.assertRaises(AssertionError, to_json_obj, d2)
        f1 = Foo()
        f1.f = f1
        self.assertRaises(AssertionError, f1.to_json_obj)

    def test_clone(self):
        class Foo(Jsonable):
            pass
        d = {'a': 1, 'b': False, 'c': [.5, None, {'x': ['h', 'y']}]}
        f = Foo.from_json_obj(d)
        self.assertEqual(f.to_json_obj(), d)
        f2 = f.clone()
        self.assertEqual(f2.to_json_obj(), d)


class TestLoadDumpJson(TestCase):

    def test_load_yaml(self):
        if not has_yaml():
            return
        s = '''nm: foo
vs:
  - n: b1
    v: true
  - n: b2
    v: null
  - n: 3
    v: [1, 2, {a: 1, b: false, c: cc}]
'''
        o = load_json(s)
        self.assertEqual(o, {'nm': 'foo', 'vs': [
            {'n': 'b1', 'v': True}, {'n': 'b2', 'v': None},
            {'n': 3, 'v': [1, 2, {'a': 1, 'b': False, 'c': 'cc'}]}]})

        class F(Jsonable):
            pass

        f = load_json(s, cls=F)
        self.assertIs(type(f), F)
        self.assertEqual(f.nm, 'foo')
        self.assertEqual(f.vs[0], {'n': 'b1', 'v': True})
        self.assertEqual(f.vs[2]['v'][2]['c'], 'cc')
        self.assertEqual(to_json_obj(f, filter=False), o)

    def test_load_json(self):
        s = '''[1, null, false, {"0": 0.5, "b": {"x": 1, "y": [2, true]}}]'''
        o = load_json(s, auto_yaml=False)
        self.assertEqual(o, [1, None, False,
                         {'0': .5, 'b': {'x': 1, 'y': [2, True]}}])

        class F(Jsonable, list):
            def __init__(self, *args):
                list.__init__(self, args)

        f = load_json(s, auto_yaml=False, cls=F)
        self.assertEqual(dump_json(f, auto_yaml=False), s)
        f2 = StringIO(s)
        o2 = load_json(file=f2)
        self.assertEqual(o, o2)
        f3 = StringIO()
        dump_json(f, file=f3, auto_yaml=False)
        self.assertEqual(f3.getvalue(), s)

        with TemporaryDirectory() as td:
            f4 = Path(td) / 'f4.yaml'
            dump_json(f, file=f4)
            o4 = load_json(file=f4)
            self.assertEqual(o4, o)
            f5 = Path(td) / 'f5.json'
            dump_json(f, file=f5, auto_yaml=False)
            o5 = load_json(file=f5, auto_yaml=False)
            self.assertEqual(o5, o)


if __name__ == '__main__':
    main()
