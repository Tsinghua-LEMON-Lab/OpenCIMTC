from unittest import TestCase, main
from . import flatten_layers    # noqa
from ..core import make_layer
from ..core.jsonable import to_json_obj


class TestFlattenLayers(TestCase):

    def test_flatten_tree_1(self):
        b = make_layer(type='block', inputs=['inp:0', 'inp:1'])
        b.add_layer('in', type='input', inputs=[{}, {}])
        b.add_layer('x', op='add', inputs=['in:0', 'in:1'])
        b.add_layer('out', type='output', inputs=['x', 'in:1'])

        oups, layers = b.flatten_tree('b')
        self.assertEqual(to_json_obj(oups), ['b-x', 'inp:1'])
        self.assertEqual(to_json_obj(layers), {
            'b-x': {'op': {'op_id': 'add'}, 'inputs': ['inp:0', 'inp:1']}
        })

        b.number = 2
        oups, layers = b.flatten_tree('b')
        self.assertEqual(to_json_obj(oups), ['b-1-x', 'inp:1'])
        self.assertEqual(to_json_obj(layers), {
            'b-0-x': {'op': {'op_id': 'add'}, 'inputs': ['inp:0', 'inp:1']},
            'b-1-x': {'op': {'op_id': 'add'}, 'inputs': ['b-0-x', 'inp:1']}
        })

    def test_flatten_tree_2(self):
        b1 = make_layer(type='block')
        b1.add_layer('in', type='input', inputs=[{}, {}])
        b1.add_layer('x', op='add', inputs=['in:0', 'in:1'])
        b1.add_layer('out', type='output', inputs=['in:0', 'x'])

        b = make_layer(type='block', inputs=['inp2:0', 'inp2:1'])
        b.add_layer('in', type='input', inputs=[{}, {}])
        b.add_layer('b1', b1.clone(inputs=['in:1', 'in:0']))
        b.add_layer('out', type='output', inputs=['b1:1', 'in:0'])

        oups, layers = b.flatten_tree('b')
        self.assertEqual(to_json_obj(oups), ['b-b1-x', 'inp2:0'])
        self.assertEqual(to_json_obj(layers), {
            'b-b1-x': {'op': {'op_id': 'add'}, 'inputs': ['inp2:1', 'inp2:0']}
        })

        b.get_layer('b1').number = 2
        b.number = 2
        oups, layers = b.flatten_tree('b')
        self.assertEqual(to_json_obj(oups), ['b-1-b1-1-x', 'b-0-b1-1-x'])
        self.assertEqual(to_json_obj(layers), {
            'b-0-b1-0-x': {'op': {'op_id': 'add'},
                           'inputs': ['inp2:1', 'inp2:0']},
            'b-0-b1-1-x': {'op': {'op_id': 'add'},
                           'inputs': ['inp2:1', 'b-0-b1-0-x']},
            'b-1-b1-0-x': {'op': {'op_id': 'add'},
                           'inputs': ['inp2:0', 'b-0-b1-1-x']},
            'b-1-b1-1-x': {'op': {'op_id': 'add'},
                           'inputs': ['inp2:0', 'b-1-b1-0-x']},
        })

    def test_flatten_tree_3(self):
        b1 = make_layer(type='block')
        b1.add_layer('in', type='input', inputs=[{}, {}])
        b1.add_layer('out', type='output', inputs=['in:1', 'in:0'])

        b = make_layer(type='block', inputs=['inp:0', 'inp:1'])
        b.add_layer('in', type='input', inputs=[{}, {}])
        b.add_layer('b1', b1.clone(inputs=['in:1', 'in:0']))
        b.add_layer('out', type='output', inputs=['b1:1', 'in:0'])

        oups, layers = b.flatten_tree('b')
        self.assertEqual(to_json_obj(oups), ['inp:1', 'inp:0'])
        self.assertEqual(to_json_obj(layers), {})

        b.get_layer('b1').number = 2
        b.number = 2
        oups, layers = b.flatten_tree('b')
        self.assertEqual(to_json_obj(oups), ['inp:0', 'inp:0'])
        self.assertEqual(to_json_obj(layers), {})


if __name__ == '__main__':
    main()
