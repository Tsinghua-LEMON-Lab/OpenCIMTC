from unittest import TestCase, main
from .layer import make_layer, OpLayer, BlockLayer, InputLayer
from .op import make_op
from .. import ops


class TestLayer(TestCase):

    def test_make_layer(self):
        layer = make_layer(op=dict(op_id='add'))
        self.assertTrue(isinstance(layer, OpLayer))
        self.assertTrue(isinstance(layer.op, ops.AddOp))
        self.assertEqual(layer.op.op_id, 'add')

        layer = make_layer('block', layers=dict(a=make_layer(op='add')),
                           number=4)
        self.assertTrue(isinstance(layer, BlockLayer))
        self.assertEqual(layer.number, 4)
        self.assertEqual(layer.layers['a'].op.op_id, 'add')

        layer = make_layer('input', inputs=[{'dims': [2, 3, 4]}])
        self.assertTrue(isinstance(layer, InputLayer))
        self.assertEqual(layer.inputs[0].dims, (2, 3, 4))

    def test_layer_tree(self):
        b = make_layer(type='block', inputs=['i1', 'i2'])
        b.add_layer('in', type='input', inputs=[{}, {}])
        b.add_layer('cv3', op=make_op('conv2d', kernel=1), inputs=['cv1'])
        b.add_layer('cv5', op=make_op('conv2d', kernel=1), inputs=['cv4'])
        b.add_layer('out', type='output', inputs=['ad6', 'in:2'])
        b.add_layer('cv4', op=make_op('conv2d', kernel=1), inputs=['cv2'])
        b.add_layer('ad6', op='add', inputs=['cv3', 'cv5'])
        b.add_layer('cv1', op=make_op('conv2d', kernel=1), inputs=['in:1'])
        b.add_layer('cv2', op=make_op('conv2d', kernel=1), inputs=['in:2'])
        b.add_layer('np7', op=make_op('conv2d', kernel=1), inputs=['in:2'])

        b.validate_graph()
        self.assertEqual(b.find_io_layers(), (['in'], ['out']))
        self.assertEqual(b.sorted_layer_names(),
                         ['in', 'cv1', 'cv2', 'np7', 'cv3', 'cv4', 'cv5',
                          'ad6', 'out'])


if __name__ == '__main__':
    main()
