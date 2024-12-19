# flake8: noqa: E501
from unittest import TestCase, main
from . import *


class TestIR(TestCase):

    def test_make_ir(self):
        ir = make_ir()

        image = make_datadef(channel=3, channel_last=True, dtype='uint8')

        conv3x3 = make_op('conv2d', kernel=3, padding=1, bias=True, relu=True)
        conv1x1 = make_op('conv2d', kernel=1, bias=True, relu=True)

        block = make_layer(type='block')
        block.add_layer('out', type='output', inputs=['add'])
        block.add_layer('conv-1', op=conv3x3.clone(in_channel=64, out_channel=64), inputs=['in'])
        block.add_layer('conv-3', op=conv1x1.clone(in_channel=64, out_channel=64), inputs=['in'])
        block.add_layer('conv-2', op=conv1x1.clone(in_channel=64, out_channel=64), inputs=['conv-1'])
        block.add_layer('add', op='add', inputs=['conv-2', 'conv-3'])
        block.add_layer('in', type='input', inputs=[dict(channel=64)])

        ir.add_layer('image', type='input', inputs=[image.clone()])
        ir.add_layer('conv-1', op=conv3x3.clone(in_channel=3, out_channel=64), inputs=['image'])
        ir.add_layer('res-blk', block.clone(inputs=['conv-1'], number=8))

        ir.add_layer('out', type='output', inputs=['res-blk'])

        ir.add_device('xb', 'rram-144k', number=4)

        self.assertEqual(ir.layers['image'].inputs[0].channel, 3)
        self.assertEqual(block.number, 1)
        self.assertEqual(block.layers['add'].op.op_id, 'add')
        self.assertEqual(ir.get_layer('res-blk.conv-3').inputs[0].ref, 'in')

        ref = 'res-blk.conv-2'
        layer = ir.get_layer(ref)
        self.assertEqual(layer.inputs[0].ref, 'conv-1')

        dev = ir.get_device('xb')
        self.assertEqual(dev.profile.weight_bits, 4)

        ir.validate_graph()

        self.assertEqual(block.sorted_layer_names(), \
            ['in', 'conv-1', 'conv-3', 'conv-2', 'add', 'out'])

        from ..tools import flatten_layers
        self.assertFalse(ir.is_flat_graph())


if __name__ == '__main__':
    main()
