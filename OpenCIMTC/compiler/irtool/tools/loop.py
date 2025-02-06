from ..core.layer import BaseLayer, LayerTree, make_layer
from ..core.type_util import mixin, is_one_of, to_cls_obj, is_integer
from ..core.datadef import DataDef
from ..core.jsonable import Jsonable
from ..core.op import make_op


class LoopProp(Jsonable):

    SOURCES = ('split', 'output')
    source = None   # one of SOURCES
    axis = 1        # axis to split, channel_pos='first', with_batch=True
    index = 0       # index of output

    def __init__(self, *, source=None, axis=None, index=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('source', source, is_one_of, values=self.SOURCES)
        self.set_attr('axis', axis, is_integer, min_val=0)
        self.set_attr('index', index, is_integer, min_val=0)


@mixin(DataDef)
class LoopDataDef(Jsonable):

    loop = None

    def __init__(self, *, loop=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('loop', to_cls_obj(loop, LoopProp), none_ok=True)


class LoopLayer(LayerTree, BaseLayer):

    type = 'loop'
    repeat = None

    def __init__(self, *, repeat=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('repeat', repeat, is_integer, min_val=2)

    def validate_graph(self):
        super().validate_graph()
        inp, oup = self.get_io_layers()
        noups = len(self.outputs or (None,))
        with self.with_layer(inp) as layer:
            n = len(layer.inputs or ())
            assert n == len(self.inputs), f'invalid {n} inputs'
            for i, dd in enumerate(layer.inputs):
                if dd.loop == 'output':
                    assert dd.index >= 0 and dd.index < noups, \
                           f'invalid input[{i}].index not in [0, {noups})'
        with self.with_layer(oup) as layer:
            n = len(layer.inputs or ())
            assert n == noups, f'invalid {n} outputs'

    def flatten_tree(self, ns_name):
        if not self.layers:
            return self.inputs, {}

        from_layers = self.flatten_layers()
        layers = {}
        inputs = []

        inps = [DataDef(dd.ref) for dd in self.inputs]
        inp, oup = self.get_io_layers()
        with self.with_layer(inp) as layer:
            assert len(self.inputs) == len(layer.inputs)
            for i, dd in enumerate(layer.inputs):
                di = inps[i]
                if dd.loop is None:
                    inputs.append(('I', di.ref))
                elif dd.loop.source == 'split':
                    name = f'{ns_name}_split_{inp}{i}'
                    op = make_op('split', axis=dd.loop.axis, split=self.repeat)
                    layers[name] = make_layer(op=op, inputs=[dict(ref=di.ref)])
                    inputs.append(('S', name))
                elif dd.loop.source == 'output':
                    ol = self.layers[oup]
                    oi = dd.loop.index
                    inputs.append(('O', di.ref,
                                   *ol.inputs[oi].parse_ref()))
                else:
                    assert False, f'invalid loop.source={dd.loop.source}'

        def new_ref(dd, ith):
            name, idx = dd.parse_ref()
            i = 0 if idx is None else idx
            if name == inp:
                t = inputs[i]
                if t[0] == 'I':
                    return (t[1],)
                elif t[0] == 'S':
                    return (t[1], ith)
                elif t[0] == 'O':
                    if ith == 0:
                        return (t[1],)
                    else:
                        name2, idx2 = t[2:]
                        return (f'{ns_name}_{name2}_{ith-1}', idx2)
                else:
                    assert False
            elif self.repeat > 1:
                return (f'{ns_name}_{name}_{ith}', idx)
            else:
                return (f'{ns_name}_{name}', idx)

        for i in range(self.repeat):
            for name, layer in from_layers.items():
                if layer.is_io_layer():
                    continue
                if i == 0:
                    layer2 = layer.clone()
                else:
                    layer2 = make_layer('reuse', layer=f'{ns_name}_{name}_{0}')
                    layer2.inputs = [dd.clone() for dd in layer.inputs]
                for d in layer2.inputs:
                    d.set_ref(*new_ref(d, i))
                if self.repeat > 1:
                    layers[f'{ns_name}_{name}_{i}'] = layer2
                else:
                    layers[f'{ns_name}_{name}'] = layer2

        outputs = [d.clone() for d in from_layers[oup].inputs]
        for d in outputs:
            d.set_ref(*new_ref(d, self.repeat - 1))

        return outputs, layers

    def infer_shapes(self, *shapes, **kwargs):
        ins = list(shapes)
        inp, _ = self.get_io_layers()
        layer = self.layers[inp]
        assert len(shapes) == len(layer.inputs)
        for i, dd in enumerate(layer.inputs):
            lo = dd.loop
            if lo.source == 'split':
                s = shapes[i]
                a = lo.axis - 1
                n = self.repeat
                sa = s[a] // n
                assert a >= 0 and a < len(s) and sa * n == s[a], \
                       f'invalid split {s}[{a}] to {n} secs'
                ins[i] = (*s[:a], sa, *s[a+1:])
        for r in range(self.repeat):
            outs = super().infer_shapes(*ins, **kwargs)
            if r + 1 == self.repeat:
                break
            for i, dd in enumerate(layer.inputs):
                lo = dd.loop
                if lo.source == 'output':
                    ins[i] = outs[lo.index]
        return outs
