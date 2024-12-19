from .reg import AbsReg, RegBase
from .jsonable import Jsonable
from .type_util import to_obj_dict, is_integer
from .op import make_op
from .datadef import DataDef
from .ref import is_valid_name, query_tree_ref, make_ref
from .ns import ns_push
from contextlib import contextmanager

class LayerReg(AbsReg, key='type', dft_key='op'):
    pass


class BaseLayer(Jsonable, RegBase, metaclass=LayerReg, ):

    inputs = None
    outputs = None
    weights = None

    def __init__(self, *, inputs=None, outputs=None,
                 weights=None, datadef=DataDef, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('inputs', to_obj_dict(inputs, datadef))
        self.set_attr('outputs', to_obj_dict(outputs, datadef))
        self.set_attr('weights', to_obj_dict(weights, datadef))
        
    def validate(self):
        super().validate()
        assert self.inputs is None or isinstance(self.inputs, list)
        assert self.outputs is None or isinstance(self.outputs, list)
        assert self.weights is None or isinstance(self.weights, dict)

    def validate_graph(self):
        pass

    def is_layer_tree(self):
        return False

    def is_io_layer(self):
        return False

    def iter_inputs(self):
        if self.inputs:
            for i, dd in enumerate(self.inputs):
                with ns_push(f'inputs[{i}]'):
                    yield i, dd

    def iter_outputs(self):
        if self.outputs:
            for i, dd in enumerate(self.outputs):
                with ns_push(f'outputs[{i}]'):
                    yield i, dd

    def iter_weights(self):
        if self.weights:
            for name, dd in self.weights.items():
                with ns_push(f'weights[{name!r}]'):
                    yield name, dd


class OpLayer(BaseLayer):

    type = 'op'
    op = None

    def __init__(self, *, op, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('op', make_op(op), not_none=True)

    def validate(self):
        super().validate()
        assert not hasattr(self, 'layers')

    def validate_graph(self):
        n = len(self.inputs or ())
        if self.op.num_inputs is None:
            assert n > 1, f'invalid {n} inputs'
        else:
            assert n == self.op.num_inputs, \
                f'invalid {n} inputs, expects {self.op.num_inputs}'


class LayerTree(Jsonable):

    layers = None

    def __init__(self, *, layers=None, **kwargs):
        super().__init__(**kwargs)
        if layers is None:
            layers = self.layers
        self.set_attr('layers', to_obj_dict(layers, BaseLayer, make_layer))

    def is_layer_tree(self):
        return True

    def add_layer(self, name, layer=None, **kwargs):
        '''add a layer object'''
        if self.layers is None:
            self.layers = {}
        assert is_valid_name(name), f'invalid layer name={name!r}'
        assert name not in self.layers, f'layer name={name!r} exists'
        if layer is None:
            self.layers[name] = make_layer(**kwargs)
        elif isinstance(layer, BaseLayer):
            self.layers[name] = layer.clone(**kwargs)
        else:
            assert False, f'invalid layer={layer!r}'

    def get_layer(self, ref):
        '''Get the layer object by layer reference name'''
        return query_tree_ref(self, 'layers', ref)

    def iter_layers(self, *, deep=True, sorted=False, _ns=[]):
        if not self.layers:
            return
        keys = self.sorted_layer_names() if sorted else self.layers.keys()
        for name in keys:
            with self.with_layer(name) as layer:
                _ns.append(name)
                try:
                    yield make_ref(*_ns), layer
                    if deep and isinstance(layer, LayerTree):
                        yield from layer.iter_layers(deep=deep, sorted=sorted,
                                                     _ns=_ns)
                except StopIteration:
                    break
                finally:
                    _ns.pop()

    def is_flat_graph(self):
        if not self.layers:
            return True
        for name, layer in self.layers.items():
            if layer.is_layer_tree():
                return False
        return True

    def is_simple_graph(self):
        if not self.layers:
            return True
        return not self.layers or all(
            not layer.is_layer_tree() or layer.is_simple_graph()
            for layer in self.layers.values())

    def find_io_layers(self, layers=None):
        if layers is None:
            layers = self.layers
        inps, oups = [], []
        if layers is not None:
            for name, layer in layers.items():
                if layer.type == 'input':
                    inps.append(name)
                elif layer.type == 'output':
                    oups.append(name)
        return inps, oups

    def get_io_layers(self, layers=None):
        inps, oups = self.find_io_layers(layers)
        assert len(inps) == len(oups) == 1
        return inps[0], oups[0]

    @contextmanager
    def with_layer(self, name, layer=None):
        with ns_push(f'layers[{name!r}]'):
            yield layer if layer is not None else self.layers[name]

    def validate_graph(self):
        if not self.layers:
            return
        inps, oups = self.find_io_layers()
        assert len(inps) == 1, f'invalid {len(inps)} input layers'
        assert len(oups) == 1, f'invalid {len(oups)} output layers'
        for key, layer in self.iter_layers(deep=False):
            layer.validate_graph()
            for i, dd in layer.iter_inputs():
                if layer.type == 'input':
                    assert dd.ref is None, f'invalid ref {dd.ref!r}'
                else:
                    nm, idx = dd.parse_ref()
                    assert nm in self.layers, f'invalid ref {dd.ref!r}'
            if layer.type == 'reuse':
                assert layer.layer in self.layers, \
                       f'invalid reuse layer {layer.layer}'

    def sorted_layer_names(self):
        if not self.layers:
            return []
        names = list(self.layers)
        _cache = dict()

        def dist(name):
            if name is None:
                return 0
            d = _cache.get(name)
            if d is not None:
                return d
            layer = self.layers[name]
            if not layer.inputs:
                return 0
            d = max(dist(dd.parse_ref()[0]) + 1 for dd in layer.inputs)
            _cache[name] = d
            return d

        names.sort(key=dist)
        return names


class BlockLayer(LayerTree, BaseLayer):

    type = 'block'
    number = 1

    def __init__(self, *, number=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('number', number, is_integer, min_val=1)

    def validate(self):
        super().validate()
        assert self.number > 0

    def validate_graph(self):
        super().validate_graph()
        inp, oup = self.get_io_layers()
        with self.with_layer(inp) as layer:
            n = len(layer.inputs or ())
            assert n == len(self.inputs), f'invalid {n} inputs'
        if self.number > 1:
            with self.with_layer(oup) as layer:
                n = len(layer.inputs or ())
                assert n == len(self.inputs), f'invalid {n} outputs'

    def is_simple_graph(self):
        return self.number == 1 and super().is_simple_graph()


class IOLayer(BaseLayer):

    def validate(self):
        super().validate()
        assert self.weights is None
        assert not hasattr(self, 'layers')
        assert not hasattr(self, 'op')

    def is_io_layer(self):
        return True

    def validate_graph(self):
        assert len(self.inputs or ()), 'empty inputs is invalid'


class InputLayer(IOLayer):

    type = 'input'


class OutputLayer(IOLayer):

    type = 'output'


class ReuseLayer(BaseLayer):

    type = 'reuse'
    layer = None

    def __init__(self, *, layer=None, **kwargs):
        super().__init__(**kwargs)
        self.set_attr('layer', layer, is_valid_name, not_none=True)

    def validate(self):
        super().validate()
        assert self.weights is None


make_layer = BaseLayer.make_obj
