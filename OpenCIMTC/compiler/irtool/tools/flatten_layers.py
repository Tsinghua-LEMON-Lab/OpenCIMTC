from ..core.layer import LayerTree, BlockLayer
from ..core import mixin


@mixin(LayerTree)
class FlattenLayerTree:

    def flatten_layers(self):
        if not self.layers:
            return self.layers
        layers = {}
        trees = {}
        for name, layer in self.layers.items():
            if layer.is_layer_tree():
                oups, olayers = layer.flatten_tree(name)
                layers.update(olayers)
                trees[name] = oups
            else:
                layers[name] = layer.clone()
        for layer in layers.values():
            if layer.type == 'op' and layer.op.op_id in ['constant']:
                continue
            for dd in layer.inputs:
                nm, idx = dd.parse_ref()
                i = 0 if idx is None else idx
                ous = trees.get(nm)
                if ous:
                    dd.set_ref(*ous[i].parse_ref())
        return layers

    def flatten_tree(self, ns_name):
        raise NotImplementedError


@mixin(BlockLayer)
class FlattenBlockLayer(FlattenLayerTree):

    def flatten_tree(self, ns_name):
        if not self.layers:
            return self.inputs, {}

        from_layers = self.flatten_layers()

        layers = {}
        inp, oup = self.get_io_layers(from_layers)
        oul = from_layers[oup]

        def new_ref(dd, ith):
            name, idx = dd.parse_ref()
            i = 0 if idx is None else idx
            if name == inp:
                if ith == 0:
                    return self.inputs[i].parse_ref()
                else:
                    return new_ref(oul.inputs[i], ith - 1)
            elif self.number > 1:
                return (f'{ns_name}-{ith}-{name}', idx)
            else:
                return (f'{ns_name}-{name}', idx)

        for i in range(self.number):
            for name, layer in from_layers.items():
                if layer.is_io_layer():
                    continue
                layer = layer.clone()
                for d in layer.inputs:
                    d.set_ref(*new_ref(d, i))
                if self.number > 1:
                    layers[f'{ns_name}-{i}-{name}'] = layer
                else:
                    layers[f'{ns_name}-{name}'] = layer

        outputs = [d.clone() for d in oul.inputs]
        for d in outputs:
            d.set_ref(*new_ref(d, self.number - 1))

        return outputs, layers
