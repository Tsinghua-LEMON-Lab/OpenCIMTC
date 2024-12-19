
from ..core import LayerTree, mixin
from ..core.ref import parse_ref, make_ref
from .graph import DagGraph


@mixin(LayerTree)
class LayerGraph:

    def build_flat_graph(self):
        grf = DagGraph()
        layers = dict(self.iter_layers())
        ins, outs = {}, {}

        for name, layer in layers.items():
            parts = parse_ref(name)
            if layer.is_layer_tree():
                grf.add_group(name, group=make_ref(*parts[:-1]),
                              tag=layer.type, number=layer.number)
                continue
            if layer.type == 'input':
                ins[make_ref(*parts[:-1])] = name
            elif layer.type == 'output':
                outs[make_ref(*parts[:-1])] = name
            else:
                pass
            grf.add_node(name, group=make_ref(*parts[:-1]),
                         tag=layer.type, number=None)

        for name, layer in layers.items():
            parts = parse_ref(name)
            if layer.is_layer_tree():
                name = ins[name]
            for dd in layer.inputs:
                if dd.ref is None:
                    continue
                nm, _ = dd.parse_ref()
                nm = make_ref(*parts[:-1], nm)
                if layers[nm].is_layer_tree():
                    grf.add_edge(outs[nm], name, tag='block-io')
                elif layer.is_layer_tree():
                    grf.add_edge(nm, name, tag='block-io')
                else:
                    grf.add_edge(nm, name, tag=None)

        return grf

    def build_tree_graph(self, *, ns=()):
        assert self.layers
        grf = DagGraph()
        for name, layer in self.layers.items():
            grf2 = None
            if layer.is_layer_tree():
                grf2 = layer.build_tree_graph(ns=(*ns, name))
            grf.add_node(make_ref(*ns, name), graph=grf2)
        for name, layer in self.layers.items():
            for dd in layer.inputs:
                if dd.ref is None:
                    assert layer.type == 'input'
                    continue
                nm, idx = dd.parse_ref()
                assert nm in self.layers
                grf.add_edge(make_ref(*ns, nm), make_ref(*ns, name))
        return grf
