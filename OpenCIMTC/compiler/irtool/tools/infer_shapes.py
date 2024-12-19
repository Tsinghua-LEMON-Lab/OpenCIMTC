from ..core import mixin, DataDef
from ..core.layer import LayerTree
from .shape_inferer import ShapeInferer


@mixin(LayerTree)
class InferShapes:

    def infer_shapes(self, *shapes, dims_only=False, channel_last=None,
                     inferer=None):
        if inferer is None:
            inferer = ShapeInferer()

        inp, oup = self.get_io_layers()

        with self.with_layer(inp) as layer:
            assert len(shapes) in (0, len(layer.inputs or ())), \
                f'need {len(layer.inputs)} input shapes'
            if channel_last is None:
                channel_last = self._infer_cl(layer)
                assert channel_last is not None, 'can\'t infer channel_last'
            cl = channel_last
            for i, dd in layer.iter_inputs():
                if not shapes:
                    shape = dd.make_shape(channel_last=cl)
                elif dims_only:
                    shape = dd.make_shape(dims=shapes[i], channel_last=cl)
                else:
                    shape = shapes[i]
                dd.set_shape(shape)
                dd.channel_last = cl

        for name, layer in self.iter_layers(deep=False, sorted=True):
            x = []
            for i, dd in layer.iter_inputs():
                nm, idx = dd.parse_ref()
                if idx is None:
                    idx = 0
                if nm is not None:
                    dd.set_shape(self.layers[nm].outputs[idx].shape)
                x.append(dd.shape)
                if dd.channel_last is not None:
                    dd.channel_last = cl
            if layer.is_io_layer():
                y = x
            elif layer.is_layer_tree():
                y = layer.infer_shapes(*x, inferer=inferer, channel_last=cl)
            elif layer.type == 'reuse':
                ll = self.layers[layer.layer]
                assert ll.type == 'op'
                y = inferer.infer_op(ll.op, *x, channel_last=cl)
            else:
                assert layer.type == 'op'
                y = inferer.infer_op(layer.op, *x, channel_last=cl)
                self._infer_wts(layer, *x, channel_last=cl)
            if layer.outputs is None:
                layer.outputs = [DataDef(shape=shape) for shape in y]
            else:
                assert len(layer.outputs) == len(y), \
                    f'outputs length {len(layer.outputs)} != {len(y)}'
                for i, dd in layer.iter_outputs():
                    dd.set_shape(y[i])
                    if dd.channel_last is not None:
                        dd.channel_last = cl

        with self.with_layer(oup) as layer:
            return [dd.shape for i, dd in layer.iter_outputs()]

    def _infer_cl(self, layer):
        channel_last = None
        for i, dd in layer.iter_inputs():
            if dd.channel_last is None:
                continue
            elif channel_last is None:
                channel_last = bool(dd.channel_last)
            else:
                assert dd.channel_last == channel_last, \
                    f'channel_last {dd.channel_last} != {channel_last}'
        return channel_last

    def _infer_wts(self, layer, *shapes, channel_last):
        if channel_last:
            channel = [s[-1] if s else None for s in shapes]
        else:
            channel = [s[0] if s else None for s in shapes]
        if not channel:
            channel = None
        elif all(c == channel[0] for c in channel[1:]):
            channel = channel[0]
        else:
            channel = None

        wts = layer.op.weight_shapes(channel_last=channel_last,
                                     channel=channel)
        wts = {k: v for k, v in wts.items() if v is not None}

        if not wts:
            assert not layer.weights, \
                f'weights {set(layer.weights)} are invalid'
        elif layer.weights is None:
            layer.weights = {k: DataDef(shape=v) for k, v in wts.items()}
        else:
            assert set(wts) == set(layer.weights), \
                f'weights {set(layer.weights)} != {set(wts)}'
            for k, dd in layer.iter_weights():
                if dd.shape is None:
                    dd.shape = wts[k]
                else:
                    assert dd.shape == wts[k], \
                        f'shape {dd.shape} != {wts[k]}'
                if dd.channel_last is not None:
                    dd.channel_last = channel_last
