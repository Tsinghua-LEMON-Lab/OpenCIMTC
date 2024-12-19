import math
from ..core.ns import ns_push
from .utils import pool_shapes, conv_shapes, conv_t_shapes, \
                   resize_size_scale, concat_axis, split_to_secs
from typing import Callable
from ..core.type_util import to_var_token


class ShapeInferer:
    # no batch in inputs

    all_ops = {
        'const': {
            'constant',
        },
        'unary': {
            'identity', 'sign', 'abs', 'neg', 'ceil', 'floor',
            'exp', 'log', 'sqrt',
            'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
            'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
            'logical_not', 'bitwise_not',
            'relu', 'leaky_relu', 'prelu', 'selu', 'celu', 'elu', 'softmax',
            'log_softmax', 'sigmoid', 'hard_sigmoid', 'softplus', 'softsign',
            'silu'
        },
        'binary': {
            'add', 'sub', 'mul', 'div', 'mod', 'pow',
            'logical_and', 'logical_or', 'logical_xor',
            'bitwise_and', 'bitwise_or', 'bitwise_xor',
            'equal', 'less', 'less_or_equal', 'greater', 'greater_or_equal',
        },
        'trans': {
            'concat', 'reshape', 'flatten', 'transpose', 'pad', 'split'
        },
        'norm': {
            'batch_norm', 'batch_norm1d', 'batch_norm2d', 'batch_norm3d',
            'instance_norm1d', 'instance_norm2d', 'instance_norm3d'
        },
        'pool': {
            'avgpool1d', 'avgpool2d', 'avgpool3d',
            'avg_pool1d', 'avg_pool2d', 'avg_pool3d',
            'maxpool1d', 'maxpool2d', 'maxpool3d',
            'max_pool1d', 'max_pool2d', 'max_pool3d',
        },
        'global_pool': {
            'global_avg_pool1d', 'global_avg_pool2d', 'global_avg_pool3d',
            'global_max_pool1d', 'global_max_pool2d', 'global_max_pool3d'
        },
        'matmul': {
            'matmul', 'linear', 'fc'
        },
        'conv': {
            'conv1d', 'conv2d', 'conv3d'
        },
        'conv_transpose': {
            'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d'
        },
        'resize': {
            'resize',
        },
        'others': {
            'reducemean',
        },
    }

    def infer_op(self, op, *input_shapes, channel_last):
        with ns_push(f'op[{op.op_id!r}]'):
            y = self._infer(op, *input_shapes, channel_last=channel_last)
        if not isinstance(y, list):
            y = [y]
        return y

    def _infer(self, op, *x, channel_last):
        fn = getattr(op, 'infer_shape', None)
        if isinstance(fn, Callable):
            return fn(*x, channel_last=channel_last)
        fn = getattr(self, f'fn_{to_var_token(op.op_id)}', None)
        if isinstance(fn, Callable):
            return fn(op, *x, channel_last=channel_last)
        for k, v in self.all_ops.items():
            if op.op_id in v:
                fn = getattr(self, f'fn_{k}')
                return fn(op, *x, channel_last=channel_last)
        raise ValueError(f'can\'t find op {op.op_id!r} shape inferer')

    def fn_constant(self, op, **kwargs):
        import numpy
        return numpy.shape(op.value)

    def fn_unary(self, op, x, **kwargs):
        return x

    def fn_binary(self, op, x1, x2, **kwargs):
        if x1 == x2:
            return x1
        if len(x1) < len(x2):
            x1 = (1,) * (len(x2) - len(x1)) + x1
        elif len(x1) > len(x2):
            x2 = (1,) * (len(x1) - len(x2)) + x2
        else:
            pass
        y = []
        for a, b in zip(x1, x2):
            if a == b:
                y.append(a)
            elif a == 1:
                y.append(b)
            elif b == 1:
                y.append(a)
            else:
                assert False, f'invalid shape {op.op_id}({x1}, {x2})'
        return tuple(y)

    def fn_concat(self, op, *x, channel_last, **kwargs):
        x0 = x[0]
        rank = len(x0)
        assert all(len(v) == rank for v in x)
        axis = concat_axis(rank, channel_last, op.axis,
                           op.with_batch, op.channel_pos)
        for s in x[1:]:
            assert s[:axis] == x0[:axis] and s[axis+1:] == x0[axis+1:], \
                f'can\'t concat shape {x0} with {s}'
        return x0[:axis] + (sum(s[axis] for s in x),) + x0[axis+1:]

    def fn_reshape(self, op, x, **kwargs):
        shape = list(op.shape)
        n = shape.count(-1)
        assert n in (0, 1)
        if op.with_batch:
            assert shape[0] == -1, 'only axis 0 (batch) can be -1'
            shape = shape[1:]
        elif n:
            shape[shape.index(-1)] = math.prod(x) // -math.prod(shape)
        assert math.prod(x) == math.prod(shape), \
            f'can\'t reshape {x} to {shape}'
        return tuple(shape)

    def fn_flatten(self, op, x, **kwargs):
        start_dim = op.start_dim
        if op.with_batch:
            assert start_dim > 0
            start_dim -= 1
        assert -len(x) <= start_dim < len(x)
        return (*x[:start_dim], math.prod(x[start_dim:]))

    def fn_transpose(self, op, x, **kwargs):
        perm = tuple(op.perm)
        if op.with_batch:
            assert perm[0] == 0, \
                'axis 0 (batch) can\'t transpose to other axis'
            perm = tuple(p - 1 for p in perm[1:])
        assert tuple(sorted(perm)) == tuple(range(len(x)))
        return tuple(x[i] for i in perm)

    def fn_pad(self, op, x, *, channel_last, **kwargs):
        if channel_last:
            *xd, ci = x
        else:
            ci, *xd = x
        ndim = len(xd)
        p = op.pads
        assert len(p) == ndim * 2, \
            f'invalid pads {p} rank != {ndim}*2'
        os = tuple(xd[i] + p[i] + p[ndim + i] for i in range(ndim))
        return (*os, ci) if channel_last else (ci, *os)

    def fn_norm(self, op, x, **kwargs):
        return x

    def fn_pool(self, op, x, *, channel_last, **kwargs):
        ndim = op.ndim
        assert len(x) == ndim + 1, \
            f'invalid input rank {len(x)} != ndim({ndim}) + 1'
        if channel_last:
            *xd, ci = x
        else:
            ci, *xd = x
        os, _ = pool_shapes(xd, op.kernel, op.stride, op.padding, op.dilation,
                            op.auto_pad, op.ceil_mode)
        return (*os, ci) if channel_last else (ci, *os)

    def fn_global_pool(self, op, x, *, channel_last, **kwargs):
        ndim = op.ndim
        assert len(x) == ndim + 1, \
            f'invalid input rank {len(x)} != ndim({ndim}) + 1'
        c = x[-1] if channel_last else x[0]
        os = (1,) * ndim
        return (*os, c) if channel_last else (c, *os)

    def fn_matmul(self, op, x, **kwargs):
        ci = op.in_channel
        assert x == (ci,), f'invalid input shape {x} != {(ci,)}'
        co = op.out_channel
        return (co,)

    def fn_conv(self, op, x, *, channel_last, **kwargs):
        if channel_last:
            *xd, ci = x
        else:
            ci, *xd = x
        assert ci == op.in_channel, \
            f'invalid input {x} channel {ci} != {op.in_channel}'
        co = op.out_channel
        os, _ = conv_shapes(xd, op.kernel, op.stride, op.padding, op.dilation,
                            op.auto_pad)
        return (*os, co) if channel_last else (co, *os)

    def fn_conv_transpose(self, op, x, *, channel_last, **kwargs):
        if channel_last:
            *xd, ci = x
        else:
            ci, *xd = x
        assert ci == op.in_channel, \
            f'invalid input {x} channel {ci} != {op.in_channel}'
        co = op.out_channel
        os, _ = conv_t_shapes(xd, op.kernel, op.stride, op.padding,
                              op.dilation, op.output_padding, op.auto_pad)
        return (*os, co) if channel_last else (co, *os)

    def fn_resize(self, op, x, *, channel_last, **kwargs):
        if channel_last:
            *xd, ci = x
        else:
            ci, *xd = x
        yd, _ = resize_size_scale(xd, op.size, op.scale)
        return (*yd, ci) if channel_last else (ci, *yd)

    def fn_reducemean(self, op, x, *, channel_last, **kwargs):
        a = op.axes + (len(x) if op.axes < 0 else 0)
        assert a >= 0 and a < len(x), f'invalid input {x} axes={op.axes}'
        return (*x[:a], 1, x[a+1:]) if op.keepdims else (*x[:a], *x[a+1:])

    def fn_split(self, op, x, *, channel_last, **kwargs):
        a = op.axis
        if op.with_batch:
            a -= 1
        assert a >= 0 and a < len(x), f'invalid input {x} axis={op.axis}'
        s = split_to_secs(op.split, x[a])
        return [(*x[:a], n, *x[a+1:]) for n in s]
