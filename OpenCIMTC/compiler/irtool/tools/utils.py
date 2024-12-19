from ..core.type_util import to_int_tuple

VALID_PAD = 'VALID'
SAME_PAD = 'SAME'


def to_auto_pad(dims, kernel, stride, padding, dilation):
    ndim = len(dims)
    p = to_int_tuple(padding, ndim=ndim * 2)
    if not any(p):
        return VALID_PAD
    x = dims
    k = to_int_tuple(kernel, ndim=ndim)
    s = to_int_tuple(stride, ndim=ndim)
    d = to_int_tuple(dilation, ndim=ndim)
    dk = tuple(d[i] * (k[i] - 1) + 1 for i in range(ndim))
    q = [0] * ndim
    for i in range(ndim):
        if x[i] % s[i] == 0:
            q[i] = max(dk[i] - s[i], 0)
        else:
            q[i] = max(dk[i] - (x[i] % s[i]), 0)
    if all(p[i] == q[i] // 2 and p[i] + p[ndim + i] == q[i]
           for i in range(ndim)):
        return SAME_PAD
    assert False, f'not VALID or SAME padding: input {x}, ' \
        f'kernel {k}, stride {s}, padding {padding}, dilation {dilation}'


def _auto_padding(auto_pad, dims, dilated_kernel, stride):
    ndim = len(dims)
    x, dk, s = dims, dilated_kernel, stride
    p = [0] * (ndim * 2)
    if auto_pad == VALID_PAD:
        pass
    elif auto_pad == SAME_PAD:
        for i in range(ndim):
            q = max(dk[i] - (x[i] % s[i] or s[i]), 0)
            p[i] = q // 2
            p[ndim + i] = q - p[i]
    else:
        assert False, f'invalid auto_pad {auto_pad!r}'
    return tuple(p)


def auto_pad_pool(auto_pad, dims, kernel, stride, dilation):
    if auto_pad == VALID_PAD:
        return 0
    ndim = len(dims)
    x = dims
    k = to_int_tuple(kernel, ndim=ndim)
    s = to_int_tuple(stride, ndim=ndim)
    d = to_int_tuple(dilation, ndim=ndim)
    dk = tuple(d[i] * (k[i] - 1) + 1 for i in range(ndim))
    if auto_pad == SAME_PAD:
        y = tuple((x[i] + s[i] - 1) // s[i] for i in range(ndim))
        q = tuple((y[i] - 1) * s[i] + dk[i] - x[i] for i in range(ndim))
        return tuple(q[i] // 2 for i in range(ndim))
    assert False, f'invalid auto_pad {auto_pad!r}'


def pool_shapes(dims, kernel, stride, padding, dilation, auto_pad,
                ceil_mode=False):
    ndim = len(dims)
    x = dims
    k = to_int_tuple(kernel, ndim=ndim)
    s = to_int_tuple(stride, ndim=ndim)
    d = to_int_tuple(dilation, ndim=ndim)
    dk = tuple(d[i] * (k[i] - 1) + 1 for i in range(ndim))
    if padding is None:
        p = _auto_padding(auto_pad, x, dk, s)
    else:
        p = to_int_tuple(padding, ndim=ndim * 2)
    if ceil_mode:
        os = [(x[i] + p[i] + p[ndim + i] - dk[i] + s[i] - 1) // s[i] + 1
              for i in range(ndim)]
        for i in range(ndim):
            if (os[i] - 1) * s[i] >= x[i] + p[i]:
                os[i] -= 1
        os = tuple(os)
    else:
        os = tuple((x[i] + p[i] + p[ndim + i] - dk[i]) // s[i] + 1
                   for i in range(ndim))
    return os, (k, s, p, d, dk)


def conv_shapes(dims, kernel, stride, padding, dilation, auto_pad):
    return pool_shapes(dims, kernel, stride, padding, dilation, auto_pad)


def conv_t_shapes(dims, kernel, stride, padding, dilation, output_padding,
                  auto_pad):
    ndim = len(dims)
    x = dims
    k = to_int_tuple(kernel, ndim=ndim)
    s = to_int_tuple(stride, ndim=ndim)
    d = to_int_tuple(dilation, ndim=ndim)
    a = to_int_tuple(output_padding, ndim=ndim)
    dk = tuple(d[i] * (k[i] - 1) + 1 for i in range(ndim))
    if padding is None:
        p = _auto_padding(auto_pad, x, dk, s)
    else:
        p = to_int_tuple(padding, ndim=ndim*2)
    dp = tuple(dk[i % ndim] - p[i] - 1 for i in range(ndim*2))
    assert all(i >= 0 for i in dp), \
        f'invalid padding {padding} for kernel {kernel} dilation {dilation}'
    di = tuple((x[i] - 1) * s[i] + 1 + dp[i] + dp[ndim + i] + a[i]
               for i in range(ndim))
    os = tuple((x[i] - 1) * s[i] - p[i] - p[ndim + i] + dk[i] + a[i]
               for i in range(ndim))
    return os, (k, s, p, d, dk, dp, di)


def resize_size_scale(dims, size, scale):
    ndim = len(dims)
    if size is not None:
        assert scale is None
        if isinstance(size, int):
            size = (size,) * ndim
        else:
            assert len(size) == ndim, \
                f'invalid resize {dims} to size {size}'
            size = tuple(size)
        scale = tuple(size[i] / dims[i] for i in range(ndim))
    else:
        if isinstance(scale, (float, int)):
            scale = (scale,) * ndim
        else:
            assert len(scale) == ndim, \
                f'invalid resize {dims} with scale {scale}'
            scale = tuple(scale)
        size = tuple(int(dims[i] * scale[i]) for i in range(ndim))
    assert all(s > 0 for s in size), f'invalid resize {dims} to size {size}'
    return size, scale


def concat_axis(rank, channel_last, axis, with_batch, channel_pos):
    if axis < 0:
        a = rank + axis
    elif with_batch:
        a = axis - 1
    else:
        a = axis
    assert 0 <= a < rank, f'invalid concat rank {rank} axis {axis}'
    assert channel_pos in ('first', 'last', 'ignore'), \
        f'invalid channel_pos {channel_pos!r}'
    if channel_pos == 'ignore':
        pass
    elif channel_last ^ (channel_pos == 'last'):
        a = (a - 1) % rank if channel_last else (a + 1) % rank
    return a


def split_to_secs(split, size):
    assert size >= 0, f'invalid split size={size}'
    if isinstance(split, int):
        assert split > 0, f'invalid split split={split}'
        n = split
        m = (size + n - 1) // n
        s = [m] * (n - 1) + [size - m * (n - 1)]
    else:
        assert sum(split) == size, \
               f'invalid split {size} to {split}'
        s = split
    return tuple(s)
