import pickle
import re

RE_DTYPE = re.compile(r'^(float|f|uint|u|int|i)(\d+)$', re.I)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def parse_dtype(dtype, scale=0):
    if dtype is None:
        return None, scale
    m = RE_DTYPE.match(dtype)
    if m:
        tag, bits = m.group(1)[0].lower(), int(m.group(2))
        if tag == 'f' and bits in (16, 32, 64):
            return f'float{bits}', scale
        if tag == 'u' and 0 < bits <= 64 and 0 <= scale <= 2**bits:
            assert 0 <= scale <= 2**bits, f'invalid scale {scale}'
            scale = max(2**7, 2**bits, scale)
            return f'uint{(bits + 7) // 8 * 8}', max(2**8, 2**bits, scale)
        if tag == 'i' and 0 < bits <= 64 and abs(scale) <= 2**(bits-1):
            assert 0 <= abs(scale) <= 2**(bits-1), f'invalid scale {scale}'
            s = max(2**7, 2**(bits-1), abs(scale))
            scale = -s if scale < 0 else s
            return f'int{(bits + 7) // 8 * 8}', scale
    raise ValueError(f'invalid dtype {dtype!r}')


class Same:

    __close = object()

    def __init__(self, value):
        self.reset(value)

    def reset(self, value):
        if value in ('close', 'Close', 'CLOSE', self.__close):
            value = self.__close
        elif isinstance(value, Same):
            value = value.__value
        else:
            value = bool(value)
        self.__value = value

    def __bool__(self):
        return self.__value in (True, self.__close)

    def isclose(self):
        return self.__value is self.__close

    def isdifferent(self):
        return self.__value in (False, self.__close)

    def __iand__(self, other):
        other = Same(other)
        a, b = self.__value, other.__value
        if a is self.__close and b in (True, self.__close):
            pass
        elif b is self.__close and a is True:
            a = b
        else:
            a = a and b
        self.__value = a
        return self

    def symbol(self):
        b = self.__value
        return '~=' if b is self.__close else '==' if b else '!='

    def __str__(self):
        if self.__value is self.__close:
            return 'Close'
        else:
            return str(self.__value)
