from .base import BaseCode
from contextlib import contextmanager
from pathlib import Path
from io import StringIO
from types import ModuleType


def comma(args):
    return ', '.join(args)


class PythonCode(BaseCode):

    def __init__(self, *, indent=None):
        self._indent = ' ' * 4 if indent is None else indent
        self._level = 0

    @contextmanager
    def indent(self, level=1):
        self._level += level
        yield
        self._level -= level

    def to_code(self, generator, file=None):
        if isinstance(file, (str, Path)):
            with open(file, 'w') as f:
                return self.to_code(generator, file=f)

        f = StringIO() if file is None else file

        for line in generator:
            if line is None:
                f.write('\n')
            else:
                f.write(self._indent * self._level)
                f.write(line)
                f.write('\n')
                f.flush()

        if file is None:
            return f.getvalue()

    def compile(self, name, code):
        if not isinstance(code, str):
            code = self.to_code(code)
        ast = compile(code, f'<{name}>', 'exec')
        mod = ModuleType(name)
        exec(ast, mod.__dict__)
        return mod
