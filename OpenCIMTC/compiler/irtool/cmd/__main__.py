import os
import sys
from argparse import ArgumentParser
from .base import CmdReg
from ..core.type_util import to_boolean


def import_modules(imports):
    if not imports:
        return
    for name in imports.split(':'):
        if name:
            __import__(name)


def main():
    import_modules(os.environ.get('MODULES'))

    argp = ArgumentParser('python -m irtool.cmd')
    argp.add_argument('-D', '--debug', action='store_true',
                      help='dump exception')
    argp.add_argument('-M', '--modules',
                      help='":" seperated module names, env MODULES')
    argp.set_defaults(cmd_cls=None)
    subp = argp.add_subparsers(title='commands')
    for cmd, cmd_cls in CmdReg.iter_reg():
        p = subp.add_parser(cmd, help=cmd_cls.help)
        cmd_cls.add_args(p)
        p.set_defaults(cmd_cls=cmd_cls)
    args = argp.parse_args()

    if args.cmd_cls is None:
        argp.print_help()
        sys.exit(1)

    if to_boolean(os.environ.get('DEBUG')):
        args.debug = True

    import_modules(args.modules)

    cmd = args.cmd_cls(args)
    r = None
    try:
        r = cmd.run()
    except KeyboardInterrupt:
        cmd.info('canceled')
        if args.debug:
            raise
    except Exception as e:
        cmd.error(e)
        if args.debug:
            raise
        else:
            r = 1
    sys.exit(r)


if __name__ == '__main__':
    main()
