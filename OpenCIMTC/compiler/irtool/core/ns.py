from contextlib import contextmanager
import threading


_local = threading.local()


@contextmanager
def ns_push(name):
    ns = getattr(_local, 'namespace', None)
    if ns is None:
        ns = []
        _local.namespace = ns
    ns.append(name)
    yield tuple(ns)
    ns.pop()


def ns_get(sep='/'):
    ns = getattr(_local, 'namespace', ())
    return sep.join(ns)
