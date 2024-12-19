from copy import deepcopy

_themes = {
    'default': {
        'node': {
            'shape': 'box', 'style': 'rounded,filled',
            'fillcolor': 'cyan', 'color': 'cyan',
        },
        'edge': {
            'color': 'blue',
        },
        'cluster': {
            'color': 'gray90', 'labeljust': 'r',
        },
        'input': {
            'fillcolor': 'green',
        },
        'output': {
            'fillcolor': 'green'
        },
        'block-io': {
            'style': 'dotted'
        }
    }
}


def get_attrs(theme, obj, bracket=True, **kwargs):
    if isinstance(obj, str):
        key = obj
    else:
        key = getattr(obj, 'tag', None)
    attrs = {}
    for k, v in theme.get(key, {}).items():
        if v not in (None, ''):
            attrs[k] = v
    for k, v in kwargs.items():
        if v not in (None, ''):
            attrs[k] = v
    if attrs:
        if bracket:
            yield '['
        for k, v in attrs.items():
            yield f'{k}=\"{v}\"'
        if bracket:
            yield ']'


def graph_to_dot(grf, theme='default', label=None, **kwargs):

    theme = deepcopy(_themes[theme])
    for k, v in kwargs.items():
        if k not in theme:
            theme[k] = v
        else:
            theme[k].update(v)

    if label is None:
        def label(n, v):
            return

    print('digraph', '{')
    print(' ', 'node', *get_attrs(theme, 'node'))
    print(' ', 'edge', *get_attrs(theme, 'edge'))
    nds = grf.sorted_nodes()
    for v in nds:
        if v.group is None:
            print(' ', f'\"{v.name}\"', *get_attrs(theme, v,
                  label=label(v.name, v)))
    pre = 0
    for lev, g in grf.iter_groups():
        for i in range(pre, lev - 1, -1):
            print(' ' * (i * 2 - 1), '}')
        ind = ' ' * (lev * 2 - 1)
        print(ind, 'subgraph', f'\"cluster_{lev}_{g.name}\"', '{')
        for a in get_attrs(theme, 'cluster', False):
            print(ind, ' ', a)
        for a in get_attrs(theme, g, False, label=label(g.name, g)):
            print(ind, ' ', a)
        for v in nds:
            if v.group == g.name:
                print(ind, ' ', f'\"{v.name}\"', *get_attrs(theme, v))
        pre = lev
    for i in range(pre, 0, -1):
        print(' ' * (2 * i - 1), '}')
    for e in grf.iter_edges():
        print(' ', f'\"{e.name[0]}\"', '->', f'\"{e.name[1]}\"',
              *get_attrs(theme, e))
    print('}')
