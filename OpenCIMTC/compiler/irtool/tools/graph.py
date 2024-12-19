
from ..core.jsonable import Jsonable

class BaseElement(Jsonable):

    def __init__(self, name, *, group=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.group = group


class Node(BaseElement):

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self._inputs = set()
        self._outputs = set()

    def get_inputs(self):
        return tuple(self._inputs)

    def get_outputs(self):
        return tuple(self._outputs)


class Edge(BaseElement):

    def __init__(self, f_name, t_name, **kwargs):
        super().__init__((f_name, t_name), **kwargs)


class Group(BaseElement):
    pass


class DagGraph(Jsonable):

    def __init__(self):
        super().__init__()
        self.nodes = {}
        self.edges = []
        self.groups = {}

    def add_node(self, name, *, group=None, **kwargs):
        assert name not in self.nodes
        assert group is None or group in self.groups
        self.nodes[name] = Node(name, group=group, **kwargs)

    def add_edge(self, f_name, t_name, **kwargs):
        fn, tn = self.nodes[f_name], self.nodes[t_name]
        fn._outputs.add(t_name)
        tn._inputs.add(f_name)
        self.edges.append(Edge(f_name, t_name, **kwargs))

    def add_group(self, name, *, group=None, **kwargs):
        assert name not in self.groups
        assert group is None or group in self.groups
        self.groups[name] = Group(name, group=group, **kwargs)

    def get_node(self, name):
        return self.nodes[name]

    def get_inputs(self, name):
        return tuple(self.nodes[name].inputs)

    def get_outputs(self, name):
        return tuple(self.nodes[name].outputs)

    def sorted_nodes(self):
        order = {n: 0 for n in self.nodes}

        def score(nd):
            for n in nd._outputs:
                order[n] += 1
                score(self.nodes[n])

        for node in self.nodes.values():
            score(node)
        order = [(v, n) for n, v in order.items()]
        order.sort()
        return [self.nodes[n] for v, n in order]

    def iter_groups(self, name=None, level=0):
        for n, g in self.groups.items():
            if g.group == name:
                yield level + 1, g
                yield from self.iter_groups(n, level + 1)

    def iter_nodes(self):
        yield from self.nodes.values()

    def iter_edges(self):
        yield from self.edges
