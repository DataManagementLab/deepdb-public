from spn.structure.Base import Node


class Sum(Node):
    def __init__(self, weights=None, children=None, cluster_centers=None, cardinality=None):
        Node.__init__(self)

        if weights is None:
            weights = []
        self.weights = weights

        if children is None:
            children = []
        self.children = children

        if cluster_centers is None:
            cluster_centers = []
        self.cluster_centers = cluster_centers

        if cardinality is None:
            cardinality = 0
        self.cardinality = cardinality

    @property
    def parameters(self):
        sorted_children = sorted(self.children, key=lambda c: c.id)
        params = [(n.id, self.weights[i]) for i, n in enumerate(sorted_children)]
        return tuple(params)
