from spn.structure.Base import get_nodes_by_type, Product, Leaf, assign_ids

from aqp_spn.aqp_leaves import Sum
from aqp_spn.custom_spflow.custom_validity import is_valid


def Prune(node, light=False):
    """
    Prunes spn. Ensures that nodes have at least one child and that types of node and children differ.
    Adapts weigths and optionally bloom filters accordingly.
    :param node:
    :return:
    """

    # v, err = is_valid(node)
    # assert v, err
    nodes = get_nodes_by_type(node, (Product, Sum))

    while len(nodes) > 0:
        n = nodes.pop()

        n_type = type(n)
        is_sum = n_type == Sum
        is_product = n_type == Product

        i = 0
        while i < len(n.children):
            c = n.children[i]

            # if my child has only one node, we can get rid of it and link directly to that grandchildren
            # in this case, no bloom filters can be lost since we do not split
            if not isinstance(c, Leaf) and len(c.children) == 1:
                n.children[i] = c.children[0]
                continue

            # if the type is similar to the type of the child
            if n_type == type(c):

                if is_sum and not light:
                    old_len = len(n.cluster_centers)
                    len_child_cluster = len(c.cluster_centers)
                    del n.cluster_centers[i]
                    n.cluster_centers.extend(c.cluster_centers)

                    assert old_len - 1 + len_child_cluster == len(
                        n.cluster_centers), "cluster_center length mismatch, node " + n + c

                del n.children[i]
                n.children.extend(c.children)

                if is_sum:
                    w = n.weights[i]
                    del n.weights[i]

                    n.weights.extend([cw * w for cw in c.weights])

                if is_product:
                    # hence, child type is also product and we should not loose bloom filter
                    if hasattr(n, 'binary_bloom_filters'):
                        n.binary_bloom_filters = {**n.binary_bloom_filters, **c.binary_bloom_filters}

                continue

            i += 1
        if is_sum and i > 0:
            n.weights[0] = 1.0 - sum(n.weights[1:])

    if isinstance(node, (Product, Sum)) and len(node.children) == 1:
        node = node.children[0]

    assign_ids(node)
    v, err = is_valid(node, light=light)
    assert v, err

    return node
