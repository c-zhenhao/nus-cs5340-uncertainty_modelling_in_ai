import copy
import numpy as np
import networkx as nx
from networkx.algorithms import tree
from factor import Factor
from factor_utils import factor_product


""" ADD HELPER FUNCTIONS HERE (IF NEEDED) """


""" END ADD HELPER FUNCTIONS HERE """


def _get_clique_factors(jt_cliques, factors):
    """
    Assign node factors to cliques in the junction tree and derive the clique factors.

    Args:
        jt_cliques: list of junction tree maximal cliques e.g. [[x1, x2, x3], [x2, x3], ... ]
        factors: list of factors from the original graph

    Returns:
        list of clique factors where the factor(jt_cliques[i]) = clique_factors[i]
    """
    clique_factors = [Factor() for _ in jt_cliques]

    """ YOUR CODE HERE """
    for factor in factors:
        factor_vars = set(factor.var)
        assigned = False

        for i, clique in enumerate(jt_cliques):
            clique_vars = set(clique)

            if factor_vars.issubset(clique_vars):

                if clique_factors[i].is_empty():
                    clique_factors[i] = copy.deepcopy(factor)
                else:
                    clique_factors[i] = factor_product(clique_factors[i], factor)

                assigned = True
                break

        assert assigned, (
            f"No clique contains factor variables "
            f"{factor.var}"
        )

    """ END YOUR CODE HERE """

    assert len(clique_factors) == len(jt_cliques), 'there should be equal number of cliques and clique factors'
    return clique_factors


def _get_jt_clique_and_edges(nodes, edges):
    """
    Construct the structure of the junction tree and return the list of cliques (nodes) in the junction tree and
    the list of edges between cliques in the junction tree. [i, j] in jt_edges means that cliques[j] is a neighbor
    of cliques[i] and vice versa. [j, i] should also be included in the numpy array of edges if [i, j] is present.
    You can use nx.Graph() and nx.find_cliques().

    Args:
        nodes: numpy array of nodes [x1, ..., xN]
        edges: numpy array of edges e.g. [x1, x2] implies that x1 and x2 are neighbors.

    Returns:
        list of junction tree cliques. each clique should be a maximal clique. e.g. [[X1, X2], ...]
        numpy array of junction tree edges e.g. [[0,1], ...], [i,j] means that cliques[i]
            and cliques[j] are neighbors.
    """
    jt_cliques = []
    # jt_edges = np.array(edges)  # dummy value

    """ YOUR CODE HERE """
    # build original undirected graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # from lecture 5, triangulation first before clique construction
    # complete_to_chordal_graph returns the chordal graph and an alpha dict (which we can ignore)
    G_chordal, _ = nx.complete_to_chordal_graph(G)

    # 1. find maximal cliques
    jt_cliques = [
        sorted(list(c))
        # nx.find_cliques returns a generator, so we iterate through it and sort them into standard lists
        for c in nx.find_cliques(G_chordal)  
    ]

    # 2. build clique-intersection graph
    clique_graph = nx.Graph()

    for i in range(len(jt_cliques)):
        clique_graph.add_node(i)

    for i in range(len(jt_cliques)):
        for j in range(i + 1, len(jt_cliques)):

            separator = set(jt_cliques[i]).intersection(
                jt_cliques[j]
            )

            if len(separator) > 0:
                clique_graph.add_edge(
                    i,
                    j,
                    weight=len(separator)
                )

    # 3. take maximum spanning tree
    # nx.maximum_spanning_tree expects a weight attribute to maximize
    mst = nx.maximum_spanning_tree(
        clique_graph,
        weight="weight"
    )

    # 4. store both directions
    jt_edges = []
    
    for i, j in mst.edges():
        jt_edges.append([i, j])
        jt_edges.append([j, i])

    if len(jt_edges) == 0:
        jt_edges = np.empty((0, 2), dtype=np.int64)
    else:
        jt_edges = np.array(jt_edges, dtype=np.int64)

    """ END YOUR CODE HERE """

    return jt_cliques, jt_edges


def construct_junction_tree(nodes, edges, factors):
    """
    Constructs the junction tree and returns its the cliques, edges and clique factors in the junction tree.
    DO NOT EDIT THIS FUNCTION.

    Args:
        nodes: numpy array of random variables e.g. [X1, X2, ..., Xv]
        edges: numpy array of edges e.g. [[X1,X2], [X2,X1], ...]
        factors: list of factors in the graph

    Returns:
        list of cliques e.g. [[X1, X2], ...]
        numpy array of edges e.g. [[0,1], ...], [i,j] means that cliques[i] and cliques[j] are neighbors.
        list of clique factors where jt_cliques[i] has factor jt_factors[i] where i is an index
    """
    jt_cliques, jt_edges = _get_jt_clique_and_edges(nodes=nodes, edges=edges)
    jt_factors = _get_clique_factors(jt_cliques=jt_cliques, factors=factors)
    return jt_cliques, jt_edges, jt_factors
