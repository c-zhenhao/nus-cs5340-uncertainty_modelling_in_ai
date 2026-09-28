import os
import numpy as np
import networkx as nx

from main import parse_input_file
from jt_construction import (
    _get_clique_factors,
    _get_jt_clique_and_edges,                         
)


BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def test_get_clique_factors():
    print("\n=== Testing _get_clique_factors ===")

    # Load the real Assignment 2 case
    input_file = os.path.join(
        BASE_DIR,
        "data",
        "inputs",
        "1.json"
    )

    nodes, edges, evidence, factors = parse_input_file(input_file)

    # Manually provide the maximal cliques for case 1.
    # This isolates _get_clique_factors() from
    # _get_jt_clique_and_edges().
    jt_cliques = [
        [0, 2],
        [1, 2],
        [2, 3, 4],
        [3, 4, 5],
    ]

    clique_factors = _get_clique_factors(
        jt_cliques,
        factors
    )

    print("Number of clique factors:", len(clique_factors))

    for i, factor in enumerate(clique_factors):
        print(f"\nClique {i}: {jt_cliques[i]}")
        print(factor)
        print("var :", factor.var)
        print("card:", factor.card)
        print("val :", factor.val)

    # --------------------------------------------------
    # Basic checks
    # --------------------------------------------------

    assert len(clique_factors) == 4

    # Clique [0, 2]
    assert np.array_equal(
        clique_factors[0].var,
        np.array([0, 2])
    )

    assert np.array_equal(
        clique_factors[0].card,
        np.array([2, 2])
    )

    assert np.allclose(
        clique_factors[0].val,
        np.array([1, 2, 3, 4])
    )

    # --------------------------------------------------
    # Clique [1, 2]
    # --------------------------------------------------

    assert np.array_equal(
        clique_factors[1].var,
        np.array([1, 2])
    )

    assert np.array_equal(
        clique_factors[1].card,
        np.array([2, 2])
    )

    assert np.allclose(
        clique_factors[1].val,
        np.array([4, 6, 2, 1])
    )

    # --------------------------------------------------
    # Clique [2, 3, 4]
    #
    # Receives:
    #   factor [2,3]
    #   factor [2,4]
    #   factor [3,4]
    #
    # because [3,4] fits this clique first.
    # --------------------------------------------------

    assert np.array_equal(
        clique_factors[2].var,
        np.array([2, 3, 4])
    )

    assert np.array_equal(
        clique_factors[2].card,
        np.array([2, 2, 2])
    )

    expected_clique_2 = np.array([
        100,
        40,
        360,
        324,
        20,
        30,
        64,
        216,
    ])

    assert np.allclose(
        clique_factors[2].val,
        expected_clique_2
    )

    # --------------------------------------------------
    # Clique [3, 4, 5]
    #
    # Receives:
    #   factor [3,5]
    #   factor [4,5]
    # --------------------------------------------------

    assert np.array_equal(
        clique_factors[3].var,
        np.array([3, 4, 5])
    )

    assert np.array_equal(
        clique_factors[3].card,
        np.array([2, 2, 2])
    )

    expected_clique_3 = np.array([
        8,
        6,
        28,
        21,
        16,
        8,
        4,
        2,
    ])

    assert np.allclose(
        clique_factors[3].val,
        expected_clique_3
    )

    print("\n_get_clique_factors PASSED")


def test_get_jt_clique_and_edges_non_chordal():
    print("\n=== Testing triangulation on non-chordal graph ===")

    # Chordless 4-cycle:
    #
    # 0 ----- 1
    # |       |
    # |       |
    # 3 ----- 2

    nodes = np.array([0, 1, 2, 3])

    edges = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
    ])

    jt_cliques, jt_edges = _get_jt_clique_and_edges(
        nodes,
        edges
    )

    print("\nCliques:")
    for i, clique in enumerate(jt_cliques):
        print(f"{i}: {clique}")

    print("\nJT edges:")
    print(jt_edges)

    # After triangulation, the square should become
    # two maximal cliques of size 3.
    assert len(jt_cliques) == 2

    for clique in jt_cliques:
        assert len(clique) == 3

    print("Clique test PASSED")

    # All original variables must still appear
    variables = set()

    for clique in jt_cliques:
        variables.update(clique)

    assert variables == set(nodes)

    print("Variable coverage test PASSED")

    # The two cliques should share the added diagonal,
    # so their separator has size 2.
    separator = (
        set(jt_cliques[0])
        & set(jt_cliques[1])
    )

    assert len(separator) == 2

    print("Separator test PASSED")

    # One undirected tree edge should be represented
    # by two directed edges.
    directed_edges = {
        (int(i), int(j))
        for i, j in jt_edges
    }

    assert len(directed_edges) == 2

    for i, j in directed_edges:
        assert (j, i) in directed_edges

    print("Bidirectional edge test PASSED")

    # Check that the result really forms a tree
    tree = nx.Graph()
    tree.add_nodes_from(range(len(jt_cliques)))
    tree.add_edges_from(directed_edges)

    assert nx.is_tree(tree)

    print("Tree test PASSED")

    # Running intersection property
    for node in nodes:
        containing_cliques = [
            i
            for i, clique in enumerate(jt_cliques)
            if node in clique
        ]

        if len(containing_cliques) <= 1:
            continue

        subtree = tree.subgraph(containing_cliques)

        assert nx.is_connected(subtree), (
            f"Running intersection failed for node {node}"
        )

    print("Running intersection test PASSED")

    print("\nNon-chordal triangulation test PASSED")


if __name__ == "__main__":
    test_get_clique_factors()
    test_get_jt_clique_and_edges_non_chordal()

    print("\n==============================")
    print("ALL JT CONSTRUCTION TESTS PASSED")
    print("==============================")