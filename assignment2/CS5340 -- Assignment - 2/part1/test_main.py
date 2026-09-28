import copy
import json
import os
import numpy as np

from main import (
    parse_input_file,
    construct_junction_tree,
    _update_mrf_w_evidence,
    _get_clique_potentials,
    _get_node_marginal_probabilities,
)

from factor_utils import (
    factor_product,
    factor_marginalize,
)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def test_update_mrf_w_evidence():
    print("\n=== Testing _update_mrf_w_evidence ===")

    input_file = os.path.join(
        BASE_DIR,
        "data",
        "inputs",
        "1.json"
    )

    nodes, edges, _, factors = parse_input_file(input_file)

    # --------------------------------------------------
    # Test 1: observe X2 = 1
    # --------------------------------------------------

    evidence = {2: 1}

    query_nodes, updated_edges, updated_factors = (
        _update_mrf_w_evidence(
            all_nodes=nodes,
            evidence=evidence,
            edges=edges,
            factors=factors,
        )
    )

    print("\nEvidence:", evidence)
    print("Query nodes:", query_nodes)
    print("Updated edges:")
    print(updated_edges)

    print("\nUpdated factors:")
    for i, factor in enumerate(updated_factors):
        print(f"\nFactor {i}")
        print(factor)

    # --------------------------------------------------
    # Query nodes
    # --------------------------------------------------

    expected_nodes = np.array([
        0, 1, 3, 4, 5
    ])

    assert np.array_equal(
        query_nodes,
        expected_nodes
    )

    print("Query node test PASSED")

    # --------------------------------------------------
    # Edges touching node 2 should disappear
    # --------------------------------------------------

    expected_edges = np.array([
        [3, 4],
        [3, 5],
        [4, 5],
    ])

    assert np.array_equal(
        updated_edges,
        expected_edges
    )

    print("Updated edge test PASSED")

    # --------------------------------------------------
    # Number of factors
    #
    # No factor becomes completely empty for evidence {2: 1},
    # so all 7 factors should remain.
    # --------------------------------------------------

    assert len(updated_factors) == 7

    print("Factor count test PASSED")

    # --------------------------------------------------
    # Original factor [0,2]
    #
    # [1,2,3,4]
    #
    # X2 = 1 leaves:
    # X0=0 -> 3
    # X0=1 -> 4
    # --------------------------------------------------

    assert np.array_equal(
        updated_factors[0].var,
        np.array([0])
    )

    assert np.array_equal(
        updated_factors[0].card,
        np.array([2])
    )

    assert np.allclose(
        updated_factors[0].val,
        np.array([3, 4])
    )

    print("Factor [0,2] evidence test PASSED")

    # --------------------------------------------------
    # Original factor [1,2]
    #
    # [4,6,2,1]
    #
    # X2 = 1 leaves [2,1]
    # --------------------------------------------------

    assert np.array_equal(
        updated_factors[1].var,
        np.array([1])
    )

    assert np.allclose(
        updated_factors[1].val,
        np.array([2, 1])
    )

    print("Factor [1,2] evidence test PASSED")

    # --------------------------------------------------
    # Original factor [2,3]
    #
    # [10,5,8,9]
    #
    # X2 = 1 leaves [5,9]
    # --------------------------------------------------

    assert np.array_equal(
        updated_factors[2].var,
        np.array([3])
    )

    assert np.allclose(
        updated_factors[2].val,
        np.array([5, 9])
    )

    print("Factor [2,3] evidence test PASSED")

    # --------------------------------------------------
    # Original factor [2,4]
    #
    # [5,4,2,6]
    #
    # X2 = 1 leaves [4,6]
    # --------------------------------------------------

    assert np.array_equal(
        updated_factors[3].var,
        np.array([4])
    )

    assert np.allclose(
        updated_factors[3].val,
        np.array([4, 6])
    )

    print("Factor [2,4] evidence test PASSED")

    # --------------------------------------------------
    # Factors not involving node 2 should be unchanged
    # --------------------------------------------------

    for i in range(4, 7):
        assert np.array_equal(
            updated_factors[i].var,
            factors[i].var
        )

        assert np.array_equal(
            updated_factors[i].card,
            factors[i].card
        )

        assert np.allclose(
            updated_factors[i].val,
            factors[i].val
        )

    print("Unaffected factor test PASSED")

    print("\n_update_mrf_w_evidence PASSED")


def test_update_mrf_no_evidence():
    print("\n=== Testing empty evidence ===")

    input_file = os.path.join(
        BASE_DIR,
        "data",
        "inputs",
        "1.json"
    )

    nodes, edges, _, factors = parse_input_file(input_file)

    query_nodes, updated_edges, updated_factors = (
        _update_mrf_w_evidence(
            all_nodes=nodes,
            evidence={},
            edges=edges,
            factors=factors,
        )
    )

    assert np.array_equal(
        query_nodes,
        nodes
    )

    assert np.array_equal(
        updated_edges,
        edges
    )

    assert len(updated_factors) == len(factors)

    for original, updated in zip(
        factors,
        updated_factors
    ):
        assert np.array_equal(
            original.var,
            updated.var
        )

        assert np.array_equal(
            original.card,
            updated.card
        )

        assert np.allclose(
            original.val,
            updated.val
        )

    print("Empty evidence test PASSED")


def test_get_clique_potentials():
    print("\n=== Testing _get_clique_potentials ===")

    input_file = os.path.join(
        BASE_DIR,
        "data",
        "inputs",
        "1.json"
    )

    nodes, edges, evidence, factors = parse_input_file(
        input_file
    )

    # --------------------------------------------------
    # Prepare MRF
    # --------------------------------------------------

    query_nodes, updated_edges, updated_factors = (
        _update_mrf_w_evidence(
            all_nodes=nodes,
            evidence=evidence,
            edges=edges,
            factors=factors,
        )
    )

    # --------------------------------------------------
    # Construct junction tree
    # --------------------------------------------------

    jt_cliques, jt_edges, jt_factors = (
        construct_junction_tree(
            nodes=query_nodes,
            edges=updated_edges,
            factors=updated_factors,
        )
    )

    print("\nJT cliques:")
    for i, clique in enumerate(jt_cliques):
        print(f"{i}: {clique}")

    print("\nJT edges:")
    print(jt_edges)

    # --------------------------------------------------
    # Run sum-product
    # --------------------------------------------------

    clique_potentials = _get_clique_potentials(
        jt_cliques=jt_cliques,
        jt_edges=jt_edges,
        jt_clique_factors=jt_factors,
    )

    assert len(clique_potentials) == len(jt_cliques)

    # --------------------------------------------------
    # Construct brute-force global joint
    # --------------------------------------------------

    joint = copy.deepcopy(updated_factors[0])

    for factor in updated_factors[1:]:
        joint = factor_product(
            joint,
            factor
        )

    print("\nFull joint:")
    print("var :", joint.var)
    print("card:", joint.card)
    print("sum :", np.sum(joint.val))

    partition_function = np.sum(joint.val)

    # --------------------------------------------------
    # Each clique potential should equal the
    # unnormalized marginal of the full joint
    # onto that clique.
    # --------------------------------------------------

    for i, clique in enumerate(jt_cliques):

        potential = clique_potentials[i]

        vars_to_eliminate = np.setdiff1d(
            joint.var,
            np.array(clique)
        )

        if len(vars_to_eliminate) == 0:
            expected = copy.deepcopy(joint)
        else:
            expected = factor_marginalize(
                joint,
                vars_to_eliminate
            )

        print(f"\nClique {i}: {clique}")

        print("Expected:")
        print(expected)

        print("Actual:")
        print(potential)

        assert np.array_equal(
            potential.var,
            expected.var
        )

        assert np.array_equal(
            potential.card,
            expected.card
        )

        assert np.allclose(
            potential.val,
            expected.val,
            rtol=1e-6,
            atol=1e-6,
        )

        # Every calibrated clique potential should
        # sum to the same global partition function.
        assert np.isclose(
            np.sum(potential.val),
            partition_function,
            rtol=1e-6,
            atol=1e-6,
        )

        print(f"Clique {i} PASSED")

    print("\n_get_clique_potentials PASSED")


def test_get_node_marginal_probabilities():
    print("\n=== Testing _get_node_marginal_probabilities ===")

    input_file = os.path.join(
        BASE_DIR,
        "data",
        "inputs",
        "1.json"
    )

    ground_truth_file = os.path.join(
        BASE_DIR,
        "data",
        "ground-truth",
        "1.json"
    )

    nodes, edges, evidence, factors = parse_input_file(
        input_file
    )

    # Update MRF
    query_nodes, updated_edges, updated_factors = (
        _update_mrf_w_evidence(
            all_nodes=nodes,
            evidence=evidence,
            edges=edges,
            factors=factors,
        )
    )

    # Construct JT
    jt_cliques, jt_edges, jt_factors = (
        construct_junction_tree(
            nodes=query_nodes,
            edges=updated_edges,
            factors=updated_factors,
        )
    )

    # Sum-product
    clique_potentials = _get_clique_potentials(
        jt_cliques=jt_cliques,
        jt_edges=jt_edges,
        jt_clique_factors=jt_factors,
    )

    # Get node marginals
    marginals = _get_node_marginal_probabilities(
        query_nodes=query_nodes,
        cliques=jt_cliques,
        clique_potentials=clique_potentials,
    )

    assert len(marginals) == len(query_nodes)

    # Load supplied ground truth
    with open(ground_truth_file, "r") as f:
        ground_truth = json.load(f)

    for node, marginal in zip(query_nodes, marginals):

        expected = np.array(
            ground_truth[str(int(node))],
            dtype=float
        )

        actual = np.array(
            marginal.val,
            dtype=float
        )

        print(f"\nNode {node}")
        print("Expected:", expected)
        print("Actual:  ", actual)
        print("Sum:     ", np.sum(actual))

        # Should be a unary factor
        assert np.array_equal(
            marginal.var,
            np.array([node])
        )

        # Probabilities should sum to 1
        assert np.isclose(
            np.sum(actual),
            1.0,
            atol=1e-6
        )

        # Compare against assignment ground truth
        assert np.allclose(
            actual,
            expected,
            rtol=1e-6,
            atol=1e-6,
        )

        print(f"Node {node} PASSED")

    print("\n_get_node_marginal_probabilities PASSED")


if __name__ == "__main__":
    test_update_mrf_w_evidence()
    test_update_mrf_no_evidence()
    test_get_clique_potentials()
    test_get_node_marginal_probabilities()

    print("\n============================")
    print("ALL MAIN HELPER TESTS PASSED")
    print("============================")