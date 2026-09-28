import json
import os
import numpy as np
import sys

from main import parse_input_file, get_conditional_probabilities


# ============================================================
# Configuration
# ============================================================

CASE = int(sys.argv[1]) if len(sys.argv) > 1 else 1

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

INPUT_FILE = os.path.join(
    DATA_DIR,
    "inputs",
    f"{CASE}.json"
)

GROUND_TRUTH_FILE = os.path.join(
    DATA_DIR,
    "ground-truth",
    f"{CASE}.json"
)


# ============================================================
# Helpers
# ============================================================

def print_factor_list(factors):
    for i, factor in enumerate(factors):
        print(f"\nFactor {i}")
        print(factor)


def compare_predictions(predictions, ground_truth):
    all_passed = True

    print("\n========================================")
    print("COMPARING AGAINST GROUND TRUTH")
    print("========================================")

    for node, expected_values in ground_truth.items():

        print(f"\nNode {node}")

        # Check that a prediction actually exists
        if node not in predictions:
            print("Expected:", expected_values)
            print("Actual:   MISSING")
            print("FAIL")
            all_passed = False
            continue

        expected = np.array(expected_values, dtype=float)
        actual = np.array(predictions[node], dtype=float)

        print("Expected:", expected)
        print("Actual:  ", actual)

        # Shape/cardinality check
        if expected.shape != actual.shape:
            print(
                f"FAIL - shape mismatch "
                f"{actual.shape} != {expected.shape}"
            )
            all_passed = False
            continue

        passed = np.allclose(
            actual,
            expected,
            rtol=1e-6,
            atol=1e-6
        )

        if passed:
            print("PASS")
        else:
            print("FAIL")

            print(
                "Difference:",
                actual - expected
            )

            all_passed = False

    # Check for unexpected nodes
    extra_nodes = set(predictions.keys()) - set(ground_truth.keys())

    if extra_nodes:
        print("\nUnexpected prediction nodes:", extra_nodes)
        all_passed = False

    return all_passed


# ============================================================
# Main test
# ============================================================

def test_case(case):
    print("========================================")
    print(f"TESTING CASE {case}")
    print("========================================")

    # --------------------------------------------------------
    # Check files exist
    # --------------------------------------------------------

    assert os.path.exists(INPUT_FILE), (
        f"Input file not found: {INPUT_FILE}"
    )

    assert os.path.exists(GROUND_TRUTH_FILE), (
        f"Ground-truth file not found: {GROUND_TRUTH_FILE}"
    )

    # --------------------------------------------------------
    # Load assignment input
    # --------------------------------------------------------

    nodes, edges, evidence, factors = parse_input_file(
        input_file=INPUT_FILE
    )

    print("\n=== INPUT ===")

    print("Nodes:")
    print(nodes)

    print("\nEdges:")
    print(edges)

    print("\nEvidence:")
    print(evidence)

    print("\nNumber of factors:")
    print(len(factors))

    # Optional:
    # print_factor_list(factors)

    # --------------------------------------------------------
    # Run junction-tree inference
    # --------------------------------------------------------

    print("\n=== RUNNING INFERENCE ===")

    query_nodes, query_marginals = get_conditional_probabilities(
        all_nodes=nodes,
        evidence=evidence,
        edges=edges,
        factors=factors
    )

    print("Query nodes:", query_nodes)
    print("Number of marginals:", len(query_marginals))

    # --------------------------------------------------------
    # Basic sanity check
    # --------------------------------------------------------

    if len(query_nodes) != len(query_marginals):
        print("\nFAIL")
        print(
            "Number of query nodes does not match "
            "number of returned marginals."
        )
        print("query_nodes:", len(query_nodes))
        print("marginals:  ", len(query_marginals))

        return False

    # --------------------------------------------------------
    # Convert Factor objects into JSON-like predictions
    # --------------------------------------------------------

    predictions = {}

    print("\n=== PREDICTIONS ===")

    for node, marginal in zip(query_nodes, query_marginals):

        node_key = str(int(node))

        probabilities = np.array(
            marginal.val,
            dtype=float
        )

        predictions[node_key] = probabilities.tolist()

        print(f"\nNode {node_key}")
        print("var :", marginal.var)
        print("card:", marginal.card)
        print("val :", probabilities)
        print("sum :", np.sum(probabilities))

        # Final node marginals should normally be normalized
        if not np.isclose(
            np.sum(probabilities),
            1.0,
            atol=1e-6
        ):
            print(
                "WARNING: probabilities do not sum to 1"
            )

    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    with open(GROUND_TRUTH_FILE, "r") as f:
        ground_truth = json.load(f)

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    all_passed = compare_predictions(
        predictions,
        ground_truth
    )

    print("\n========================================")

    if all_passed:
        print(f"CASE {case}: ALL TESTS PASSED")
    else:
        print(f"CASE {case}: SOME TESTS FAILED")

    print("========================================")

    return all_passed


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    test_case(CASE)