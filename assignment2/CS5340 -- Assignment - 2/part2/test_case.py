import sys
import json
import os
import numpy as np

from main import _get_learned_parameters


CASE = int(sys.argv[1]) if len(sys.argv) > 1 else 1

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
OBS_DIR = os.path.join(BASE_DIR, "data", "observations")
ANSWER_DIR = os.path.join(BASE_DIR, "data", "answers")


with open(os.path.join(OBS_DIR, f"{CASE}.json")) as f:
    config = json.load(f)

with open(os.path.join(ANSWER_DIR, f"{CASE}.json")) as f:
    expected = json.load(f)


actual = _get_learned_parameters(
    nodes=config["nodes"],
    edges=config["edges"],
    observations=config["observations"],
)


for node in expected:
    assert node in actual

    for param in expected[node]:
        assert param in actual[node]

        actual_value = actual[node][param]
        expected_value = expected[node][param]

        assert np.isclose(
            actual_value,
            expected_value,
            atol=1e-6,
            rtol=1e-6,
        ), (
            f"Case {CASE}, node {node}, param {param}: "
            f"got {actual_value}, expected {expected_value}"
        )


print(f"CASE {CASE}: ALL TESTS PASSED")