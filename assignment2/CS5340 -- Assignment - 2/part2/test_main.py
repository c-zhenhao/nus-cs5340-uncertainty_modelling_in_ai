import numpy as np

from main import (
    _learn_node_parameter_w,
    _learn_node_parameter_var,
    _get_learned_parameters
)


def test_learn_node_parameter_w_no_parents():
    print("\n=== Test: no parents ===")

    outputs = np.array([
        1.0,
        2.0,
        3.0,
        4.0,
    ])

    weights = _learn_node_parameter_w(
        outputs=outputs,
        inputs=None,
    )

    print("weights:", weights)

    # With no parents:
    # w0 = mean(outputs) = 2.5
    expected = np.array([2.5])

    assert weights.shape == (1,)
    assert np.allclose(
        weights,
        expected,
        atol=1e-8,
    )

    print("PASSED")


def test_learn_node_parameter_w_one_parent():
    print("\n=== Test: one parent ===")

    # y = 2 + 3x
    inputs = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    ])

    outputs = np.array([
        2.0,
        5.0,
        8.0,
        11.0,
    ])

    weights = _learn_node_parameter_w(
        outputs=outputs,
        inputs=inputs,
    )

    print("weights:", weights)

    # [bias, x coefficient]
    expected = np.array([
        2.0,
        3.0,
    ])

    assert weights.shape == (2,)
    assert np.allclose(
        weights,
        expected,
        atol=1e-8,
    )

    print("PASSED")


def test_learn_node_parameter_w_two_parents():
    print("\n=== Test: two parents ===")

    # y = 1 + 2*x1 - 3*x2
    inputs = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
    ])

    outputs = np.array([
        1.0,   # 1 + 2(0) - 3(0)
        3.0,   # 1 + 2(1) - 3(0)
        -2.0,  # 1 + 2(0) - 3(1)
        0.0,   # 1 + 2(1) - 3(1)
        2.0,   # 1 + 2(2) - 3(1)
    ])

    weights = _learn_node_parameter_w(
        outputs=outputs,
        inputs=inputs,
    )

    print("weights:", weights)

    # [w0, w1, w2]
    expected = np.array([
        1.0,
        2.0,
        -3.0,
    ])

    assert weights.shape == (3,)
    assert np.allclose(
        weights,
        expected,
        atol=1e-8,
    )

    print("PASSED")


def test_learn_node_parameter_var_no_parents():
    print("\n=== Test variance: no parents ===")

    outputs = np.array([
        1.0,
        2.0,
        3.0,
        4.0,
    ])

    weights = np.array([2.5])

    var = _learn_node_parameter_var(
        outputs=outputs,
        weights=weights,
        inputs=None,
    )

    print("variance:", var)

    # residuals:
    # [-1.5, -0.5, 0.5, 1.5]
    #
    # squared residuals:
    # [2.25, 0.25, 0.25, 2.25]
    #
    # mean = 5 / 4 = 1.25
    expected = 1.25

    assert np.isclose(
        var,
        expected,
        atol=1e-8,
    )

    print("PASSED")


def test_learn_node_parameter_var_exact_fit():
    print("\n=== Test variance: exact fit ===")

    # y = 2 + 3x
    inputs = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    ])

    outputs = np.array([
        2.0,
        5.0,
        8.0,
        11.0,
    ])

    weights = np.array([
        2.0,
        3.0,
    ])

    var = _learn_node_parameter_var(
        outputs=outputs,
        weights=weights,
        inputs=inputs,
    )

    print("variance:", var)

    # Predictions equal outputs exactly,
    # so every residual is zero.
    expected = 0.0

    assert np.isclose(
        var,
        expected,
        atol=1e-8,
    )

    print("PASSED")


def test_learn_node_parameter_var_with_noise():
    print("\n=== Test variance: noisy observations ===")

    inputs = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    ])

    # Using model:
    # y_hat = 2 + 3x
    #
    # predictions:
    # [2, 5, 8, 11]
    #
    # actual outputs differ by:
    # [+1, -1, +1, -1]
    outputs = np.array([
        3.0,
        4.0,
        9.0,
        10.0,
    ])

    weights = np.array([
        2.0,
        3.0,
    ])

    var = _learn_node_parameter_var(
        outputs=outputs,
        weights=weights,
        inputs=inputs,
    )

    print("variance:", var)

    # residuals = [1, -1, 1, -1]
    # squared residuals = [1, 1, 1, 1]
    # MLE variance = mean = 1
    expected = 1.0

    assert np.isclose(
        var,
        expected,
        atol=1e-8,
    )

    print("PASSED")


def test_get_learned_parameters():
    print("\n=== Test: full parameter learning ===")

    # Graph:
    #
    # 1 ----\
    #        > 3
    # 2 ----/
    #
    nodes = [1, 2, 3]

    edges = [
        [1, 3],
        [2, 3],
    ]

    x1 = np.array([
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
    ])

    x2 = np.array([
        1.0,
        0.0,
        2.0,
        1.0,
        3.0,
    ])

    # x3 = 5 + 2*x1 - 4*x2
    x3 = 5 + 2 * x1 - 4 * x2

    observations = {
        "1": x1.tolist(),
        "2": x2.tolist(),
        "3": x3.tolist(),
    }

    parameters = _get_learned_parameters(
        nodes=nodes,
        edges=edges,
        observations=observations,
    )

    print(parameters)

    # Node 1: root node
    assert np.isclose(
        parameters["1"]["bias"],
        2.0,
        atol=1e-8,
    )

    assert np.isclose(
        parameters["1"]["variance"],
        2.0,
        atol=1e-8,
    )

    # Node 2: root node
    assert np.isclose(
        parameters["2"]["bias"],
        1.4,
        atol=1e-8,
    )

    assert np.isclose(
        parameters["2"]["variance"],
        1.04,
        atol=1e-8,
    )

    # Node 3:
    # x3 = 5 + 2*x1 - 4*x2
    assert np.isclose(
        parameters["3"]["bias"],
        5.0,
        atol=1e-8,
    )

    assert np.isclose(
        parameters["3"]["1"],
        2.0,
        atol=1e-8,
    )

    assert np.isclose(
        parameters["3"]["2"],
        -4.0,
        atol=1e-8,
    )

    assert np.isclose(
        parameters["3"]["variance"],
        0.0,
        atol=1e-8,
    )

    print("PASSED")


if __name__ == "__main__":
    test_learn_node_parameter_w_no_parents()
    test_learn_node_parameter_w_one_parent()
    test_learn_node_parameter_w_two_parents()

    test_learn_node_parameter_var_no_parents()
    test_learn_node_parameter_var_exact_fit()
    test_learn_node_parameter_var_with_noise()

    test_get_learned_parameters()

    print("\n============================")
    print("ALL PART 2 TESTS PASSED")
    print("============================")