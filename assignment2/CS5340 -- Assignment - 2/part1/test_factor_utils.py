import numpy as np

from factor import Factor
from factor_utils import (
    factor_product,
    factor_marginalize,
    factor_evidence,
)


def test_factor_product():
    print("\n=== Testing factor_product ===")

    A = Factor(
        var=np.array([0]),
        card=np.array([2]),
        val=np.array([0.2, 0.8])
    )

    B = Factor(
        var=np.array([0, 1]),
        card=np.array([2, 2]),
        val=np.array([0.5, 0.1, 0.5, 0.9])
    )

    C = factor_product(A, B)

    print(C)
    print("var :", C.var)
    print("card:", C.card)
    print("val :", C.val)

    expected_var = np.array([0, 1])
    expected_card = np.array([2, 2])
    expected_val = np.array([0.1, 0.08, 0.1, 0.72])

    assert np.array_equal(C.var, expected_var)
    assert np.array_equal(C.card, expected_card)
    assert np.allclose(C.val, expected_val)

    print("factor_product PASSED")


def test_factor_marginalize():
    print("\n=== Testing factor_marginalize ===")

    f = Factor(
        var=np.array([0, 1]),
        card=np.array([2, 2]),
        val=np.array([0.1, 0.08, 0.1, 0.72])
    )

    m = factor_marginalize(f, np.array([1]))

    print(m)
    print("var :", m.var)
    print("card:", m.card)
    print("val :", m.val)

    expected_var = np.array([0])
    expected_card = np.array([2])
    expected_val = np.array([0.2, 0.8])

    assert np.array_equal(m.var, expected_var)
    assert np.array_equal(m.card, expected_card)
    assert np.allclose(m.val, expected_val)

    print("factor_marginalize PASSED")


def test_factor_evidence():
    print("\n=== Testing factor_evidence ===")

    f = Factor(
        var=np.array([0, 2]),
        card=np.array([2, 2]),
        val=np.array([1, 2, 3, 4])
    )

    # ----------------------------------------
    # Test 1: X2 = 1
    # ----------------------------------------

    out = factor_evidence(f, {2: 1})

    print("\nEvidence {2: 1}")
    print(out)

    assert np.array_equal(out.var, np.array([0]))
    assert np.array_equal(out.card, np.array([2]))
    assert np.array_equal(out.val, np.array([3, 4]))

    print("Test 1 PASSED")

    # ----------------------------------------
    # Test 2: X0 = 1
    # ----------------------------------------

    out = factor_evidence(f, {0: 1})

    print("\nEvidence {0: 1}")
    print(out)

    assert np.array_equal(out.var, np.array([2]))
    assert np.array_equal(out.card, np.array([2]))
    assert np.array_equal(out.val, np.array([2, 4]))

    print("Test 2 PASSED")

    # ----------------------------------------
    # Test 3: irrelevant evidence
    # ----------------------------------------

    out = factor_evidence(f, {5: 1})

    print("\nIrrelevant evidence {5: 1}")
    print(out)

    assert np.array_equal(out.var, f.var)
    assert np.array_equal(out.card, f.card)
    assert np.array_equal(out.val, f.val)

    print("Test 3 PASSED")

    # ----------------------------------------
    # Test 4: no evidence
    # ----------------------------------------

    out = factor_evidence(f, {})

    print("\nNo evidence")
    print(out)

    assert np.array_equal(out.var, f.var)
    assert np.array_equal(out.card, f.card)
    assert np.array_equal(out.val, f.val)

    print("Test 4 PASSED")

    # ----------------------------------------
    # Test 5: all variables observed
    # ----------------------------------------

    out = factor_evidence(f, {0: 1, 2: 0})

    print("\nAll variables observed")
    print(out)

    assert out.is_empty()

    print("Test 5 PASSED")

    print("factor_evidence PASSED")


if __name__ == "__main__":
    test_factor_product()
    test_factor_marginalize()
    test_factor_evidence()

    print("\n========================")
    print("ALL FACTOR UTILS TESTS PASSED")
    print("========================")