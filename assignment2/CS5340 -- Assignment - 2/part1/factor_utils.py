# taken from part 1
import copy
import numpy as np
from factor import Factor, index_to_assignment, assignment_to_index


def factor_product(A, B):
    """
    Computes the factor product of A and B e.g. A = f(x1, x2); B = f(x1, x3); out=f(x1, x2, x3) = f(x1, x2)f(x1, x3)

    Args:
        A: first Factor
        B: second Factor

    Returns:
        Returns the factor product of A and B
    """
    out = Factor()

    """ YOUR CODE HERE """
    if A.is_empty():
        return copy.deepcopy(B)
    if B.is_empty():
        return copy.deepcopy(A)

    out.var = np.union1d(A.var, B.var)

    map_A = np.array([
        np.where(out.var == v)[0][0]
        for v in A.var
    ])

    map_B = np.array([
        np.where(out.var == v)[0][0]
        for v in B.var
    ])

    out.card = np.zeros(len(out.var), dtype=np.int64)

    out.card[map_A] = A.card
    out.card[map_B] = B.card

    assignments = index_to_assignment(
        np.arange(int(np.prod(out.card))),
        out.card
    )

    assignment_A = assignments[:, map_A]
    assignment_B = assignments[:, map_B]

    idxA = assignment_to_index(
        assignment_A,
        A.card
    )

    idxB = assignment_to_index(
        assignment_B,
        B.card
    )

    out.val = A.val[idxA] * B.val[idxB]

    """ END YOUR CODE HERE """
    return out


def factor_marginalize(factor, var):
    """
    Returns factor after variables in var have been marginalized out.

    Args:
        factor: factor to be marginalized
        var: numpy array of variables to be marginalized over

    Returns:
        marginalized factor
    """
    out = copy.deepcopy(factor)

    """ YOUR CODE HERE
     HINT: Use the code from lab1 """
    out = Factor()

    remaining_vars = []
    remaining_cards = []
    remaining_indices = []

    for i in range(len(factor.var)):
        variable = factor.var[i]

        if variable not in var:
            remaining_vars.append(variable)
            remaining_cards.append(factor.card[i])
            remaining_indices.append(i)

    out.var = np.array(remaining_vars, dtype=np.int64)
    out.card = np.array(remaining_cards, dtype=np.int64)
    remaining_indices = np.array(remaining_indices, dtype=np.int64)

    num_assignments = int(np.prod(out.card))
    out.val = np.zeros(num_assignments)

    input_assignments = factor.get_all_assignments()

    for i in range(len(factor.val)):
        assignment = input_assignments[i]

        output_assignment = assignment[remaining_indices]

        output_index = assignment_to_index(
            output_assignment,
            out.card
        )

        out.val[output_index] += factor.val[i]

    """ END YOUR CODE HERE """
    return out


def factor_evidence(factor, evidence):
    """
    Observes evidence and retains entries containing the observed evidence. Also removes the evidence random variables
    because they are already observed e.g. factor=f(1, 2) and evidence={1: 0} returns f(2) with entries from node1=0
    Args:
        factor: factor to reduce using evidence
        evidence:  dictionary of node:evidence pair where evidence[1] = evidence of node 1.
    Returns:
        Reduced factor that does not contain any variables in the evidence. Return an empty factor if all the
        factor's variables are observed.
    """
    out = copy.deepcopy(factor)

    """ YOUR CODE HERE,     HINT: copy from lab2 part 1! """
    observed_vars = [
        v for v in factor.var
        if v in evidence
    ]

    if len(observed_vars) == 0:
        return out

    if len(observed_vars) == len(factor.var):
        return Factor()

    assignments = factor.get_all_assignments()

    # build a boolean mask:
    mask = np.ones(len(factor.val), dtype=bool)

    # True only if assignment agrees with every observed value
    for v in observed_vars:
        pos = np.where(factor.var == v)[0][0]
        mask &= assignments[:, pos] == evidence[v]

    # select surviving assignments / values
    # kept_assignments = assignments[mask]
    kept_values = factor.val[mask]

    # determine positions of non-evidence variables
    remaining_indices = [
        i for i, v in enumerate(factor.var)
        if v not in evidence
    ]

    out.var = factor.var[remaining_indices]
    out.card = factor.card[remaining_indices]

    # surviving rows keep their relative order, which matches the reduced factor's index order
    out.val = kept_values

    """ END YOUR CODE HERE """

    return out



if __name__ == '__main__':
    main()
