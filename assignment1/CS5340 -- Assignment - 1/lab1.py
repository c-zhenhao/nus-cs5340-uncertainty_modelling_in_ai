""" CS5340 Lab 1: Belief Propagation and Maximal Probability
See accompanying PDF for instructions.

Name: Chen Zhenhao
Email: zhenhaoc@u.nus.edu / e1590318@u.nus.edu
Student ID: A0338240L
"""

import copy
from typing import List

import numpy as np

from factor import Factor, index_to_assignment, assignment_to_index, generate_graph_from_factors, \
    visualize_graph


"""For sum product message passing"""
def factor_product(A, B):
    """Compute product of two factors.

    Suppose A = phi(X_1, X_2), B = phi(X_2, X_3), the function should return
    phi(X_1, X_2, X_3)
    """
    if A.is_empty():
        return B
    if B.is_empty():
        return A

    # Create output factor. Variables should be the union between of the
    # variables contained in the two input factors
    out = Factor()
    out.var = np.union1d(A.var, B.var)

    # Compute mapping between the variable ordering between the two factors
    # and the output to set the cardinality
    out.card = np.zeros(len(out.var), np.int64)
    mapA = np.argmax(out.var[None, :] == A.var[:, None], axis=-1)
    mapB = np.argmax(out.var[None, :] == B.var[:, None], axis=-1)
    out.card[mapA] = A.card
    out.card[mapB] = B.card

    # For each assignment in the output, compute which row of the input factors
    # it comes from
    out.val = np.zeros(np.prod(out.card))
    assignments = out.get_all_assignments()
    idxA = assignment_to_index(assignments[:, mapA], A.card)
    idxB = assignment_to_index(assignments[:, mapB], B.card)

    """ YOUR CODE HERE
    You should populate the .val field with the factor product
    Hint: The code for this function should be very short (~1 line). Try to
      understand what the above lines are doing, in order to implement
      subsequent parts.
    """
    # A.val = [0.8, 0.2]
    # B.val = [0.4, 0.55, 0.6, 0.45]
    # A.val[idxA] = [0.8, 0.2, 0.8, 0.2]
    # B.val[idxB] = [0.4, 0.55, 0.6, 0.45]
    # = [0.32, 0.11, 0.48, 0.09]
    out.val = A.val[idxA] * B.val[idxB]

    return out


def factor_marginalize(factor, var):
    """Sums over a list of variables.

    Args:
        factor (Factor): Input factor
        var (List): Variables to marginalize out

    Returns:
        out: Factor with variables in 'var' marginalized out.
    """
    out = Factor()

    """ YOUR CODE HERE
    Marginalize out the variables given in var
    """
    # 1. create output factor over remaining variables
    remaining_vars = []
    remaining_cards = []
    remaining_indices = []

    for i in range(len(factor.var)):
        variable = factor.var[i]

        if variable not in var:
            remaining_vars.append(variable)
            remaining_cards.append(factor.card[i])
            remaining_indices.append(i)

    out.var = np.array(remaining_vars)
    out.card = np.array(remaining_cards)

    # 2. enumerate every input assignment
    num_assignments = int(np.prod(out.card))
    out.val = np.zeros(num_assignments)

    # 3. project each input assignment onto the remaining variables.
    input_assignments = factor.get_all_assignments()

    # 4. find which output row it maps to
    for i in range(len(factor.val)):
        assignment = input_assignments[i]

        output_assignment = assignment[remaining_indices]

        output_index = assignment_to_index(
            output_assignment,
            out.card
        )

        # and accumulate its value
        out.val[output_index] += factor.val[i]

    return out


def observe_evidence(factors, evidence=None):
    """Modify a set of factors given some evidence

    Args:
        factors (List[Factor]): List of input factors
        evidence (Dict): Dictionary, where the keys are the observed variables
          and the values are the observed values.

    Returns:
        List of factors after observing evidence
    """
    out = copy.deepcopy(factors)

    # Always return a copy. Inference routines may transform factor values
    # (for example, MAP inference converts them to log-space), and must not
    # mutate factors supplied by the caller.
    if evidence is None:
        return out

    """ YOUR CODE HERE
    Set the probabilities of assignments which are inconsistent with the 
    evidence to zero.
    """
    for factor in out:
        assignments = factor.get_all_assignments()

        for evidence_var, evidence_value in evidence.items():
            if evidence_var in factor.var:
                column = np.where(factor.var == evidence_var)[0][0]
                inconsistent = assignments[:, column] != evidence_value
                factor.val[inconsistent] = 0

    return out


"""For max sum message passing (for MAP)"""
def factor_sum(A, B):
    """Same as factor_product, but sums instead of multiplies
    """
    if A.is_empty():
        return B
    if B.is_empty():
        return A

    # Create output factor. Variables should be the union between of the
    # variables contained in the two input factors
    out = Factor()
    out.var = np.union1d(A.var, B.var)

    # Compute mapping between the variable ordering between the two factors
    # and the output to set the cardinality
    out.card = np.zeros(len(out.var), np.int64)
    mapA = np.argmax(out.var[None, :] == A.var[:, None], axis=-1)
    mapB = np.argmax(out.var[None, :] == B.var[:, None], axis=-1)
    out.card[mapA] = A.card
    out.card[mapB] = B.card

    # For each assignment in the output, compute which row of the input factors
    # it comes from
    out.val = np.zeros(np.prod(out.card))
    assignments = out.get_all_assignments()
    idxA = assignment_to_index(assignments[:, mapA], A.card)
    idxB = assignment_to_index(assignments[:, mapB], B.card)

    """ YOUR CODE HERE
    You should populate the .val field with the factor sum. The code for this
    should be very similar to the factor_product().
    """
    # [0.8, 0.2, 0.8, 0.2] + [0.4, 0.55, 0.6, 0.45]
    # =[1,2, 0.75, 1.4, 0.65]
    out.val = A.val[idxA] + B.val[idxB]

    return out


def compute_joint_distribution(factors):
    """Computes the joint distribution defined by a list of given factors

    Args:
        factors (List[Factor]): List of factors

    Returns:
        Factor containing the joint distribution of the input factor list
    """
    joint = Factor()

    """ YOUR CODE HERE
    Compute the joint distribution from the list of factors. You may assume
    that the input factors are valid so no input checking is required.
    """
    for factor in factors:
        joint = factor_product(joint, factor)

    return joint


def compute_marginals_naive(V, factors, evidence):
    """Computes the marginal over a set of given variables

    Args:
        V (int): Single Variable to perform inference on
        factors (List[Factor]): List of factors representing the graphical model
        evidence (Dict): Observed evidence. evidence[k] = v indicates that
          variable k has the value v. Evidence must have positive probability;
          otherwise the conditional marginal is undefined and ValueError
          should be raised.

    Returns:
        Factor representing the marginals
    """

    output = Factor()

    """ YOUR CODE HERE
    Compute the marginal. Output should be a factor.
    Remember to normalize the probabilities!
    """
    # 1. compute full join distribution
    joint = compute_joint_distribution(factors)

    # 2. apply evidence
    conditioned = observe_evidence([joint], evidence)[0]

    # 3. marginalize out irrelevant variables
    vars_to_marginalize = []
    for variable in conditioned.var:
        if variable != V:
            vars_to_marginalize.append(variable)

    # 4. normalize final factor
    output = factor_marginalize(conditioned, vars_to_marginalize)

    total = np.sum(output.val)

    if np.isclose(total, 0.0):
        raise ValueError("Evidence has zero probability")

    output.val = output.val / total

    return output


def send_message(graph, messages, i, j):
    message_factor = Factor()

    if 'factor' in graph.nodes[i]:
        message_factor = factor_product(
            message_factor,
            graph.nodes[i]['factor']
        )

    for neighbor in graph.neighbors(i):
        if neighbor != j:
            message_factor = factor_product(
                message_factor,
                messages[neighbor][i]
            )

    message_factor = factor_product(
        message_factor,
        graph.edges[i, j]['factor']
    )

    messages[i][j] = factor_marginalize(message_factor, [i])


def collect_messages(graph, messages, node, parent):
    for neighbor in graph.neighbors(node):
        if neighbor == parent:
            continue

        collect_messages(graph, messages, neighbor, node)

    if parent is not None:
        send_message(graph, messages, node, parent)


def distribute_messages(graph, messages, node, parent):
    for neighbor in graph.neighbors(node):
        if neighbor == parent:
            continue

        send_message(graph, messages, node, neighbor)
        distribute_messages(graph, messages, neighbor, node)


def compute_marginals_bp(V, factors, evidence):
    """Compute single node marginals for multiple variables
    using sum-product belief propagation algorithm

    Args:
        V (List): Variables to infer single node marginals for
        factors (List[Factor]): List of factors representing the grpahical model
        evidence (Dict): Observed evidence. evidence[k]=v denotes that the
          variable k is assigned to value v. Evidence must have positive
          probability; otherwise raise ValueError.

    Returns:
        marginals: List of factors. The ordering of the factors should follow
          that of V, i.e. marginals[i] should be the factor for variable V[i].
    """
    # Dummy outputs, you should overwrite this with the correct factors
    marginals = []

    # Setting up messages which will be passed
    factors = observe_evidence(factors, evidence)
    graph = generate_graph_from_factors(factors)

    # Uncomment the following line to visualize the graph. Note that we create
    # an undirected graph regardless of the input graph since 1) this
    # facilitates graph traversal, and 2) the algorithm for undirected and
    # directed graphs is essentially the same for tree-like graphs.
    # visualize_graph(graph)

    # You can use any node as the root since the graph is a tree. For simplicity
    # we always use node 0 for this assignment.
    root = 0

    # Create structure to hold messages
    num_nodes = graph.number_of_nodes()
    messages = [[None] * num_nodes for _ in range(num_nodes)]

    """ YOUR CODE HERE
    Use the algorithm from lecture 4 and perform message passing over the entire
    graph. Recall the message passing protocol, that a node can only send a
    message to a neighboring node only when it has received messages from all
    its other neighbors.
    
    Since the provided graphical model is a tree, we can use a two-phase 
    approach. First we send messages inward from leaves towards the root.
    After this is done, we can send messages from the root node outward.
    
    Hint: You might find it useful to add auxilliary functions. You may add 
      them as either inner (nested) or external functions.
    """
    collect_messages(graph, messages, root, None)
    distribute_messages(graph, messages, root, None)

    # compute marginals
    for variable in V:
        marginal = Factor()

        # include unary factor if this node has one
        if 'factor' in graph.nodes[variable]:
            marginal = factor_product(
                marginal,
                graph.nodes[variable]['factor']
            )

        # multiply all incoming messages
        for neighbor in graph.neighbors(variable):
            marginal = factor_product(
                marginal,
                messages[neighbor][variable]
            )

        # normalize
        total = np.sum(marginal.val)

        if np.isclose(total, 0.0):
            raise ValueError("Evidence has zero probability")

        marginal.val = marginal.val / total

        marginals.append(marginal)

    return marginals


def factor_max_marginalize(factor, var):
    """Marginalize over a list of variables by taking the max.

    Args:
        factor (Factor): Input factor
        var (List): Variable to marginalize out.

    Returns:
        out: Factor with variables in 'var' marginalized out. The factor's
          .val_argmax field should be a list of dictionary that keep track
          of the maximizing values of the marginalized variables.
          e.g. when out.val_argmax[i][j] = k, this means that
            when assignments of out is index_to_assignment[i],
            variable j has a maximizing value of k.
          See test_lab1.py::test_factor_max_marginalize() for an example.
    """
    out = Factor()

    """ YOUR CODE HERE
    Marginalize out the variables given in var.

    You should make use of val_argmax to keep track of the location with the
    maximum probability.
    """
    # separate variables into those we keep and those we maximize out
    remaining_vars = []
    remaining_cards = []
    remaining_indices = []
    marginalized_indices = []

    for i in range(len(factor.var)):
        variable = factor.var[i]

        if variable not in var:
            remaining_vars.append(variable)
            remaining_cards.append(factor.card[i])
            remaining_indices.append(i)
        else:
            marginalized_indices.append(i)

    # initialize output factor and enum all input assignments
    out.var = np.array(remaining_vars)
    out.card = np.array(remaining_cards)

    num_assignments = int(np.prod(out.card))

    out.val = np.full(num_assignments, -np.inf)
    out.val_argmax = [None] * num_assignments

    input_assignments = factor.get_all_assignments()

    # map each input assignment to output assignment
    for i in range(len(factor.val)):
        assignment = input_assignments[i]

        output_assignment = assignment[remaining_indices]
        output_index = assignment_to_index(
            output_assignment,
            out.card
        )

        # keep only maximum value for each output assignment
        if factor.val[i] > out.val[output_index]:
            out.val[output_index] = factor.val[i]

            argmax_assignment = {}

            for index in marginalized_indices:
                variable = factor.var[index]
                value = assignment[index]
                argmax_assignment[variable] = value

            # record marginalised values that achieved it
            out.val_argmax[output_index] = argmax_assignment
    
    return out


def send_max_message(graph, messages, i, j):
    message_factor = Factor()

    if 'factor' in graph.nodes[i]:
        message_factor = factor_sum(
            message_factor,
            graph.nodes[i]['factor']
        )

    for neighbor in graph.neighbors(i):
        # in max message skip receipient
        # message to j must only use information coming into i from other neighbours
        if neighbor == j:
            continue

        # In log-space, multiplying probabilities becomes addition
        # so we use factor_sum instead of factor_product
        message_factor = factor_sum(
            message_factor,
            messages[neighbor][i]
        )

    message_factor = factor_sum(
        message_factor,
        graph.edges[i, j]['factor']
    )

    messages[i][j] = factor_max_marginalize(
        message_factor,
        [i]
    )


def collect_max_messages(graph, messages, node, parent):
    for neighbor in graph.neighbors(node):
        if neighbor == parent:
            continue

        collect_max_messages(graph, messages, neighbor, node)

    if parent is not None:
        send_max_message(graph, messages, node, parent)


def map_eliminate(factors, evidence):
    """Obtains the maximum a posteriori configuration for a tree graph
    given optional evidence

    Args:
        factors (List[Factor]): List of factors representing the graphical model
        evidence (Dict): Observed evidence. evidence[k]=v denotes that the
          variable k is assigned to value v. Pass None or {} for no evidence.
          The input factors must not be modified. If evidence has zero
          probability, the MAP configuration is undefined and ValueError
          should be raised.

    Returns:
        max_decoding (Dict): MAP configuration
        log_prob_max: Log probability of MAP configuration. Note that this is
          log p(MAP, e) instead of p(MAP|e), i.e. it is the unnormalized
          representation of the conditional probability.
    """

    # Treat None and {} equivalently; this also makes membership checks on
    # evidence safe during decoding.
    if evidence is None:
        evidence = {}

    max_decoding = {}
    log_prob_max = 0.0

    """ YOUR CODE HERE
    Use the algorithm from lecture 5 and perform message passing over the entire
    graph to obtain the MAP configuration. Again, recall the message passing 
    protocol.

    Your code should be similar to compute_marginals_bp().
    
    To avoid underflow, first transform the factors in the probabilities
    to **log scale** and perform all operations on log scale instead.
    
    You may ignore the warning for taking log of zero, that is the desired
    behavior.
    """
    # 1. apply evidence and make a copy so the caller's factors are not modified
    factors = observe_evidence(factors, evidence)

    # convert probabilities to log-space
    for factor in factors:
        with np.errstate(divide='ignore'):
            factor.val = np.log(factor.val)

    # 2. build tree and initialise message storage
    graph = generate_graph_from_factors(factors)
    root = 0

    num_nodes = graph.number_of_nodes()
    messages = [[None] * num_nodes for _ in range(num_nodes)]

    # 3. send max-sum messages inward towards root
    collect_max_messages(graph, messages, root, None)

    # 4. compute the root score after the inward max-message pass
    root_factor = Factor()

    if 'factor' in graph.nodes[root]:
        root_factor = factor_sum(
            root_factor,
            graph.nodes[root]['factor']
        )

    # combine at root
    for neighbor in graph.neighbors(root):
        root_factor = factor_sum(
            root_factor,
            messages[neighbor][root]
        )

    # pick best root state
    # this should be cast to int because index_to_assignment
    # expect a python int for scalar input
    root_index = int(np.argmax(root_factor.val))
    log_prob_max = root_factor.val[root_index]

    if np.isneginf(log_prob_max):
        raise ValueError("Evidence has zero probability")

    root_assignment = index_to_assignment(
        root_index,
        root_factor.card
    )

    root_value = root_assignment[0]

    if root not in evidence:
        max_decoding[root] = root_value

    # 5. backtrack from chosen root value using message's val_argmax
    # to recover the maximizing value of every child variable
    def decode_children(node, parent, node_value):
        for child in graph.neighbors(node):
            if child == parent:
                continue

            message = messages[child][node]

            # message is a factor over node
            # this should also be int
            message_index = int(assignment_to_index(
                [node_value],
                message.card
            ))

            argmax = message.val_argmax[message_index]

            child_value = argmax[child]

            if child not in evidence:
                max_decoding[child] = child_value

            decode_children(child, node, child_value)

    decode_children(root, None, root_value)

    return max_decoding, log_prob_max

