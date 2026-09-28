""" CS5340 Lab 2 Part 1: Junction Tree Algorithm
See accompanying PDF for instructions.

Name: Chen Zhenhao
Email: e1590318@u.nus.edu
Student ID: A0332840L
"""
import copy
import os
import numpy as np
import json
import networkx as nx
from argparse import ArgumentParser

from factor import Factor
from jt_construction import construct_junction_tree
from factor_utils import factor_product, factor_evidence, factor_marginalize

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'inputs')  # we will store the input data files here!
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')  # we will store the prediction files here!


""" ADD HELPER FUNCTIONS HERE """
def _send_message(
    src,
    dst,
    jt_cliques,
    neighbors,
    jt_clique_factors,
    messages,
):
    # Already computed?
    if (src, dst) in messages:
        return messages[(src, dst)]

    # Start with src's own clique factor
    combined = copy.deepcopy(
        jt_clique_factors[src]
    )

    # Collect incoming messages from every neighbor
    # except the destination
    for neighbor in neighbors[src]:

        if neighbor == dst:
            continue

        incoming = _send_message(
            neighbor,
            src,
            jt_cliques,
            neighbors,
            jt_clique_factors,
            messages,
        )
        if incoming.is_empty():
            continue

        if combined.is_empty():
            combined = copy.deepcopy(incoming)
        else:
            combined = factor_product(combined, incoming)

    # Separator S_src,dst
    separator = np.intersect1d(
        jt_cliques[src],
        jt_cliques[dst]
    )

    # Variables that must be summed out
    vars_to_eliminate = np.setdiff1d(
        combined.var,
        separator
    )

    if len(vars_to_eliminate) == 0:
        message = copy.deepcopy(combined)
    else:
        message = factor_marginalize(combined, vars_to_eliminate)

    messages[(src, dst)] = message

    return message

""" END HELPER FUNCTIONS HERE """


def _update_mrf_w_evidence(all_nodes, evidence, edges, factors):
    """
    Update the MRF graph structure from observing the evidence

    Args:
        all_nodes: numpy array of nodes in the MRF
        evidence: dictionary of node:observation pairs where evidence[x1] returns the observed value of x1
        edges: numpy array of edges in the MRF
        factors: list of Factors in teh MRF

    Returns:
        numpy array of query nodes
        numpy array of updated edges (after observing evidence)
        list of Factors (after observing evidence; empty factors should be removed)
    """

    query_nodes = all_nodes
    updated_edges = edges
    updated_factors = factors

    """ YOUR CODE HERE """
    evidence_nodes = set(evidence.keys())

    # 1. keep only non-evidence nodes
    query_nodes = np.array([
        node
        for node in all_nodes
        if node not in evidence_nodes
    ])

    # 2. keep only edges whose endpoints are both unobserved
    remaining_edges = []

    for edge in edges:
        u, v = edge

        if u not in evidence_nodes and v not in evidence_nodes:
            remaining_edges.append([u, v])

    if len(remaining_edges) == 0:
        updated_edges = np.empty((0, 2), dtype=np.int64)
    else:
        updated_edges = np.array(
            remaining_edges,
            dtype=np.int64
        )

    # 3. condition every factor on evidence
    updated_factors = []

    for factor in factors:

        reduced_factor = factor_evidence(factor, evidence)

        # 4. fully observed factors become empty
        if not reduced_factor.is_empty():
            updated_factors.append(reduced_factor)

    """ END YOUR CODE HERE """

    return query_nodes, updated_edges, updated_factors


def _get_clique_potentials(jt_cliques, jt_edges, jt_clique_factors):
    """
    Returns the list of clique potentials after performing the sum-product algorithm on the junction tree

    Args:
        jt_cliques: list of junction tree nodes e.g. [[x1, x2], ...]
        jt_edges: numpy array of junction tree edges e.g. [i,j] implies that jt_cliques[i] and jt_cliques[j] are
                neighbors
        jt_clique_factors: list of clique factors where jt_clique_factors[i] is the factor for cliques[i]

    Returns:
        list of clique potentials computed from the sum-product algorithm
    """
    clique_potentials = jt_clique_factors

    """ YOUR CODE HERE """
    # build neighbor lists
    neighbors = {
        i: set()
        for i in range(len(jt_cliques))
    }

    for i, j in jt_edges:
        neighbors[int(i)].add(int(j))

    messages = {}

    # compute every directed message
    for src, dst in jt_edges:
        _send_message(
            int(src),
            int(dst),
            jt_cliques,
            neighbors,
            jt_clique_factors,
            messages,
        )

    clique_potentials = []

    # each final clique potential = local factor * all incoming messages
    for i in range(len(jt_cliques)):

        potential = copy.deepcopy(
            jt_clique_factors[i]
        )

        for neighbor in neighbors[i]:

            incoming = messages[
                (neighbor, i)
            ]
            if incoming.is_empty():
                continue

            if potential.is_empty():
                potential = copy.deepcopy(incoming)
            else:
                potential = factor_product(potential, incoming)

        clique_potentials.append(potential)
    """ END YOUR CODE HERE """

    assert len(clique_potentials) == len(jt_cliques)

    return clique_potentials


def _get_node_marginal_probabilities(query_nodes, cliques, clique_potentials):
    """
    Returns the marginal probability for each query node from the clique potentials.

    Args:
        query_nodes: numpy array of query nodes e.g. [x1, x2, ..., xN]
        cliques: list of cliques e.g. [[x1, x2], ... [x2, x3, .., xN]]
        clique_potentials: list of clique potentials (Factor class)

    Returns:
        list of node marginal probabilities (Factor class)

    """
    query_marginal_probabilities = []

    """ YOUR CODE HERE """
    for node in query_nodes:

        # find every clique containing this node
        candidate_indices = [
            i
            for i, clique in enumerate(cliques)
            if node in clique
        ]

        assert len(candidate_indices) > 0, (
            f"No clique contains node {node}"
        )

        # efficient choice: use the smallest clique
        clique_idx = min(
            candidate_indices,
            key=lambda i: len(cliques[i])
        )

        potential = copy.deepcopy(
            clique_potentials[clique_idx]
        )

        # sum out everything except node
        vars_to_eliminate = np.array([
            v
            for v in potential.var
            if v != node
        ])

        if len(vars_to_eliminate) > 0:
            marginal = factor_marginalize(potential, vars_to_eliminate)
        else:
            marginal = potential

        # Normalize
        total = np.sum(marginal.val)

        assert total > 0, (
            f"Cannot normalize marginal for node {node}: "
            f"sum is {total}"
        )

        marginal.val = marginal.val / total

        query_marginal_probabilities.append(
            marginal
        )

    """ END YOUR CODE HERE """

    return query_marginal_probabilities


def get_conditional_probabilities(all_nodes, evidence, edges, factors):
    """
    Returns query nodes and query Factors representing the conditional probability of each query node
    given the evidence e.g. p(xf|Xe) where xf is a single query node and Xe is the set of evidence nodes.

    Args:
        all_nodes: numpy array of all nodes (random variables) in the graph
        evidence: dictionary of node:evidence pairs e.g. evidence[x1] returns the observed value for x1
        edges: numpy array of all edges in the graph e.g. [[x1, x2],...] implies that x1 is a neighbor of x2
        factors: list of factors in the MRF.

    Returns:
        numpy array of query nodes
        list of Factor
    """
    query_nodes, updated_edges, updated_node_factors = _update_mrf_w_evidence(all_nodes=all_nodes, evidence=evidence,
                                                                              edges=edges, factors=factors)

    jt_cliques, jt_edges, jt_factors = construct_junction_tree(nodes=query_nodes, edges=updated_edges,
                                                               factors=updated_node_factors)

    clique_potentials = _get_clique_potentials(jt_cliques=jt_cliques, jt_edges=jt_edges, jt_clique_factors=jt_factors)

    query_node_marginals = _get_node_marginal_probabilities(query_nodes=query_nodes, cliques=jt_cliques,
                                                            clique_potentials=clique_potentials)

    return query_nodes, query_node_marginals


def parse_input_file(input_file: str):
    """ Reads the input file and parses it. DO NOT EDIT THIS FUNCTION. """
    with open(input_file, 'r') as f:
        input_config = json.load(f)

    nodes = np.array(input_config['nodes'])
    edges = np.array(input_config['edges'])

    # parse evidence
    raw_evidence = input_config['evidence']
    evidence = {}
    for k, v in raw_evidence.items():
        evidence[int(k)] = v

    # parse factors
    raw_factors = input_config['factors']
    factors = []
    for raw_factor in raw_factors:
        factor = Factor(var=np.array(raw_factor['var']), card=np.array(raw_factor['card']),
                        val=np.array(raw_factor['val']))
        factors.append(factor)
    return nodes, edges, evidence, factors


def main():
    """ Entry function to handle loading inputs and saving outputs. DO NOT EDIT THIS FUNCTION. """
    argparser = ArgumentParser()
    argparser.add_argument('--case', type=int, required=True,
                           help='case number to create observations e.g. 1 if 1.json')
    args = argparser.parse_args()

    case = args.case
    input_file = os.path.join(INPUT_DIR, '{}.json'.format(case))
    nodes, edges, evidence, factors = parse_input_file(input_file=input_file)

    # solution part:
    query_nodes, query_conditional_probabilities = get_conditional_probabilities(all_nodes=nodes, edges=edges,
                                                                                 factors=factors, evidence=evidence)

    predictions = {}
    for i, node in enumerate(query_nodes):
        probability = query_conditional_probabilities[i].val
        predictions[int(node)] = list(np.array(probability, dtype=float))

    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)
    prediction_file = os.path.join(PREDICTION_DIR, '{}.json'.format(case))
    with open(prediction_file, 'w') as f:
        json.dump(predictions, f, indent=1)
    print('INFO: Results for test case {} are stored in {}'.format(case, prediction_file))


if __name__ == '__main__':
    main()
