""" CS5340 Lab 3: Hidden Markov Models
See accompanying PDF for instructions.

Name: Chen Zhenhao
Email: e1590318@u.nus.edu
Student ID: A0332840L
"""
import numpy as np
import scipy.stats
from scipy.special import softmax
from sklearn.cluster import KMeans


def initialize(n_states, x):
    """Initializes starting value for initial state distribution pi
    and state transition matrix A.

    A and pi are initialized with random starting values which satisfies the
    summation and non-negativity constraints.
    """
    n_states = int(n_states)
    seed = 5340
    np.random.seed(seed)

    pi = np.random.random(n_states)
    A = np.random.random([n_states, n_states])

    # We use softmax to satisify the summation constraints. Since the random
    # values are small and similar in magnitude, the resulting values are close
    # to a uniform distribution with small jitter
    pi = softmax(pi)
    A = softmax(A, axis=-1)

    # Gaussian Observation model parameters
    # We use k-means clustering to initalize the parameters.
    x_cat = np.concatenate(x, axis=0)
    kmeans = KMeans(n_clusters=n_states, random_state=seed).fit(x_cat[:, None])
    mu = kmeans.cluster_centers_[:, 0]
    std = np.array([np.std(x_cat[kmeans.labels_ == l]) for l in range(n_states)])
    phi = {'mu': mu, 'sigma': std}

    return pi, A, phi


"""E-step"""
def e_step(x_list, pi, A, phi):
    """ E-step: Compute posterior distribution of the latent variables,
    p(Z|X, theta_old). Specifically, we compute
      1) gamma(z_n): Marginal posterior distribution, and
      2) xi(z_n-1, z_n): Joint posterior distribution of two successive
         latent states

    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        pi (np.ndarray): Current estimated Initial state distribution (K,)
        A (np.ndarray): Current estimated Transition matrix (K, K)
        phi (Dict[np.ndarray]): Current estimated gaussian parameters

    Returns:
        gamma_list (List[np.ndarray]), xi_list (List[np.ndarray])
    """
    n_states = pi.shape[0]
    gamma_list = [np.zeros([len(x), n_states]) for x in x_list]
    xi_list = [np.zeros([len(x)-1, n_states, n_states]) for x in x_list]

    """ YOUR CODE HERE
    Use the forward-backward procedure on each input sequence to populate 
    "gamma_list" and "xi_list" with the correct values.
    Be sure to use the scaling factor for numerical stability.
    """
    mu = np.asarray(phi['mu'])
    sigma = np.asarray(phi['sigma'])

    for n, x in enumerate(x_list):
        x = np.asarray(x)
        T = len(x)

        # 1. emission probabilities B[t, k] = N(x_t; mu_k, sigma_k)
        #    x[:, None] broadcasts against mu[None, :] -> (T, K)
        B = scipy.stats.norm.pdf(x[:, None], loc=mu[None, :], scale=sigma[None, :])

        # 2. allocate
        alpha = np.zeros((T, n_states))
        beta = np.zeros((T, n_states))
        c = np.zeros(T)

        # 3. forward pass with scaling
        unnormalized = pi * B[0]
        c[0] = np.sum(unnormalized)
        alpha[0] = unnormalized / c[0]

        # forward pass, t = 1 ... T-1
        for t in range(1, T):
            predicted = np.dot(alpha[t-1], A)
            unnormalized = B[t] * predicted
            c[t] = np.sum(unnormalized)
            alpha[t] = unnormalized / c[t]

        # 4. backward pass using the same scaling factors
        beta[T-1] = 1.0
        for t in range(T-2, -1, -1):
            weighted = B[t+1] * beta[t+1]
            beta[t] = np.dot(A, weighted) / c[t+1]

        # 5. gamma
        gamma = alpha * beta
        gamma = gamma / np.sum(gamma, axis=1, keepdims=True)

        # 6. xi: alpha[t, j] * A[j, k] * B[t+1, k] * beta[t+1, k]
        xi = np.zeros((T-1, n_states, n_states))
        for t in range(T-1):
            xi_t = alpha[t][:, None] * A * (B[t+1] * beta[t+1])[None, :]
            xi[t] = xi_t / np.sum(xi_t)

        # 7. store
        gamma_list[n] = gamma
        xi_list[n] = xi


    return gamma_list, xi_list


"""M-step"""
def m_step(x_list, gamma_list, xi_list):
    """M-step of Baum-Welch: Maximises the log complete-data likelihood for
    Gaussian HMMs.
    
    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        gamma_list (List[np.ndarray]): Marginal posterior distribution
        xi_list (List[np.ndarray]): Joint posterior distribution of two
          successive latent states

    Returns:
        pi (np.ndarray): Initial state distribution
        A (np.ndarray): Transition matrix
        phi (Dict[np.ndarray]): Parameters for the Gaussian HMM model, contains
          two fields 'mu', 'sigma' for the mean and standard deviation
          respectively.
    """

    n_states = gamma_list[0].shape[1]
    pi = np.zeros([n_states])
    A = np.zeros([n_states, n_states])
    phi = {'mu': np.zeros(n_states),
           'sigma': np.zeros(n_states)}

    """ YOUR CODE HERE
    Compute the complete-data maximum likelihood estimates for pi, A, phi.
    """
    # 1. pi; first gamma of each sequence -> average across sequences
    pi[:] = np.mean([gamma[0] for gamma in gamma_list], axis=0)

    # 2. A; xi counts / gamma counts
    transition_counts = np.zeros((n_states, n_states))
    state_counts = np.zeros(n_states)
    
    for gamma, xi in zip(gamma_list, xi_list):
        transition_counts += np.sum(xi, axis=0)
        state_counts += np.sum(gamma[:-1], axis=0)
    A[:] = transition_counts / state_counts[:, None]

    # 3. mu; gamma-weighted mean of observations
    all_x = np.concatenate(x_list)
    all_gamma = np.concatenate(gamma_list, axis=0)
    weights = np.sum(all_gamma, axis=0)

    mu = np.dot(all_x, all_gamma) / weights
    phi['mu'][:] = mu

    # 4. sigma; gamma-weighted standard deviation around the new mu
    squared_diff = (all_x[:, None] - mu[None, :]) ** 2
    variance = np.sum(all_gamma * squared_diff, axis=0) / weights
    
    phi['sigma'][:] = np.sqrt(variance)

    return pi, A, phi


"""Putting them together"""
def fit_hmm(x_list, n_states):
    """Fit HMM parameters to observed data using Baum-Welch algorithm

    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        n_states (int): Number of latent states to use for the estimation.

    Returns:
        pi (np.ndarray): Initial state distribution
        A (np.ndarray): Time-independent stochastic transition matrix
        phi (Dict[np.ndarray]): Parameters for the Gaussian HMM model, contains
          two fields 'mu', 'sigma' for the mean and standard deviation
          respectively.

    """

    # We randomly initialize pi and A, and use k-means to initialize phi
    # Please do NOT change the initialization function since that will affect
    # grading
    pi, A, phi = initialize(n_states, x_list)

    """ YOUR CODE HERE
     Populate the values of pi, A, phi with the correct values. 
    """
    max_iters = 100 # super arbitrary number but tests show measured convergence: 19 (seq_short), 37 (seq_long)
    tol = 1e-4

    for i in range(max_iters):
        gamma_list, xi_list = e_step(x_list, pi, A, phi)
        new_pi, new_A, new_phi = m_step(x_list, gamma_list, xi_list)

        change = max(np.max(np.abs(new_pi - pi)),
                    np.max(np.abs(new_A - A)),
                    np.max(np.abs(new_phi['mu'] - phi['mu'])),
                    np.max(np.abs(new_phi['sigma'] - phi['sigma'])))

        pi, A, phi = new_pi, new_A, new_phi
        # print(f'iter {i + 1}: change = {change:.2e}')

        if change < tol:
            # print(f'converged after {i + 1} iterations')
            break
    # else:
    #     print(f'hit max_iters={max_iters} without converging (last change = {change:.2e})')

    return pi, A, phi
