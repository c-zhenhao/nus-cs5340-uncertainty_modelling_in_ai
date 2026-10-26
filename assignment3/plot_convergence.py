import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from lab3 import initialize, e_step, m_step

TOL = 1e-4


def log_lik(x_list, pi, A, phi):
    """ln p(X | theta) = sum of ln c_t from the scaled forward pass."""
    total = 0.0
    for x in x_list:
        B = scipy.stats.norm.pdf(x[:, None], loc=phi['mu'][None, :],
                                 scale=phi['sigma'][None, :])
        alpha = pi * B[0]
        c = np.sum(alpha)
        alpha = alpha / c
        total += np.log(c)
        for t in range(1, len(x)):
            alpha = B[t] * np.dot(alpha, A)
            c = np.sum(alpha)
            alpha = alpha / c
            total += np.log(c)
    return total


def trace(x_list, n_states, n_iters=150):
    """Run EM for a fixed number of iterations, with no early stop."""
    pi, A, phi = initialize(n_states, x_list)
    changes, lls = [], [log_lik(x_list, pi, A, phi)]   # lls[i] = after i iterations
    for _ in range(n_iters):
        gamma_list, xi_list = e_step(x_list, pi, A, phi)
        new_pi, new_A, new_phi = m_step(x_list, gamma_list, xi_list)
        changes.append(max(np.max(np.abs(new_pi - pi)),
                           np.max(np.abs(new_A - A)),
                           np.max(np.abs(new_phi['mu'] - phi['mu'])),
                           np.max(np.abs(new_phi['sigma'] - phi['sigma']))))
        pi, A, phi = new_pi, new_A, new_phi
        lls.append(log_lik(x_list, pi, A, phi))
    return np.array(changes), np.array(lls)


fig, axes = plt.subplots(2, 2, figsize=(11, 7))

for col, name in enumerate(['seq_short', 'seq_long']):
    d = np.load('data/{}.npz'.format(name))
    x_list = [np.asarray(x) for x in d['x']]
    changes, lls = trace(x_list, int(d['n_states']))

    iters = np.arange(1, len(changes) + 1)
    stop = int(np.argmax(changes < TOL)) + 1           # first iteration under tol
    gap = np.maximum(lls[-1] - lls, 1e-12)             # likelihood lost vs. final

    ax = axes[0, col]
    ax.semilogy(iters, changes)
    ax.axhline(TOL, color='gray', ls='--', label='tol = 1e-4')
    ax.axvline(stop, color='red', ls=':', label='converges at {}'.format(stop))
    ax.set(title=name, ylabel='max abs parameter change')
    ax.legend()

    ax = axes[1, col]
    ax.semilogy(np.arange(len(lls)), gap)
    ax.axvline(stop, color='red', ls=':')
    ax.set(xlabel='max_iters (cap)', ylabel='log-lik lost vs. 150 iters')

plt.tight_layout()
plt.savefig('convergence.png', dpi=150)
plt.show()