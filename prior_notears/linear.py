import argparse
from cmath import tau
import json
import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.optimize import check_grad
from scipy.special import expit as sigmoid
from scipy.optimize import approx_fprime
from notears import linear
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from Varsortability.src.varsortability import varsortability
import time
import traceback

def softplus(x, sharpness=50.0):
    return np.logaddexp(0.0, sharpness * x) / sharpness

def _minimize_finite(fun, x0, bounds, max_retries=16, max_restarts=32,
                     gtol=1e-5):
    """Recover from overflow without treating temporary bounds as convergence.

    Recenter on improving finite solutions and check stationarity against the
    original bounds. Return success=False on stagnation or the restart limit.
    """
    evaluations = 0

    def checked_fun(x):
        nonlocal evaluations
        evaluations += 1
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            value, gradient = fun(x)
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError('Non-finite objective or gradient')
        return value, gradient

    center = np.asarray(x0, dtype=float).copy()
    # Shrinking a search region cannot repair an invalid starting point.
    center_value, _ = checked_fun(center)
    lower, upper = np.asarray(bounds, dtype=float).T
    full_radius = float(np.max(upper - lower))
    radius = full_radius
    overflow_retries = 0
    iterations = 0

    for restart in range(max_restarts + 1):
        for attempt in range(max_retries + 1):
            trial_bounds = list(zip(np.maximum(lower, center - radius),
                                    np.minimum(upper, center + radius)))
            try:
                sol = sopt.minimize(
                    checked_fun, center, method='L-BFGS-B', jac=True,
                    bounds=trial_bounds,
                    options={"maxls": 100, "ftol": 1e-15, "gtol": gtol},
                )
                if not np.all(np.isfinite(sol.x)):
                    raise FloatingPointError('Non-finite optimizer weights')
                sol.fun, sol.jac = checked_fun(sol.x)
                break
            except FloatingPointError as exc:
                if attempt == max_retries:
                    raise FloatingPointError(
                        'Unable to find a finite inner solve after shrinking the search region'
                    ) from exc
                overflow_retries += 1
                radius *= 0.5
                print(f'Non-finite optimizer trial; retrying with radius={radius:g}',
                      flush=True)

        iterations += sol.nit
        sol.nit = iterations
        sol.nfev = evaluations
        sol.overflow_retries = overflow_retries
        sol.search_radius = radius
        sol.restarts = restart
        # Ignore convergence at artificial bounds: use the original bounds.
        projected = sol.x - np.clip(sol.x - sol.jac, lower, upper)
        sol.projected_gradient_inf = float(np.max(np.abs(projected)))
        if sol.projected_gradient_inf <= gtol:
            sol.success = True
            sol.status = 0
            sol.message = 'Converged under the original bounds'
            return sol

        moved = np.any(sol.x != center)
        if not moved or sol.fun >= center_value or restart == max_restarts:
            sol.success = False
            sol.status = 1
            sol.message = (
                'Restart limit reached before convergence under original bounds'
                if restart == max_restarts else
                'Stalled before convergence under original bounds'
            )
            return sol

        # Continue from the accepted finite point with original bounds restored.
        center = sol.x.copy()
        center_value = float(sol.fun)
        radius = full_radius

####just for test, to delete later #######################################

def _forbid_edges(W, edge_pairs, coefficient=None):
    """forbid edges from the list of index pairs.
    
    Args:
        W (np.ndarray): [d, d] weight matrix
        pairs (list): List of (i, j) edge pairs

    Returns:
        float: Values Sum of W[i, j] for each (i, j) pair
    """
    if len(edge_pairs) == 0:
        if coefficient is None:
            return np.empty(0)
        return np.empty(0), np.zeros(W.size)
    e = np.array([W[i, j] ** 2 for i, j in edge_pairs])
    if coefficient is None:
        return e
    coefficient = np.asarray(coefficient, dtype=float)
    if coefficient.shape != (len(edge_pairs),):
        raise ValueError("Provide one coefficient per edge pair")
    grad_W = np.zeros_like(W)
    for k, (i, j) in enumerate(edge_pairs):
        grad_W[i, j] += coefficient[k] * 2.0 * W[i, j]
    return e, grad_W.reshape(-1)

def _exist_edges(W, w_thres, edge_pairs, coefficient=None):
    """Return edge values and their coefficient-weighted gradient."""
    if len(edge_pairs) == 0:
        if coefficient is None:
            return np.empty(0)
        return np.empty(0), np.zeros(W.size)
    values = np.array([W[i, j] ** 2 - w_thres ** 2 for i, j in edge_pairs])
    if coefficient is None:
        return values
    coefficient = np.asarray(coefficient, dtype=float)
    if coefficient.shape != (len(edge_pairs),):
        raise ValueError("Provide one coefficient per edge pair")
    grad_W = np.zeros_like(W)
    for k, (i, j) in enumerate(edge_pairs):
        grad_W[i, j] += coefficient[k] * 2.0 * W[i, j]
    return values, grad_W.reshape(-1)

def _forbid_paths(W, path_pairs, coefficient=None):
    """Return path penalties and optionally their weighted gradient."""
    W = np.asarray(W, dtype=float)

    if coefficient is not None:
        coefficient = np.asarray(coefficient, dtype=float)
        if coefficient.shape != (len(path_pairs),):
            raise ValueError("Provide one coefficient per path pair")

    if len(path_pairs) == 0:
        if coefficient is None:
            return np.empty(0)
        return np.empty(0), np.zeros(W.size)

    A = W * W
    E = slin.expm(A)
    values = np.array([E[i, j] for i, j in path_pairs])

    if coefficient is None:
        return values

    M = np.zeros_like(W)
    for k, (i, j) in enumerate(path_pairs):
        M[i, j] += coefficient[k]

    grad_A = slin.expm_frechet(
        A.T, M, compute_expm=False
    )
    grad_W = 2.0 * W * grad_A

    return values, grad_W.reshape(-1)

def _exist_paths(W, w_thres, path_pairs, coefficient=None, sharpness=50.0):
    """Return path values and their coefficient-weighted gradient."""
    W = np.asarray(W, dtype=float)
    if coefficient is not None:
        coefficient = np.asarray(coefficient, dtype=float)
        if coefficient.shape != (len(path_pairs),):
            raise ValueError("Provide one coefficient per path pair")

    if len(path_pairs) == 0:
        if coefficient is None:
            return np.empty(0)
        return np.empty(0), np.zeros(W.size)


    X = W * W - w_thres * w_thres
    A = softplus(X, sharpness)
    E = slin.expm(A)

    values = np.array([E[i, j] for i, j in path_pairs])
    if coefficient is None:
        return values

    grad_E = np.zeros_like(E)
    for k, (i, j) in enumerate(path_pairs):
        grad_E[i, j] += coefficient[k]

    # Differentiate through the matrix exponential once.
    grad_A = slin.expm_frechet(
        A.T,
        grad_E,
        compute_expm=False,
    )

    # Chain rule through A = softplus(W**2 - w_thres**2).
    dA_dX = sigmoid(sharpness * X)
    grad_W = grad_A * dA_dX * (2.0 * W)

    return values, grad_W.reshape(-1)

def _exist_paths2(W, w_thres, path_pairs,
                sharpness=50.0, epsilon=10):
    """Return the sum of masked missing-path penalties and its gradient.

    p = sum_{k=1}^d abs(W)^k, using matrix powers.
    b[i, j] is one iff no walk of length 1 through d exists in
    the thresholded adjacency (abs(W) >= w_thres).
    The scalar value sums b * softplus(epsilon - p) over path_pairs.
    w_thres is epsilon_0 in the mask; epsilon is the softplus margin.
    sharpness=1 gives ordinary softplus.

    Always return (value, gradient), with gradient shaped (W.size,).
    Empty path_pairs returns zero and a zero gradient. The hard mask has zero
    derivative away from threshold crossings and is generally discontinuous
    at crossings. At W == 0, use zero as the abs subgradient.
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix")
    if len(path_pairs) == 0:
        return 0.0, np.zeros(W.size)

    d = W.shape[0]
    A = np.abs(W)
    powers = [np.eye(d)]
    p = np.zeros_like(W)
    adjacency = A >= w_thres
    reachable = np.zeros_like(adjacency)
    walk = np.eye(d, dtype=bool)
    for _ in range(d):
        powers.append(powers[-1] @ A)
        p += powers[-1]
        # Boolean matrix products avoid overflow from counting walks.
        walk = walk @ adjacency
        reachable |= walk

    b = ~reachable
    X = epsilon - p
    penalties = b * softplus(X, sharpness)
    value = float(sum(penalties[i, j] for i, j in path_pairs))

    grad_p = np.zeros_like(W)
    for i, j in path_pairs:
        grad_p[i, j] -= b[i, j] * sigmoid(sharpness * X[i, j])

    # Reverse through P_k = P_{k-1} @ A and p = sum_k P_k.
    grad_A = np.zeros_like(W)
    grad_power = np.zeros_like(W)
    for k in range(d, 0, -1):
        grad_power += grad_p
        grad_A += powers[k - 1].T @ grad_power
        grad_power = grad_power @ A.T

    return value, (grad_A * np.sign(W)).reshape(-1)


def _forbid_trek(W, trek_pairs, coefficient=None):
    """Return trek penalties and optionally their weighted gradient."""
    W = np.asarray(W, dtype=float)

    if coefficient is not None:
        coefficient = np.asarray(coefficient, dtype=float)
        if coefficient.shape != (len(trek_pairs),):
            raise ValueError("Provide one coefficient per trek pair")

    if len(trek_pairs) == 0:
        if coefficient is None:
            return np.empty(0)
        return np.empty(0), np.zeros(W.size)

    A = W * W
    E = slin.expm(A)
    T = E.T @ E
    values = np.array([T[i, j] for i, j in trek_pairs])

    if coefficient is None:
        return values

    M = np.zeros_like(W)
    for k, (i, j) in enumerate(trek_pairs):
        M[i, j] += coefficient[k]

    grad_E = E @ (M + M.T)
    grad_A = slin.expm_frechet(
        A.T, grad_E, compute_expm=False
    )
    grad_W = 2.0 * W * grad_A

    return values, grad_W.reshape(-1)

def _exist_trek(W, w_thres, trek_pairs, coefficient=None, sharpness=50.0):
    """Return trek values and their coefficient-weighted gradient.

    coefficient[k] weights the gradient for trek_pairs[k].
    weighted_grad has shape (W.size,).
    """
    W = np.asarray(W, dtype=float)
    if coefficient is not None:
        coefficient = np.asarray(coefficient, dtype=float)
        if coefficient.shape != (len(trek_pairs),):
            raise ValueError("Provide one coefficient per path pair")

    if len(trek_pairs) == 0:
        if coefficient is None:
            return np.empty(0)
        return np.empty(0), np.zeros(W.size)

    X = W * W - w_thres * w_thres
    A = softplus(X, sharpness)
    E = slin.expm(A)
    T = E.T @ E

    values = np.empty(len(trek_pairs), dtype=float)

    # Accumulate sum_k coefficient[k] * dv_k/dE.
    grad_E = np.zeros_like(E)
    values = np.array([T[i, j] for i, j in trek_pairs])
    if coefficient is None:
        return values
    for k, (i, j) in enumerate(trek_pairs):
        c = coefficient[k]
        # When i == j, both additions accumulate into the same column.
        grad_E[:, i] += c * E[:, j]
        grad_E[:, j] += c * E[:, i]

    
    # Differentiate through the matrix exponential once.
    grad_A = slin.expm_frechet(
        A.T,
        grad_E,
        compute_expm=False,
    )

    # Chain rule through A = softplus(W**2 - w_thres**2).
    dA_dX = sigmoid(sharpness * X)
    grad_W = grad_A * dA_dX * (2.0 * W)

    return values, grad_W.reshape(-1)

def combined_equality_constraints(W, forbid_edge_pairs, forbid_path_pairs, forbid_trek_pairs, coefficient=None):
    """Return values only, or (values, coefficient-weighted gradient)."""
    values = []
    gradients = []

    total = (
        len(forbid_edge_pairs)
        + len(forbid_path_pairs)
        + len(forbid_trek_pairs)
    )

    values_only = coefficient is None
    if values_only:
        coefficient = np.zeros(total)
    else:
        coefficient = np.asarray(coefficient, dtype=float)
        if coefficient.shape != (total,):
            raise ValueError("Provide one coefficient per equality penalty")

    offset = 0

    if forbid_edge_pairs:
        n = len(forbid_edge_pairs)
        edge_values, edge_grad = _forbid_edges(
            W, forbid_edge_pairs,
            coefficient[offset:offset + n],
        )
        values.append(np.atleast_1d(edge_values))
        gradients.append(edge_grad)
        offset += n

    if forbid_path_pairs:
        n = len(forbid_path_pairs)
        path_values, path_grad = _forbid_paths(
            W, forbid_path_pairs,
            coefficient[offset:offset + n],
        )
        values.append(np.atleast_1d(path_values))
        gradients.append(path_grad)
        offset += n

    if forbid_trek_pairs:
        n = len(forbid_trek_pairs)
        trek_values, trek_grad = _forbid_trek(
            W, forbid_trek_pairs,
            coefficient[offset:offset + n],
        )
        values.append(np.atleast_1d(trek_values))
        gradients.append(trek_grad)
        offset += n

    values = np.concatenate(values) if values else np.empty(0)

    if values_only:
        return values

    weighted_grad = (
        np.sum(gradients, axis=0)
        if gradients else np.zeros(W.size)
    )
    return values, weighted_grad

def combined_inequality_constraints(W, w_threshold, exist_edge_pairs, exist_path_pairs, exist_trek_pairs, coefficient=None):
    """Return values only, or (values, coefficient-weighted gradient)."""
    values = []
    gradients = []

    total = (
        len(exist_edge_pairs)
        + len(exist_path_pairs)
        + len(exist_trek_pairs)
    )

    values_only = coefficient is None
    if values_only:
        coefficient = np.zeros(total)
    else:
        coefficient = np.asarray(coefficient, dtype=float)
        if coefficient.shape != (total,):
            raise ValueError("Provide one coefficient per inequality")

    offset = 0

    if exist_edge_pairs:
        n = len(exist_edge_pairs)
        edge_values, edge_grad = _exist_edges(
            W,
            w_threshold,
            exist_edge_pairs,
            coefficient[offset:offset + n],
        )
        values.append(np.atleast_1d(edge_values))
        gradients.append(edge_grad)
        offset += n

    if exist_path_pairs:
        n = len(exist_path_pairs)
        path_values, path_grad = _exist_paths(
            W,
            w_threshold,
            exist_path_pairs,
            coefficient[offset:offset + n],
            sharpness=sharpness,
        )
        values.append(np.atleast_1d(path_values))
        gradients.append(path_grad)
        offset += n

    if exist_trek_pairs:
        n = len(exist_trek_pairs)
        trek_values, trek_grad = _exist_trek(
            W,
            w_threshold,
            exist_trek_pairs,
            coefficient[offset:offset + n],
            sharpness=sharpness,
        )
        values.append(np.atleast_1d(trek_values))
        gradients.append(trek_grad)

    values = np.concatenate(values) if values else np.empty(0)

    if values_only:
        return values

    weighted_grad = (
        np.sum(gradients, axis=0)
        if gradients else np.zeros(W.size)
    )
    return values, weighted_grad

def _h(W):
    """Evaluate value and gradient of acyclicity constraint."""
    E = slin.expm(W * W)  # (Zheng et al. 2018)
    h = np.trace(E) - d
    #     # A different formulation, slightly faster at the cost of numerical stability
    #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
    #     E = np.linalg.matrix_power(M, d - 1)
    #     h = (E.T * M).sum() - d
    G_h = E.T * W * 2
    return h, G_h

def _p0(W):
    """Evaluate value and gradient of prior distribution of W."""
    value = 0.5*np.sum(W ** 2)
    grad = W
    return value, grad.reshape(-1)

def evaluate_prior_values(W, prior_knowledge, w_threshold, compare = False):
    constraint_values = {}

    forbid_functions = {
        "forbid_edge_pairs": _forbid_edges,
        "forbid_path_pairs": _forbid_paths,
        "forbid_trek_pairs": _forbid_trek,
    }

    exist_functions = {
        "exist_edge_pairs": _exist_edges,
        "exist_path_pairs": _exist_paths if not compare else _exist_paths2,
        "exist_trek_pairs": _exist_trek,
    }

    for prior_key, pairs in prior_knowledge.items():
        if not pairs:
            constraint_values[prior_key] = []
            continue

        if prior_key in forbid_functions:
            constraint_function = forbid_functions[prior_key]

            # Call with one pair at a time because forbidden functions
            # otherwise return the mean over all supplied pairs.
            values = [
                float(constraint_function(W, [pair])[0])
                for pair in pairs
            ]

        elif compare and prior_key == "exist_path_pairs":
            values = [
                _exist_paths2(W, w_threshold, [pair])[0]
                for pair in pairs
            ]

        elif prior_key in exist_functions:
            values = exist_functions[prior_key](
                W,
                w_threshold,
                pairs,
            )

            values = np.asarray(values).reshape(-1).tolist()

        else:
            raise ValueError(
                f"Unknown prior-knowledge key: {prior_key}"
            )

        constraint_values[prior_key] = values

    return constraint_values

def _loss(W, X, loss_type):
    """Evaluate value and gradient of loss."""
    M = X @ W
    if loss_type == 'l2':
        R = X - M
        loss = 0.5 / X.shape[0] * (R ** 2).sum()
        G_loss = - 1.0 / X.shape[0] * X.T @ R
        return loss, G_loss
    elif loss_type == 'likelihood':
        R = X - M
        residual_var = np.mean(R ** 2, axis=0)
        A = np.eye(W.shape[0]) - W
        det_sign, log_det = np.linalg.slogdet(A)
        if det_sign == 0 or np.any(residual_var <= 0):
            return np.inf, np.zeros(W.size, dtype=W.dtype)
        loss = 0.5 * np.log(residual_var).sum() - log_det
        G_loss = -(X.T @ R) / (X.shape[0] * residual_var)
        G_loss += np.linalg.inv(A).T
        return loss, G_loss.ravel()
####just for test, to delete later #######################################

def notears_linear(X, lambda1, loss_type, tau, prior_knowledge=None, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3, sharpness=50.0, epsilon=1e-1, compare = False):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, likelihood, logistic, poisson
        tau (float): trust
        prior_knowledge (dict): prior knowledge
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if h(W) <= h_tol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold
        sharpness (float): softplus sharpness parameter
        epsilon (float): inequality constraint tolerance

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    if prior_knowledge is None:
        prior_knowledge = {}

    forbid_edge_pairs = prior_knowledge.get("forbid_edge_pairs", [])
    forbid_path_pairs = prior_knowledge.get("forbid_path_pairs", [])
    forbid_trek_pairs = prior_knowledge.get("forbid_trek_pairs", [])

    exist_edge_pairs = prior_knowledge.get("exist_edge_pairs", [])
    exist_path_pairs = prior_knowledge.get("exist_path_pairs", [])
    exist_trek_pairs = prior_knowledge.get("exist_trek_pairs", [])

    tau = np.array(tau, dtype=np.float64)

    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'likelihood':
            """Evaluate value and gradient of loss."""
            R = X - M
            residual_var = np.mean(R ** 2, axis=0)
            A = np.eye(W.shape[0]) - W
            det_sign, log_det = np.linalg.slogdet(A)
            if det_sign == 0 or np.any(residual_var <= 0):
                return np.inf, np.zeros(W.size, dtype=W.dtype)
            loss = 0.5 * np.log(residual_var).sum() - log_det
            G_loss = -(X.T @ R) / (X.shape[0] * residual_var)
            G_loss += np.linalg.inv(A).T
            return loss, G_loss.ravel()
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _p0(W):
        """Evaluate value and gradient of prior distribution of W."""
        value = 0.5*np.sum(W ** 2)
        grad = W
        return value, grad.reshape(-1)

    def _forbid_edges(W, edge_pairs, coefficient=None):
        """forbid edges from the list of index pairs.
        
        Args:
            W (np.ndarray): [d, d] weight matrix
            pairs (list): List of (i, j) edge pairs

        Returns:
            float: Values Sum of W[i, j] for each (i, j) pair
        """
        if len(edge_pairs) == 0:
            if coefficient is None:
                return np.empty(0)
            return np.empty(0), np.zeros(W.size)
        e = np.array([W[i, j] ** 2 for i, j in edge_pairs])
        if coefficient is None:
            return e
        coefficient = np.asarray(coefficient, dtype=float)
        if coefficient.shape != (len(edge_pairs),):
            raise ValueError("Provide one coefficient per edge pair")
        grad_W = np.zeros_like(W)
        for k, (i, j) in enumerate(edge_pairs):
            grad_W[i, j] += coefficient[k] * 2.0 * W[i, j]
        return e, grad_W.reshape(-1)

    def _exist_edges(W, w_thres, edge_pairs, coefficient=None):
        """Return edge values and their coefficient-weighted gradient."""
        if len(edge_pairs) == 0:
            if coefficient is None:
                return np.empty(0)
            return np.empty(0), np.zeros(W.size)
        values = np.array([W[i, j] ** 2 - w_thres ** 2 for i, j in edge_pairs])
        if coefficient is None:
            return values
        coefficient = np.asarray(coefficient, dtype=float)
        if coefficient.shape != (len(edge_pairs),):
            raise ValueError("Provide one coefficient per edge pair")
        grad_W = np.zeros_like(W)
        for k, (i, j) in enumerate(edge_pairs):
            grad_W[i, j] += coefficient[k] * 2.0 * W[i, j]
        return values, grad_W.reshape(-1)

    def _forbid_paths(W, path_pairs, coefficient=None):
        """Return path penalties and optionally their weighted gradient."""
        W = np.asarray(W, dtype=float)

        if coefficient is not None:
            coefficient = np.asarray(coefficient, dtype=float)
            if coefficient.shape != (len(path_pairs),):
                raise ValueError("Provide one coefficient per path pair")

        if len(path_pairs) == 0:
            if coefficient is None:
                return np.empty(0)
            return np.empty(0), np.zeros(W.size)

        A = W * W
        E = slin.expm(A)
        values = np.array([E[i, j] for i, j in path_pairs])

        if coefficient is None:
            return values

        M = np.zeros_like(W)
        for k, (i, j) in enumerate(path_pairs):
            M[i, j] += coefficient[k]

        grad_A = slin.expm_frechet(
            A.T, M, compute_expm=False
        )
        grad_W = 2.0 * W * grad_A

        return values, grad_W.reshape(-1)

    def _exist_paths(W, w_thres, path_pairs, coefficient=None, sharpness=50.0):
        """Return path values and their coefficient-weighted gradient."""
        W = np.asarray(W, dtype=float)
        if coefficient is not None:
            coefficient = np.asarray(coefficient, dtype=float)
            if coefficient.shape != (len(path_pairs),):
                raise ValueError("Provide one coefficient per path pair")

        if len(path_pairs) == 0:
            if coefficient is None:
                return np.empty(0)
            return np.empty(0), np.zeros(W.size)


        X = W * W - w_thres * w_thres
        A = softplus(X, sharpness)
        E = slin.expm(A)

        values = np.array([E[i, j] for i, j in path_pairs])
        if coefficient is None:
            return values

        grad_E = np.zeros_like(E)
        for k, (i, j) in enumerate(path_pairs):
            grad_E[i, j] += coefficient[k]

        # Differentiate through the matrix exponential once.
        grad_A = slin.expm_frechet(
            A.T,
            grad_E,
            compute_expm=False,
        )

        # Chain rule through A = softplus(W**2 - w_thres**2).
        dA_dX = sigmoid(sharpness * X)
        grad_W = grad_A * dA_dX * (2.0 * W)

        return values, grad_W.reshape(-1)

    def _exist_paths2(W, w_thres, path_pairs,
                  sharpness=50.0, epsilon=10):
        """Return the sum of masked missing-path penalties and its gradient.

        p = sum_{k=1}^d abs(W)^k, using matrix powers.
        b[i, j] is one iff no walk of length 1 through d exists in
        the thresholded adjacency (abs(W) >= w_thres).
        The scalar value sums b * softplus(epsilon - p) over path_pairs.
        w_thres is epsilon_0 in the mask; epsilon is the softplus margin.
        sharpness=1 gives ordinary softplus.

        Always return (value, gradient), with gradient shaped (W.size,).
        Empty path_pairs returns zero and a zero gradient. The hard mask has zero
        derivative away from threshold crossings and is generally discontinuous
        at crossings. At W == 0, use zero as the abs subgradient.
        """
        W = np.asarray(W, dtype=float)
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("W must be a square matrix")
        if len(path_pairs) == 0:
            return 0.0, np.zeros(W.size)

        d = W.shape[0]
        A = np.abs(W)
        powers = [np.eye(d)]
        p = np.zeros_like(W)
        adjacency = A >= w_thres
        reachable = np.zeros_like(adjacency)
        walk = np.eye(d, dtype=bool)
        for _ in range(d):
            powers.append(powers[-1] @ A)
            p += powers[-1]
            # Boolean matrix products avoid overflow from counting walks.
            walk = walk @ adjacency
            reachable |= walk

        b = ~reachable
        X = epsilon - p
        penalties = b * softplus(X, sharpness)
        value = float(sum(penalties[i, j] for i, j in path_pairs))

        grad_p = np.zeros_like(W)
        for i, j in path_pairs:
            grad_p[i, j] -= b[i, j] * sigmoid(sharpness * X[i, j])

        # Reverse through P_k = P_{k-1} @ A and p = sum_k P_k.
        grad_A = np.zeros_like(W)
        grad_power = np.zeros_like(W)
        for k in range(d, 0, -1):
            grad_power += grad_p
            grad_A += powers[k - 1].T @ grad_power
            grad_power = grad_power @ A.T

        return value, (grad_A * np.sign(W)).reshape(-1)

    
    def _forbid_trek(W, trek_pairs, coefficient=None):
        """Return trek penalties and optionally their weighted gradient."""
        W = np.asarray(W, dtype=float)

        if coefficient is not None:
            coefficient = np.asarray(coefficient, dtype=float)
            if coefficient.shape != (len(trek_pairs),):
                raise ValueError("Provide one coefficient per trek pair")

        if len(trek_pairs) == 0:
            if coefficient is None:
                return np.empty(0)
            return np.empty(0), np.zeros(W.size)

        A = W * W
        E = slin.expm(A)
        T = E.T @ E
        values = np.array([T[i, j] for i, j in trek_pairs])

        if coefficient is None:
            return values

        M = np.zeros_like(W)
        for k, (i, j) in enumerate(trek_pairs):
            M[i, j] += coefficient[k]

        grad_E = E @ (M + M.T)
        grad_A = slin.expm_frechet(
            A.T, grad_E, compute_expm=False
        )
        grad_W = 2.0 * W * grad_A

        return values, grad_W.reshape(-1)
    
    def _exist_trek(W, w_thres, trek_pairs, coefficient=None, sharpness=50.0):
        """Return trek values and their coefficient-weighted gradient.

        coefficient[k] weights the gradient for trek_pairs[k].
        weighted_grad has shape (W.size,).
        """
        W = np.asarray(W, dtype=float)
        if coefficient is not None:
            coefficient = np.asarray(coefficient, dtype=float)
            if coefficient.shape != (len(trek_pairs),):
                raise ValueError("Provide one coefficient per path pair")

        if len(trek_pairs) == 0:
            if coefficient is None:
                return np.empty(0)
            return np.empty(0), np.zeros(W.size)

        X = W * W - w_thres * w_thres
        A = softplus(X, sharpness)
        E = slin.expm(A)
        T = E.T @ E

        values = np.empty(len(trek_pairs), dtype=float)

        # Accumulate sum_k coefficient[k] * dv_k/dE.
        grad_E = np.zeros_like(E)
        values = np.array([T[i, j] for i, j in trek_pairs])
        if coefficient is None:
            return values
        for k, (i, j) in enumerate(trek_pairs):
            c = coefficient[k]
            # When i == j, both additions accumulate into the same column.
            grad_E[:, i] += c * E[:, j]
            grad_E[:, j] += c * E[:, i]

        
        # Differentiate through the matrix exponential once.
        grad_A = slin.expm_frechet(
            A.T,
            grad_E,
            compute_expm=False,
        )

        # Chain rule through A = softplus(W**2 - w_thres**2).
        dA_dX = sigmoid(sharpness * X)
        grad_W = grad_A * dA_dX * (2.0 * W)

        return values, grad_W.reshape(-1)

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def combined_equality_constraints(W, coefficient=None):
        """Return values only, or (values, coefficient-weighted gradient)."""
        values = []
        gradients = []

        total = (
            len(forbid_edge_pairs)
            + len(forbid_path_pairs)
            + len(forbid_trek_pairs)
            + int(compare and bool(exist_path_pairs))
        )

        values_only = coefficient is None
        if values_only:
            coefficient = np.zeros(total)
        else:
            coefficient = np.asarray(coefficient, dtype=float)
            if coefficient.shape != (total,):
                raise ValueError("Provide one coefficient per equality penalty")

        offset = 0

        if forbid_edge_pairs:
            n = len(forbid_edge_pairs)
            edge_values, edge_grad = _forbid_edges(
                W, forbid_edge_pairs,
                coefficient[offset:offset + n],
            )
            values.append(np.atleast_1d(edge_values))
            gradients.append(edge_grad)
            offset += n

        if forbid_path_pairs:
            n = len(forbid_path_pairs)
            path_values, path_grad = _forbid_paths(
                W, forbid_path_pairs,
                coefficient[offset:offset + n],
            )
            values.append(np.atleast_1d(path_values))
            gradients.append(path_grad)
            offset += n

        if forbid_trek_pairs:
            n = len(forbid_trek_pairs)
            trek_values, trek_grad = _forbid_trek(
                W, forbid_trek_pairs,
                coefficient[offset:offset + n],
            )
            values.append(np.atleast_1d(trek_values))
            gradients.append(trek_grad)
            offset += n

        if compare and exist_path_pairs:
            # _exist_paths2 returns one aggregate scalar penalty.
            path_value, path_grad = _exist_paths2(
                W, w_threshold, exist_path_pairs,
                sharpness=sharpness, epsilon=10,
            )
            values.append(np.atleast_1d(path_value))
            gradients.append(coefficient[offset] * path_grad)

        values = np.concatenate(values) if values else np.empty(0)

        if values_only:
            return values

        weighted_grad = (
            np.sum(gradients, axis=0)
            if gradients else np.zeros(W.size)
        )
        return values, weighted_grad

    def combined_inequality_constraints(W, coefficient=None):
        """Return values only, or (values, coefficient-weighted gradient)."""
        values = []
        gradients = []

        total = (
            len(exist_edge_pairs)
            + (0 if compare else len(exist_path_pairs))
            + len(exist_trek_pairs)
        )

        values_only = coefficient is None
        if values_only:
            coefficient = np.zeros(total)
        else:
            coefficient = np.asarray(coefficient, dtype=float)
            if coefficient.shape != (total,):
                raise ValueError("Provide one coefficient per inequality")

        offset = 0

        if exist_edge_pairs:
            n = len(exist_edge_pairs)
            edge_values, edge_grad = _exist_edges(
                W,
                w_threshold,
                exist_edge_pairs,
                coefficient[offset:offset + n],
            )
            values.append(np.atleast_1d(edge_values))
            gradients.append(edge_grad)
            offset += n

        if exist_path_pairs and not compare:
            n = len(exist_path_pairs)
            path_values, path_grad = _exist_paths(
                W,
                w_threshold,
                exist_path_pairs,
                coefficient[offset:offset + n],
                sharpness=sharpness,
            )
            values.append(np.atleast_1d(path_values))
            gradients.append(path_grad)
            offset += n

        if exist_trek_pairs:
            n = len(exist_trek_pairs)
            trek_values, trek_grad = _exist_trek(
                W,
                w_threshold,
                exist_trek_pairs,
                coefficient[offset:offset + n],
                sharpness=sharpness,
            )
            values.append(np.atleast_1d(trek_values))
            gradients.append(trek_grad)

        values = np.concatenate(values) if values else np.empty(0)

        if values_only:
            return values

        weighted_grad = (
            np.sum(gradients, axis=0)
            if gradients else np.zeros(W.size)
        )
        return values, weighted_grad

    # original version of _func
    # def _func(w):
    #     """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
    #     W = _adj(w)
    #     loss, G_loss = _loss(W)
    #     h, G_h = _h(W)
    #     obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
    #     G_smooth = G_loss + (rho * h + alpha) * G_h
    #     g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
    #     return obj, g_obj

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        c_e = combined_equality_constraints(W, coefficient=None)
        i_value = combined_inequality_constraints(W, coefficient=None)
        tau_e = tau[:c_e.size]
        tau_i = tau[c_e.size:]
        residual = epsilon - i_value
        c_i = softplus(residual, sharpness)
        c = np.concatenate((c_e.reshape(-1), c_i.reshape(-1)))
        _, G_e = combined_equality_constraints(W, coefficient=tau_e)
        coefficient = -tau_i * sigmoid(sharpness * residual)
        _, G_i = combined_inequality_constraints(W, coefficient=coefficient)
        G_prior = G_e + G_i
        p0, G_p0 = _p0(W)
        obj = loss + p0 + tau @ c + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss.reshape(-1) + G_p0 + G_prior + (rho * h + alpha) * G_h.reshape(-1)
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    # def _violation(c_e, c_i):
    #     e = (
    #         np.linalg.norm(c_e, ord=np.inf)
    #         if c_e.size
    #         else 0.0
    #     )
    #     i = (
    #         np.linalg.norm(np.maximum(c_i, 0.0), ord=np.inf)
    #         if c_i.size
    #         else 0.0
    #     )
    #     return max(e, i)

    def _violation(c_e, c_i):
        active_i = np.maximum(c_i, 0.0)

        # all_violations = np.concatenate([
        #     np.asarray(c_e).reshape(-1),
        #     active_i.reshape(-1),
        # ])

        l2_violation_i = np.linalg.norm(
            active_i,
            ord=2,
        )

        l2_violation_e = np.linalg.norm(
            c_e,
            ord=2,
        )

        max_violation_i = (
            np.linalg.norm(active_i, ord=np.inf) if active_i.size else 0.0
        )

        max_violation_e = np.linalg.norm(
            c_e,
            ord=np.inf,
        )

        return l2_violation_i, max_violation_i, l2_violation_e, max_violation_e

    n, d = X.shape
    # w_est = np.random.uniform(0.0, 0.1, size=2 * d * d)  # double w_est into (w_pos, w_neg)
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf
    weight_bound = 5.0
    bnds = [(0, 0) if i == j else (0, weight_bound) for _ in range(2) for i in range(d) for j in range(d)]

    if loss_type in ('l2', 'likelihood'):
        X = X - np.mean(X, axis=0, keepdims=True)
    for outer_iter in range(max_iter):
        w_new, h_new = None, None
        inner_attempt = 0
        while rho < rho_max:
            inner_attempt += 1
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds, options={"maxls": 100, "ftol": 1e-15})

            if not sol.success:
                print("L-BFGS-B warning:", sol.message)
                print("rho:", rho)

            if not np.isfinite(sol.fun):
                raise FloatingPointError(
                    "The augmented objective became non-finite"
                )

            if not np.all(np.isfinite(sol.x)):
                raise FloatingPointError(
                    "The optimizer returned non-finite weights"
                )
            
            diagnostics = {
                "outer_iter": outer_iter + 1,
                "inner_attempt": inner_attempt,
                "success": bool(sol.success),
                "status": int(sol.status),
                "message": str(sol.message),
                "nit": int(sol.nit),
                "nfev": int(sol.nfev),
                "objective": float(sol.fun),
                "weight_change_inf": float(np.max(np.abs(_adj(sol.x) - _adj(w_est)))),
                "rho": float(rho),
                "alpha": float(alpha),
            }
            print("inner_solver:", json.dumps(diagnostics), flush=True)
            w_new = sol.x
            c_e_new = combined_equality_constraints(_adj(w_new))
            i_value_new = combined_inequality_constraints(_adj(w_new))
            c_i_new = np.maximum(epsilon - i_value_new, 0.0)
            ###############################
            print("equality constraints:", c_e_new, flush=True)
            print("inequality constraints:", c_i_new, flush=True)
            ###############################
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            print("final rho:", rho)
            print("final h:", h)
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est, bool(sol.success)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='linear NOTEARS with prior knowledge',)

    parser.add_argument('-s', '--seed', dest='s',  default=0, type=int)
    parser.add_argument('-d', '--num_nodes', dest='d', default=10, type=int)
    parser.add_argument('-e', '--num_edges_per_node', dest='e', default=1, type=int)
    parser.add_argument('-g', '--graph_type', dest='g', default="ER", type=str)
    parser.add_argument('-l', '--loss_type', dest='l', default="both", type=str)
    parser.add_argument('-n', '--noise', dest='n', default="gauss", type=str)
    parser.add_argument('-p', '--prior_type', dest='p', default="mix", type=str)
    parser.add_argument('-r', '--prior_rate', dest='r', default=0.25, type=float)
    parser.add_argument('-t', '--w_threshold', dest='t', default=0.3, type=float)
    parser.add_argument('-ep', '--epsilon', dest='ep', default=1e-1, type=float)
    parser.add_argument('-c', '--compare', dest='c', action='store_true', default=False)
    parser.add_argument('-tau', '--tau', dest='tau', default=[1.0, 1.0], nargs='+', type=float)
    args = parser.parse_args()

    from prior_notears import utils
    utils.set_random_seed(args.s)
    n, d, s0, graph_type, sem_type = 10*args.d, args.d, args.e*args.d, args.g, args.n
    B_true = utils.simulate_dag(d, s0, graph_type)
    print("B_true:", B_true)
    W_true = utils.simulate_parameter(B_true)
    if not args.c:
        filename = f"linear_{args.p}_{graph_type}{args.e}_d{d}_{sem_type}_rate{args.r}_epsilon{args.ep}_seed{args.s}_tau{args.tau[0]}_{args.tau[1]}.json"
    else:
        filename = f"linear_{args.p}_{graph_type}{args.e}_d{d}_{sem_type}_rate{args.r}_epsilon10_seed{args.s}_tau{args.tau[0]}_{args.tau[1]}_compare.json"

    noise_scale = np.exp(np.random.uniform(np.log(0.5), np.log(2.0), size=d,))
    X = utils.simulate_linear_sem(W_true, n, sem_type, noise_scale)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    varsortability_score = varsortability(X_std, W_true)
    evaluation_status = {}

    if args.l in ('both', 'likelihood'):
        print(f'>>> Evaluation without prior knowledge and likelihood loss <<<')
        start_time = time.perf_counter()
        W_est_no_prior_ll = linear.notears_linear(X_std, lambda1=0.1, loss_type="likelihood", w_threshold=args.t)
        running_time_no_prior_ll = time.perf_counter() - start_time
        h_est_no_prior_ll, _ = _h(W_est_no_prior_ll)
        loss_no_prior_ll, _ = _loss(W_est_no_prior_ll, X_std, "likelihood")
        try:
            if not utils.is_dag(W_est_no_prior_ll):
                raise ValueError("Estimated graph contains a directed cycle")

            acc_no_prior_ll = utils.count_accuracy(
                B_true, W_est_no_prior_ll != 0
            )
        except Exception as exc:
            evaluation_status["no_prior_ll"] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            print(f"Evaluation failed: {exc}")
        else:
            evaluation_status["no_prior_ll"] = {"status": "completed"}

        # if 'exist' in args.p:
        #     prior_knowledge_ll = utils.generate_prior_knowledge(
        #         B_true,
        #         prior_rate=args.r,
        #         prior_type=args.p,
        #     )
        # elif 'forbid' in args.p:
        #     prior_knowledge_ll = utils.generate_unsatisfed_prior_knowledge(
        #         B_true,
        #         B_est=W_est_no_prior_ll,
        #         prior_rate=args.r,
        #         prior_type=args.p,
        #     )
        # elif args.p=='mix':
        #     prior_knowledge_ll = utils.generate_mixed_prior_knowledge(
        #         B_true,
        #         B_est=W_est_no_prior_ll,
        #         prior_rate=args.r,
        #         prior_type=args.p,
        #     )
        prior_knowledge_ll = {"exist_edge_pairs": [(7, 0), (7, 2)]}
        print("prior_knowledge_ll:", prior_knowledge_ll)
        constraint_values_no_prior_ll = evaluate_prior_values(W_est_no_prior_ll, prior_knowledge_ll, args.t)
        satisfied_no_prior_ll, satisfied_percentage_no_prior_ll = utils.evaluate_prior_knowledge(W_est_no_prior_ll, prior_knowledge_ll)
        print(f'>>> Evaluation with prior knowledge and likelihood loss <<<')
        start_time = time.perf_counter()
        W_est_prior_ll, sol_success_ll = notears_linear(X_std, lambda1=0.1, loss_type="likelihood", tau=args.tau, prior_knowledge=prior_knowledge_ll, w_threshold=args.t, epsilon=args.ep, compare=args.c,)
        running_time_prior_ll = time.perf_counter() - start_time
        h_est_prior_ll, _ = _h(W_est_prior_ll)
        loss_prior_ll, _ = _loss(W_est_prior_ll, X_std, "likelihood")
        constraint_values_prior_ll = evaluate_prior_values(W_est_prior_ll, prior_knowledge_ll, args.t, args.c)
        satisfied_prior_ll, satisfied_percentage_prior_ll = utils.evaluate_prior_knowledge(W_est_prior_ll, prior_knowledge_ll)
        try:
            if not utils.is_dag(W_est_prior_ll):
                raise ValueError("Estimated graph contains a directed cycle")

            acc_prior_ll = utils.count_accuracy(
                B_true, W_est_prior_ll != 0
            )
        except Exception as exc:
            evaluation_status["prior_ll"] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            print(f"Evaluation failed: {exc}")
        else:
            evaluation_status["prior_ll"] = {"status": "completed"}

    if args.l in ('both', 'l2'):
            print(f'>>> Evaluation without prior knowledge and l2 loss <<<')
            start_time = time.perf_counter()
            W_est_no_prior_l2 = linear.notears_linear(X_std, lambda1=0.1, loss_type="l2", w_threshold=args.t)
            running_time_no_prior_l2 = time.perf_counter() - start_time
            h_est_no_prior_l2, _ = _h(W_est_no_prior_l2)
            loss_no_prior_l2, _ = _loss(W_est_no_prior_l2, X_std, "l2")
            try:
                if not utils.is_dag(W_est_no_prior_l2):
                    raise ValueError("Estimated graph contains a directed cycle")
    
                acc_no_prior_l2 = utils.count_accuracy(
                    B_true, W_est_no_prior_l2 != 0
                )
            except Exception as exc:
                evaluation_status["no_prior_l2"] = {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                print(f"Evaluation failed: {exc}")
            else:
                evaluation_status["no_prior_l2"] = {"status": "completed"}

            # if 'exist' in args.p:
            #     prior_knowledge_l2 = utils.generate_prior_knowledge(
            #         B_true,
            #         prior_rate=args.r,
            #         prior_type=args.p,
            #     )
            # elif 'forbid' in args.p:
            #     prior_knowledge_l2 = utils.generate_unsatisfed_prior_knowledge(
            #         B_true,
            #         B_est=W_est_no_prior_l2,
            #         prior_rate=args.r,
            #         prior_type=args.p,
            #     )
            # elif args.p=='mix':
            #     prior_knowledge_l2 = utils.generate_mixed_prior_knowledge(
            #         B_true,
            #         B_est=W_est_no_prior_l2,
            #         prior_rate=args.r,
            #         prior_type=args.p,
            #     )
            prior_knowledge_l2 = {"exist_edge_pairs": [(7, 0), (7, 2)]}
            print("prior_knowledge_l2:", prior_knowledge_l2)
            constraint_values_no_prior_l2 = evaluate_prior_values(W_est_no_prior_l2, prior_knowledge_l2, args.t)
            satisfied_no_prior_l2, satisfied_percentage_no_prior_l2 = utils.evaluate_prior_knowledge(W_est_no_prior_l2, prior_knowledge_l2)
            print(f'>>> Evaluation with prior knowledge and l2 loss <<<')
            start_time = time.perf_counter()
            W_est_prior_l2, sol_success_l2 = notears_linear(X_std, lambda1=0.1, loss_type="l2", tau=args.tau, prior_knowledge=prior_knowledge_l2, w_threshold=args.t, epsilon=args.ep, compare=args.c)
            running_time_prior_l2 = time.perf_counter() - start_time
            h_est_prior_l2, _ = _h(W_est_prior_l2)
            loss_prior_l2, _ = _loss(W_est_prior_l2, X_std, "l2")
            constraint_values_prior_l2 = evaluate_prior_values(W_est_prior_l2, prior_knowledge_l2, args.t, args.c)
            satisfied_prior_l2, satisfied_percentage_prior_l2 = utils.evaluate_prior_knowledge(W_est_prior_l2, prior_knowledge_l2)
            try:
                if not utils.is_dag(W_est_prior_l2):
                    raise ValueError("Estimated graph contains a directed cycle")
    
                acc_prior_l2 = utils.count_accuracy(
                    B_true, W_est_prior_l2 != 0
                )
            except Exception as exc:
                evaluation_status["prior_l2"] = {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                print(f"Evaluation failed: {exc}")
            else:
                evaluation_status["prior_l2"] = {"status": "completed"}

    results = {
        "B_true": B_true.tolist(),
        "W_true": W_true.tolist(),
        "X": X.tolist(),
        "X_std": X_std.tolist(),
        "varsortability_score": varsortability_score,
        "evaluation_status": evaluation_status,
    }

    optional_result_names = (
        "prior_knowledge_ll",
        "prior_knowledge_l2",
        "W_est_prior_ll",
        "W_est_prior_l2",
        "W_est_no_prior_ll",
        "W_est_no_prior_l2",
        "h_est_prior_ll",
        "h_est_prior_l2",
        "h_est_no_prior_ll",
        "h_est_no_prior_l2",
        "acc_prior_ll",
        "acc_prior_l2",
        "acc_no_prior_ll",
        "acc_no_prior_l2",
        "loss_prior_ll",
        "loss_prior_l2",
        "loss_no_prior_ll",
        "loss_no_prior_l2",
        "constraint_values_prior_ll",
        "constraint_values_prior_l2",
        "constraint_values_no_prior_ll",
        "constraint_values_no_prior_l2",
        "satisfied_prior_ll",
        "satisfied_percentage_prior_ll",
        "satisfied_prior_l2",
        "satisfied_percentage_prior_l2",
        "satisfied_no_prior_ll",
        "satisfied_percentage_no_prior_ll",
        "satisfied_no_prior_l2",
        "satisfied_percentage_no_prior_l2",
        "sol_success_ll",
        "sol_success_l2",
        "running_time_prior_ll",
        "running_time_prior_l2",
        "running_time_no_prior_ll",
        "running_time_no_prior_l2"
    )
    current_scope = locals()
    for name in optional_result_names:
        if name in current_scope:
            value = current_scope[name]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, np.generic):
                value = value.item()
            results[name] = value

    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / f"linear_{args.p}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    with output_path.open("w") as file:
        json.dump(results, file, indent=4)

    print(f"Results saved to: {output_path}")

#### check gradient of prior knowledge constraints

    # d = 4
    # W = np.array([
    #     [0.0,  0.6, -0.35,  0.8],
    #     [0.2,  0.0,  0.5,  -0.4],
    #     [0.1,  0.4,  0.0,   0.7],
    #     [-0.1, 0.35, 0.2,   0.0],
    # ], dtype=float)

    # path_pairs = [(0, 1), (1, 3), (2, 0)]
    # w_thres = 0.3
    # epsilon = 0.1

    # # Keep the zero diagonal fixed: abs(W) is not differentiable at zero.
    # off_diag = ~np.eye(d, dtype=bool)

    # def unpack(x):
    #     W_test = W.copy()
    #     W_test[off_diag] = x
    #     return W_test

    # def objective(x):
    #     value, _ = _exist_paths2(
    #         unpack(x), w_thres, path_pairs, epsilon=epsilon,
    #     )
    #     return value

    # def gradient(x):
    #     _, grad = _exist_paths2(
    #         unpack(x), w_thres, path_pairs,
    #         epsilon=epsilon,
    #     )
    #     return grad.reshape(d, d)[off_diag]

    # x0 = W[off_diag].copy()
    # absolute_error = check_grad(objective, gradient, x0)

    # print(f"absolute_error = {absolute_error:.3e}")
    # assert absolute_error < 1e-6
    # print("values: ", _exist_paths2(W, w_thres, path_pairs, epsilon=epsilon))

    # d = 4
    # W = np.array([
    #     [1.0, 2.0, 0.3, 0.8],
    #     [0.2, 1.0, 0.5, 1.0],
    #     [0.3, 0.4, 1.0, 0.7],
    #     [0.4, 0.3, 0.2, 1.0],
    # ], dtype=float)
    # edge_pairs = [(0, 1), (1, 2), (2, 3)]
    # path_pairs = [(0, 1), (1, 2), (2, 3)]
    # trek_pairs = [(0, 1), (1, 2), (2, 3)]
    # coefficient = np.array([1.0, 0.5, 0.8])
    # w_threshold = 0.3
    # epsilon = 0.1
    # sharpness = 50.0

    # # Fixed inputs and multipliers: finite differences require a deterministic
    # # objective. Check the L2 objective in the same doubled coordinates as SciPy.
    # rng = np.random.default_rng(0)
    # X = rng.normal(size=(100, d))
    # X -= X.mean(axis=0, keepdims=True)
    # forbid_edge_pairs = edge_pairs
    # forbid_path_pairs = path_pairs
    # forbid_trek_pairs = trek_pairs
    # exist_edge_pairs = edge_pairs
    # exist_path_pairs = path_pairs
    # exist_trek_pairs = trek_pairs
    # rho, alpha, lambda1 = 1.0, 0.3, 0.1
    # prior_count = sum(len(pairs) for pairs in (
    #     forbid_edge_pairs, forbid_path_pairs, forbid_trek_pairs,
    #     exist_edge_pairs, exist_path_pairs, exist_trek_pairs,
    # ))
    # tau = np.linspace(0.0, 1.0, prior_count)
    # np.fill_diagonal(W, 0.0)

    # def _func(W):
    #     """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
    #     W = W.reshape(d, d)
    #     loss, G_loss = _loss(W, X)
    #     h, G_h = _h(W)
    #     c_e = combined_equality_constraints(W, forbid_edge_pairs, forbid_path_pairs, forbid_trek_pairs, coefficient=None)
    #     i_value = combined_inequality_constraints(W, w_threshold, exist_edge_pairs, exist_path_pairs, exist_trek_pairs, coefficient=None)
    #     tau_e = tau[:c_e.size]
    #     tau_i = tau[c_e.size:]
    #     residual = epsilon - i_value
    #     c_i = softplus(residual, sharpness)
    #     c = np.concatenate((c_e.reshape(-1), c_i.reshape(-1)))
    #     assert tau.shape == c.shape
    #     _, G_e = combined_equality_constraints(W, forbid_edge_pairs, forbid_path_pairs, forbid_trek_pairs, coefficient=tau_e)
    #     coefficient = -tau_i * sigmoid(sharpness * residual)
    #     _, G_i = combined_inequality_constraints(W, w_threshold, exist_edge_pairs, exist_path_pairs, exist_trek_pairs, coefficient=coefficient)
    #     G_prior = G_e + G_i
    #     p0, G_p0 = _p0(W)
    #     obj = loss + p0 + tau @ c + 0.5 * rho * h * h + alpha * h
    #     G_smooth = G_loss.reshape(-1) + G_p0 + G_prior + (rho * h + alpha) * G_h.reshape(-1)
    #     return obj, G_smooth
    
    # def f(w):
    #     residuals, _ = _func(w)
    #     return residuals

    # def grad(w):
    #     _, jacobian = _func(w)
    #     return jacobian

    # # Positive split parts put free coordinates inside their bounds. Diagonal
    # # entries remain zero as in optimization; the smooth extension is checked.
    # off_diagonal = ~np.eye(d, dtype=bool)
    # offset = 0.1 * off_diagonal
    # x0 = W.reshape(-1)

    # print("x0:", x0.shape)
    # print("f:", np.shape(f(x0)))
    # print("grad:", grad(x0).shape)
    # absolute_error = check_grad(f, grad, x0)
    # gradient_norm = np.linalg.norm(grad(x0))
    # relative_error = absolute_error / max(1.0, gradient_norm)

    # print("absolute error:", absolute_error)
    # print("relative error:", relative_error)
    # assert np.ndim(f(x0)) == 0
    # assert grad(x0).shape == x0.shape
    # assert relative_error < 1e-6, "Combined objective gradient check failed"

    # tests = [
    #     (
    #         "exist_edges",
    #         lambda W: _exist_edges(W, w_threshold, edge_pairs, coefficient=coefficient),
    #     ),
    #     (
    #         "exist_paths",
    #         lambda W: _exist_paths(W, w_threshold, path_pairs, coefficient=coefficient),
    #     ),
    #     (
    #         "exist_trek",
    #         lambda W: _exist_trek(W, w_threshold, trek_pairs, coefficient=coefficient),
    #     ),
    #     (
    #         "forbid_edges",
    #         lambda W: _forbid_edges(W, edge_pairs, coefficient=coefficient),
    #     ),
    #     (
    #         "forbid_paths",
    #         lambda W: _forbid_paths(W, path_pairs, coefficient=coefficient),
    #     ),
    #     (
    #         "forbid_trek",
    #         lambda W: _forbid_trek(W, trek_pairs, coefficient=coefficient),
    #     ),
    # ]

    # x0 = W.reshape(-1)

    # for name, constraint_function in tests:
    #     print("\nTesting:", name)
    #     def f(w):
    #         values, _ = constraint_function(w.reshape(d, d))
    #         print("value shape:", values.shape)
    #         return coefficient @ values

    #     def grad(w):
    #         _, weighted_gradient = constraint_function(w.reshape(d, d))
    #         print("gradient shape:", weighted_gradient.shape)
    #         return weighted_gradient.reshape(-1)

    #     error = check_grad(f, grad, x0)
    #     relative_error = error / max(1.0, np.linalg.norm(grad(x0)))

    #     print(f"\nTesting: {name}")
    #     print(f"absolute error: {error:.3e}")
    #     print(f"relative error: {relative_error:.3e}")
