"""Prior NOTEARS losses with fixed-potential DAG-NoCurl parameterization."""

from functools import partial

import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit
import torch


def softplus(x, sharpness=50.0):
    return np.logaddexp(0.0, sharpness * x) / sharpness


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
    dA_dX = expit(sharpness * X)
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
        grad_p[i, j] -= b[i, j] * expit(sharpness * X[i, j])

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
    dA_dX = expit(sharpness * X)
    grad_W = grad_A * dA_dX * (2.0 * W)

    return values, grad_W.reshape(-1)


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


def potential_from_connectivity(connectivity):
    """Project binary reachability onto potentials as in DAG-NoCurl's BPR.py."""
    connectivity = np.asarray(connectivity)
    if (
        connectivity.ndim != 2
        or connectivity.shape[0] == 0
        or connectivity.shape[0] != connectivity.shape[1]
    ):
        raise ValueError("connectivity must be a nonempty square matrix")
    if not np.all((connectivity == 0) | (connectivity == 1)):
        raise ValueError("connectivity entries must be 0 or 1")

    d = connectivity.shape[0]
    preX = connectivity.astype(float)
    asymX = preX - preX.transpose()

    preM = np.ones([d - 1, d - 1])
    for i in range(d - 1):
        preM[i, i] = -(d - 1)
    preb = np.sum(asymX, axis=1)

    prephi = np.linalg.solve(preM, preb[0:d - 1])
    phi = np.zeros(d)
    phi[0:d - 1] = prephi
    return phi


def _joint_adjacency(parameters, pairs, d, gradient=None, lambda1=0.0):
    """Evaluate the NoCurl graph, or pull back a gradient through its bounds."""
    pair_count = len(pairs[0])
    positive, negative = np.split(parameters[:2 * pair_count], 2)
    phi = np.append(parameters[2 * pair_count:], 0.0)
    i, j = pairs
    gaps = phi[j] - phi[i]
    forward = gaps > 0
    source = np.where(forward, i, j)
    target = np.where(forward, j, i)
    scale = np.abs(gaps)
    # Cap the effective split weights, as in the fixed-phi subproblem.
    raw_positive, raw_negative = positive * scale, negative * scale
    pos = np.minimum(raw_positive, 5.0)
    neg = np.minimum(raw_negative, 5.0)
    if gradient is None:
        W = np.zeros((d, d))
        W[source, target] = pos - neg
        return W, pos.sum() + neg.sum()

    edge_gradient = gradient[source, target]
    pos_gradient = (edge_gradient + lambda1) * (raw_positive <= 5.0)
    neg_gradient = (lambda1 - edge_gradient) * (raw_negative <= 5.0)
    gap_gradient = (pos_gradient * positive + neg_gradient * negative) * np.sign(gaps)
    phi_gradient = np.zeros(d)
    np.add.at(phi_gradient, i, -gap_gradient)
    np.add.at(phi_gradient, j, gap_gradient)
    return np.concatenate((pos_gradient * scale, neg_gradient * scale, phi_gradient[:-1]))


def notears_linear(
    X, reachability_matrix, lambda1, loss_type, tau, prior_knowledge=None,
    w_threshold=0.3, sharpness=50.0, epsilon=1e-1, compare=False,
    adam_lr=1e-3, adam_max_steps=10000, inner_param_tol=1e-5,
    op="soft", max_iter=100, h_tol=1e-8, rho_max=1e16,
    *, return_potentials=False, violation_budget=0.0,
):
    """Fit a DAG and return (thresholded W, convergence flag).

    reachability_matrix[i, j] is 1 if node i can reach node j by a directed
    path, with rows and columns in X's column order. Its DAG-NoCurl
    projection initializes phi. Hard mode unfreezes phi at the first
    proposed penalty increase. Soft mode holds phi fixed.
    The prior penalties use the same zero-based node pairs as linear.py.
    Soft mode weights them by tau. Hard mode uses an augmented Lagrangian
    for the sum of nonnegative prior violations and checks the pruned graph.
    If return_potentials is true, append the initial and final phi to the result.
    violation_budget is the fraction of path-existence priors hard mode
    may leave unsatisfied when compare=False.
    """
    X = np.asarray(X, dtype=float)
    phi = potential_from_connectivity(reachability_matrix)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != phi.size:
        raise ValueError("X must have samples and one column per node in reachability_matrix")
    if loss_type not in ("l2", "likelihood", "logistic", "poisson"):
        raise ValueError("unknown loss type")
    if not np.isfinite(lambda1) or lambda1 < 0:
        raise ValueError("lambda1 must be finite and nonnegative")
    if op not in ("soft", "hard"):
        raise ValueError("op must be 'soft' or 'hard'")
    if not 0 <= violation_budget <= 1:
        raise ValueError("violation_budget must be between 0 and 1")
    if op == "hard" and compare and violation_budget:
        raise ValueError("A positive violation_budget requires compare=False")
    if op == "hard" and (
        max_iter < 1 or not np.isfinite(h_tol) or h_tol < 0
        or not np.isfinite(rho_max) or rho_max < 1
    ):
        raise ValueError("Invalid augmented Lagrangian iteration limit, tolerance, or rho limit")
    if adam_lr <= 0 or adam_max_steps < 1 or inner_param_tol < 0:
        raise ValueError("Invalid Adam learning rate, step budget, or tolerance")
    if loss_type in ("l2", "likelihood"):
        X = X - X.mean(axis=0, keepdims=True)

    # W = S * ReLU(phi[j] - phi[i]) for a symmetric S. Each active
    # direction represents one unordered pair. Tied pairs are always zero.
    gaps = phi[None, :] - phi[:, None]
    active = np.where(gaps > 0)
    scales = gaps[active]
    prior_knowledge = {} if prior_knowledge is None else prior_knowledge
    equality_terms = [
        (_forbid_edges, prior_knowledge.get("forbid_edge_pairs", [])),
        (_forbid_paths, prior_knowledge.get("forbid_path_pairs", [])),
        (_forbid_trek, prior_knowledge.get("forbid_trek_pairs", [])),
    ]
    exist_paths = prior_knowledge.get("exist_path_pairs", [])
    path_budget = int(violation_budget * len(exist_paths))
    inequality_terms = [
        (_exist_edges, prior_knowledge.get("exist_edge_pairs", [])),
        (partial(_exist_trek, sharpness=sharpness),
         prior_knowledge.get("exist_trek_pairs", [])),
    ]
    path_constraint = partial(_exist_paths, sharpness=sharpness)
    if not compare:
        inequality_terms.append((path_constraint, exist_paths))

    def adjacency(weights):
        W = np.zeros_like(gaps)
        W[active] = weights * scales
        return W

    def data_loss(W):
        if loss_type in ("l2", "likelihood"):
            value, gradient = _loss(W, X, loss_type)
        else:
            prediction = X @ W
            if loss_type == "logistic":
                value = np.mean(np.sum(np.logaddexp(0, prediction) - X * prediction, axis=1))
                residual = expit(prediction) - X
            else:
                mean = np.exp(prediction)
                value = np.mean(np.sum(mean - X * prediction, axis=1))
                residual = mean - X
            gradient = X.T @ residual / X.shape[0]

        return value, gradient.reshape(W.shape)

    def prior_violation(W, hard=False):
        value = 0.0
        gradient = np.zeros_like(W)
        for penalty, pairs in equality_terms:
            if len(pairs):
                values, derivative = penalty(W, pairs, coefficient=np.ones(len(pairs)))
                value += values.sum()
                gradient += derivative.reshape(W.shape)

        if compare and len(exist_paths):
            penalty, derivative = _exist_paths2(
                W, w_threshold, exist_paths, sharpness=sharpness, epsilon=10,
            )
            value += penalty
            gradient += derivative.reshape(W.shape)

        for constraint, pairs in inequality_terms:
            if len(pairs):
                residual = epsilon - constraint(W, w_threshold, pairs)
                if hard:
                    positive = np.maximum(residual, 0.0)
                    if constraint is path_constraint and path_budget:
                        # Exempt the largest path shortfalls at the current W.
                        exempt = np.argsort(-positive, kind="stable")[:path_budget]
                        positive[exempt] = 0.0
                    value += positive @ positive
                    coefficient = -2.0 * positive
                else:
                    value += softplus(residual, sharpness).sum()
                    coefficient = -expit(sharpness * residual)
                _, derivative = constraint(W, w_threshold, pairs, coefficient=coefficient)
                gradient += derivative.reshape(W.shape)
        return value, gradient

    def objective(weights):
        W = adjacency(weights)
        value, gradient = data_loss(W)
        if tau != 0:
            violation, derivative = prior_violation(W)
            value += tau * violation
            gradient += tau * derivative
        value += lambda1 * np.abs(W).sum()
        return value, gradient[active] * scales

    has_priors = any(len(pairs) for pairs in prior_knowledge.values())
    if op == "hard" and has_priors:
        from prior_notears.utils import evaluate_prior_knowledge

        # Split signed weights so L1 is linear in the bounded inner problem.
        l1_weight = np.tile(lambda1 * scales, 2)
        bounds = [(0, limit) for limit in np.tile(5.0 / scales, 2)]
        split_weights = np.zeros(2 * scales.size)
        rho, alpha, previous_h = 1.0, 0.0, np.inf
        joint_parameters = None
        pairs = np.triu_indices(phi.size, k=1)
        joint_bounds = [(0, None)] * (2 * len(pairs[0])) + [(None, None)] * (phi.size - 1)

        def hard_objective(split):
            positive, negative = np.split(split, 2)
            W = adjacency(positive - negative)
            value, gradient = data_loss(W)
            h, derivative = prior_violation(W, hard=True)
            value += alpha * h + 0.5 * rho * h * h + l1_weight @ split
            gradient += (alpha + rho * h) * derivative
            active_gradient = gradient[active] * scales
            return value, np.concatenate((active_gradient, -active_gradient)) + l1_weight

        def joint_objective(parameters):
            W, l1_norm = _joint_adjacency(parameters, pairs, phi.size)
            value, gradient = data_loss(W)
            h, derivative = prior_violation(W, hard=True)
            value += alpha * h + 0.5 * rho * h * h + lambda1 * l1_norm
            gradient += (alpha + rho * h) * derivative
            return value, _joint_adjacency(parameters, pairs, phi.size, gradient, lambda1)

        h, _ = prior_violation(adjacency(np.zeros(scales.size)), hard=True)
        converged = not scales.size and h <= h_tol
        if scales.size:
            for _ in range(max_iter):
                while True:
                    if joint_parameters is None:
                        sol = sopt.minimize(
                            hard_objective, split_weights, method="L-BFGS-B",
                            jac=True, bounds=bounds,
                            options={"ftol": 1e-12, "gtol": 1e-8},
                        )
                        positive, negative = np.split(sol.x, 2)
                        W = adjacency(positive - negative)
                    else:
                        sol = sopt.minimize(
                            joint_objective, joint_parameters, method="L-BFGS-B",
                            jac=True, bounds=joint_bounds,
                            options={"ftol": 1e-12, "gtol": 1e-8},
                        )
                        joint_parameters = sol.x
                        W, _ = _joint_adjacency(joint_parameters, pairs, phi.size)
                    h, _ = prior_violation(W, hard=True)
                    if h <= h_tol or h <= 0.25 * previous_h or rho >= rho_max:
                        break
                    if joint_parameters is None:
                        # Keep rho and alpha and warm-start from this fixed-phi solve.
                        split_by_pair = []
                        for weights in (positive, negative):
                            directed = np.zeros_like(gaps)
                            directed[active] = weights
                            split_by_pair.append((directed + directed.T)[pairs])
                        joint_parameters = np.concatenate((*split_by_pair, phi[:-1]))
                        print(f"Unfreezing phi before penalty increase (rho={rho:g}, alpha={alpha:g})",
                              flush=True)
                        continue
                    rho = min(10 * rho, rho_max)
                if joint_parameters is None:
                    split_weights = sol.x
                previous_h = h
                alpha += rho * h
                if h <= h_tol or rho >= rho_max:
                    converged = bool(sol.success and h <= h_tol)
                    break

        if joint_parameters is None:
            positive, negative = np.split(split_weights, 2)
            W = adjacency(positive - negative)
        else:
            W, _ = _joint_adjacency(joint_parameters, pairs, phi.size)
        W[np.abs(W) < w_threshold] = 0
        satisfied, _ = evaluate_prior_knowledge(W, prior_knowledge)
        total_priors = sum(len(pairs) for pairs in prior_knowledge.values())
        success = converged and satisfied == total_priors
        if converged and path_budget and not success:
            path_satisfied, _ = evaluate_prior_knowledge(W, {"exist_path_pairs": exist_paths})
            missing_paths = len(exist_paths) - path_satisfied
            success = missing_paths <= path_budget and total_priors - satisfied == missing_paths
        if return_potentials:
            final_phi = phi.copy()
            if joint_parameters is not None:
                final_phi[:-1] = joint_parameters[-(phi.size - 1):]
            return W, success, phi, final_phi
        return W, success

    weights = np.zeros(scales.size)
    converged = not weights.size
    if weights.size:
        parameter = torch.nn.Parameter(torch.from_numpy(weights.copy()))
        optimizer = torch.optim.Adam([parameter], lr=adam_lr)
        # Preserve the original bound on the effective regression coefficients.
        upper = torch.from_numpy(5.0 / scales)
        l1_weight = torch.from_numpy(lambda1 * scales)
        beta2 = optimizer.param_groups[0]["betas"][1]
        adam_eps = optimizer.param_groups[0]["eps"]
        checkpoint = weights
        stable_checks = 0
        for step in range(1, adam_max_steps + 1):
            value, gradient = objective(parameter.detach().numpy())
            if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                raise FloatingPointError("Non-finite Adam objective or gradient")
            optimizer.zero_grad(set_to_none=True)
            parameter.grad = torch.from_numpy(gradient)
            optimizer.step()
            with torch.no_grad():
                # Apply the L1 proximal step in Adam's diagonal metric.
                second_moment = optimizer.state[parameter]["exp_avg_sq"]
                denominator = (second_moment / (1 - beta2 ** step)).sqrt() + adam_eps
                shrinkage = adam_lr * l1_weight / denominator
                magnitude = (parameter.abs() - shrinkage).clamp_min(0).minimum(upper)
                parameter.copy_(parameter.sign() * magnitude)
            if step % 100 == 0:
                current = parameter.detach().numpy().copy()
                change = np.linalg.norm(current - checkpoint) / max(1.0, np.linalg.norm(checkpoint))
                stable_checks = stable_checks + 1 if change < inner_param_tol else 0
                checkpoint = current
                if stable_checks >= 3:
                    converged = True
                    break
        weights = parameter.detach().numpy()

    value, gradient = objective(weights)
    if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError("Non-finite final objective or gradient")
    W = adjacency(weights)
    W[np.abs(W) < w_threshold] = 0
    if return_potentials:
        return W, converged, phi, phi.copy()
    return W, converged


def _h(W):
    """Evaluate value and gradient of acyclicity constraint."""
    d = W.shape[0]
    E = slin.expm(W * W)  # (Zheng et al. 2018)
    h = np.trace(E) - d
    G_h = E.T * W * 2
    return h, G_h


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


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    import time

    import networkx as nx
    from sklearn.preprocessing import StandardScaler

    from Varsortability.src.varsortability import varsortability
    from prior_notears import linear_hard, utils

    parser = argparse.ArgumentParser(prog='linear NoCurl with prior knowledge',)

    parser.add_argument('-s', '--seed', dest='s',  default=0, type=int)
    parser.add_argument('-d', '--num_nodes', dest='d', default=10, type=int)
    parser.add_argument('-e', '--num_edges_per_node', dest='e', default=1, type=int)
    parser.add_argument('-g', '--graph_type', dest='g', default="ER", type=str)
    parser.add_argument('-n', '--noise', dest='n', default="gauss", type=str)
    parser.add_argument('-p', '--prior_type', dest='p', default="mix", type=str)
    parser.add_argument('-r', '--prior_rate', dest='r', default=0.25, type=float)
    parser.add_argument('-t', '--w_threshold', dest='t', default=0.3, type=float)
    parser.add_argument('-ep', '--epsilon', dest='ep', default=1e-1, type=float)
    parser.add_argument('-c', '--compare', dest='c', action='store_true', default=False)
    parser.add_argument('-tau', '--tau', dest='tau', default=1.0, type=float)
    parser.add_argument("--lambda1", type=float, default=0.1,
                        help="L1 penalty for the NoCurl fits")
    parser.add_argument("--violation-budget", type=float, default=0.0,
                        help="Fraction of path-existence priors hard mode may leave unsatisfied")
    args = parser.parse_args()
    if not np.isfinite(args.lambda1) or args.lambda1 < 0:
        parser.error("--lambda1 must be finite and nonnegative")
    if not 0 <= args.violation_budget <= 1:
        parser.error("--violation-budget must be between 0 and 1")
    if args.c and args.violation_budget:
        parser.error("--violation-budget requires compare=False")

    utils.set_random_seed(args.s)
    n, d, s0, graph_type, sem_type = 2000, args.d, args.e*args.d, args.g, args.n
    B_true = utils.simulate_dag(d, s0, graph_type)
    print(f"Run: seed={args.s} d={d} graph={graph_type} edges={s0} tau={args.tau} lambda1={args.lambda1}", flush=True)
    W_true = utils.simulate_parameter(B_true)
    if not args.c:
        filename = f"linear_nocurl_init_from_linear_hard_{args.p}_{graph_type}{args.e}_d{d}_{sem_type}_rate{args.r}_epsilon{args.ep}_seed{args.s}_tau{args.tau}_soft_hard_lambda1{args.lambda1}_budget{args.violation_budget}_joint.json"
    else:
        filename = f"linear_nocurl_init_from_linear_hard_{args.p}_{graph_type}{args.e}_d{d}_{sem_type}_rate{args.r}_epsilon10_seed{args.s}_tau{args.tau}_compare_soft_hard_lambda1{args.lambda1}_budget{args.violation_budget}_joint.json"

    X = utils.simulate_linear_sem(W_true, n, sem_type)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    varsortability_score = varsortability(X_std, W_true)
    correct_priors = utils.generate_prior_knowledge(
        B_true, prior_rate=args.r, prior_type=args.p, random_state=args.s,
    )
    print("Fully perfect prior counts:",
          {key: len(pairs) for key, pairs in correct_priors.items()}, flush=True)

    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / f"linear_nocurl_{args.p}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    initialization_epsilon = 0.1
    evaluation_status = {}
    results = {
        "B_true": B_true.tolist(),
        "W_true": W_true.tolist(),
        "X": X.tolist(),
        "X_std": X_std.tolist(),
        "varsortability_score": varsortability_score,
        "prior_knowledge_perfect_l2": correct_priors,
        "evaluation_status": evaluation_status,
        "linear_hard_initialization": {},
        "initialization_epsilon": initialization_epsilon,
        "lambda1": args.lambda1,
        "violation_budget": args.violation_budget,
        "allowed_path_violations": int(args.violation_budget * len(correct_priors.get("exist_path_pairs", []))),
    }

    reachability_by_init = {}
    for init_label, priors in (("no_prior", None), ("with_prior", correct_priors)):
        print(f">>> Linear hard initialization [{init_label}] <<<", flush=True)
        start_time = time.perf_counter()
        W_init, success = linear_hard.notears_linear(
            X_std, lambda1=0.1, loss_type="l2", prior_knowledge=priors,
            w_threshold=args.t, epsilon=initialization_epsilon, compare=args.c,
        )
        elapsed = time.perf_counter() - start_time
        values = linear_hard.evaluate_prior_values(W_init, correct_priors, args.t, args.c)
        equality = {"acyclicity": float(_h(W_init)[0])}
        inequality = {}
        for key, scores in values.items():
            if not scores:
                continue
            if key.startswith("forbid_"):
                equality[key] = float(np.mean(scores))
            elif args.c and key == "exist_path_pairs":
                equality[key] = float(sum(scores))
            else:
                inequality[key] = [initialization_epsilon - score for score in scores]

        graph = nx.from_numpy_array(W_init != 0, create_using=nx.DiGraph)
        # The general closure also handles a cyclic estimate from linear_hard.
        closure = nx.transitive_closure(graph, reflexive=None)
        reachability_matrix = nx.to_numpy_array(closure, nodelist=range(d), dtype=int)
        reachability_by_init[init_label] = reachability_matrix
        results["linear_hard_initialization"][init_label] = {
            "W_est": W_init.tolist(),
            "sol_success": bool(success),
            "running_time": elapsed,
            "is_dag": nx.is_directed_acyclic_graph(graph),
            "constraint_values": values,
            "equality_constraints": equality,
            "inequality_constraints": inequality,
            "reachability_matrix": reachability_matrix.tolist(),
        }

    with output_path.open("w") as file:
        json.dump(results, file, indent=4)

    conditions = (
        ("no_prior_l2", "soft", None, 0.0, "no_prior"),
        ("prior_perfect_l2_soft", "soft", correct_priors, args.tau, "with_prior"),
        ("prior_perfect_l2_hard", "hard", correct_priors, args.tau, "with_prior"),
    )
    for label, op, priors, tau, init_label in conditions:
        print(f">>> Evaluation [{label}] <<<", flush=True)
        start_time = time.perf_counter()
        W_est, success, p_init, p_final = notears_linear(
            X_std, reachability_matrix=reachability_by_init[init_label],
            lambda1=args.lambda1, loss_type="l2", tau=tau,
            prior_knowledge=priors, w_threshold=args.t, epsilon=args.ep,
            compare=args.c, op=op, return_potentials=True,
            violation_budget=args.violation_budget,
        )
        results[f"initialization_source_{label}"] = init_label
        results[f"running_time_{label}"] = time.perf_counter() - start_time
        results[f"W_est_{label}"] = W_est.tolist()
        results[f"p_init_{label}"] = p_init.tolist()
        results[f"p_final_{label}"] = p_final.tolist()
        results[f"sol_success_{label}"] = bool(success)
        results[f"h_est_{label}"] = float(_h(W_est)[0])
        results[f"loss_{label}"] = float(_loss(W_est, X_std, "l2")[0])
        results[f"constraint_values_{label}"] = evaluate_prior_values(
            W_est, correct_priors, args.t, args.c,
        )
        satisfied, percentage = utils.evaluate_prior_knowledge(W_est, correct_priors)
        results[f"satisfied_{label}"] = satisfied
        results[f"satisfied_percentage_{label}"] = percentage
        try:
            if not utils.is_dag(W_est):
                raise ValueError("Estimated graph contains a directed cycle")
            results[f"acc_{label}"] = utils.count_accuracy(B_true, W_est != 0)
        except Exception as exc:
            evaluation_status[label] = {
                "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
            }
            print(f"Evaluation failed: {exc}")
        else:
            evaluation_status[label] = {"status": "completed"}

    with output_path.open("w") as file:
        json.dump(results, file, indent=4)

    for label, *_ in conditions:
        metrics = results.get(f"acc_{label}")
        if isinstance(metrics, dict):
            satisfaction = results.get(f"satisfied_percentage_{label}", float("nan"))
            print(
                f"Result [{label}]: F1={metrics['f1']:.4f} "
                f"SHD={metrics['shd']} satisfaction={satisfaction:.1f}%",
                flush=True,
            )
    print(f"Results saved to: {output_path}")

