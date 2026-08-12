import argparse
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

####just for test, to delete later #######################################
def _forbid_edges(W, edge_pairs):
    """forbid edges from the list of index pairs.
    
    Args:
        W (np.ndarray): [d, d] weight matrix
        pairs (list): List of (i, j) edge pairs

    Returns:
        float: Values Sum of W[i, j] for each (i, j) pair
    """
    e = np.sum([(W * W)[i, j] for i, j in edge_pairs]) / len(edge_pairs)
    G_e = np.zeros_like(W)
    for i, j in edge_pairs:
        G_e[i, j] += 2.0 * W[i, j]
    G_e = G_e / len(edge_pairs)
    return e, G_e.reshape(-1)

def _exist_edges(W, w_thres, edge_pairs):
    """Return a vector of edge residuals for the given index pairs.
    
    Args:
        W (np.ndarray): [d, d] weight matrix
        w_thres (float): threshold for edge existence
        pairs (list): List of (i, j) edge pairs

    Returns:
        np.ndarray (1D): Vector of W[i, j] - w_thres for each (i, j) pair
    """
    residuals = np.array([
        W[i, j]**2 - w_thres**2
        for i, j in edge_pairs
    ])

    jacobian = np.zeros((len(edge_pairs), W.size))

    for k, (i, j) in enumerate(edge_pairs):
        jacobian[k, i * d + j] = 2.0 * W[i, j]
    return residuals, jacobian

def _forbid_paths(W, path_pairs):

    """Compute the forbidden-path penalty and its gradient.
    Args:
        W: Array of shape (d, d).
        path_pairs: Sequence of (start, end) index pairs.
    Returns:
        value: Scalar penalty.
        gradient: Flattened gradient with respect to W.
    """

    if len(path_pairs) == 0:
        raise ValueError("path_pairs must not be empty")
    W = np.asarray(W, dtype=float)
    scale = len(path_pairs)
    A = W * W
    E = slin.expm(A)
    M = np.zeros_like(W, dtype=float)
    for i, j in path_pairs:
        M[i, j] += 1.0
    value = np.sum(M * E) / scale

    # Adjoint of the matrix-exponential Fréchet derivative
    grad_A = slin.expm_frechet(
        A.T,
        M,
        compute_expm=False,
    ) / scale

    # A = W ⊙ W, so dA/dW = 2W
    grad_W = 2.0 * W * grad_A
    return value, grad_W.reshape(-1)

def _exist_paths(W, w_thres, path_pairs, sharpness=10.0):
    X = W * W - w_thres * w_thres
    A = softplus(X, sharpness)
    E = slin.expm(A)

    residuals = np.array([E[i, j] for i, j in path_pairs])

    dA_dX = sigmoid(sharpness * X)          # derivative of softplus
    dX_dW = 2.0 * W                    # derivative of W^2
    J = np.zeros((len(path_pairs), W.size), dtype=float)

    for k, (i, j) in enumerate(path_pairs):
        M = np.zeros_like(W, dtype=np.float64)
        M[i, j] = 1.0
        grad_A = slin.expm_frechet(A.T, M, compute_expm=False)
        J[k, :] = (dX_dW * dA_dX * grad_A).reshape(-1)

    return residuals, J

def _forbid_trek(W, trek_pairs):
    """Return the mean forbidden-trek penalty and its gradient."""
    E = slin.expm(W * W)
    scale = len(trek_pairs)

    M = np.zeros_like(W, dtype=np.result_type(W, np.float64))
    for i, j in trek_pairs:
        M[i, j] += 1.0

    value = np.sum(M * (E.T @ E)) / scale

    # Gradient with respect to E.
    grad_E = E @ (M + M.T) / scale

    # Adjoint of the matrix-exponential derivative.
    grad_A = slin.expm_frechet(
        (W * W).T,
        grad_E,
        compute_expm=False,
    )

    grad_W = 2.0 * W * grad_A
    return value, grad_W.reshape(-1)

def _exist_trek(W, w_thres, trek_pairs, sharpness=10.0):

    """Return trek residuals and their Jacobian.

    Args:
        W: Weight matrix with shape (d, d).
        w_thres: Threshold for trek existence.
        trek_pairs: Sequence of endpoint pairs (i, j).
        sharpness: Softplus sharpness parameter.

    Returns:
        residuals: Array with shape (len(trek_pairs),).
        J: Jacobian with shape (len(trek_pairs), W.size).
    """

    W = np.asarray(W, dtype=float)
    X = W * W - w_thres * w_thres
    A = softplus(X, sharpness)
    E = slin.expm(A)
    T = E.T @ E

    dA_dX = sigmoid(sharpness * X)
    residuals = np.empty(len(trek_pairs), dtype=float)
    J = np.empty((len(trek_pairs), W.size), dtype=float)
    for k, (i, j) in enumerate(trek_pairs):
        residuals[k] = T[i, j]
        # Gradient of T[i, j] with respect to E
        grad_E = np.zeros_like(E, dtype=float)
        if i == j:
            grad_E[:, i] = 2.0 * E[:, i]
        else:
            grad_E[:, i] = E[:, j]
            grad_E[:, j] = E[:, i]
        # Adjoint derivative through E = expm(A)
        grad_A = slin.expm_frechet(
            A.T,
            grad_E,
            compute_expm=False,
        )

        # Elementwise chain:
        # A = softplus_beta(X)
        # X = W**2 - w_thres**2
        grad_W = grad_A * dA_dX * (2.0 * W)
        J[k, :] = grad_W.reshape(-1)

    return residuals, J

def combined_equality_constraints(w, forbid_edge_pairs, forbid_path_pairs, forbid_trek_pairs):
    """Combine equality constraints and their Jacobian."""
    # h_value, h_grad = _h(w)
    edge_value, edge_grad = _forbid_edges(w, forbid_edge_pairs)
    path_value, path_grad = _forbid_paths(w, forbid_path_pairs)
    trek_value, trek_grad = _forbid_trek(w, forbid_trek_pairs)

    values = np.array([
        # h_value,
        edge_value,
        path_value,
        trek_value,
    ])

    jacobian = np.vstack([
        # np.asarray(h_grad).reshape(-1),
        np.asarray(edge_grad),
        np.asarray(path_grad),
        np.asarray(trek_grad),
    ])

    return values, jacobian

def combined_inequality_constraints(w, w_threshold, exist_edge_pairs, exist_path_pairs, exist_trek_pairs):
    """Combine inequality constraints and their Jacobian."""
    edge_value, edge_grad = _exist_edges(w, w_threshold, exist_edge_pairs)
    path_value, path_grad = _exist_paths(w, w_threshold, exist_path_pairs)
    trek_value, trek_grad = _exist_trek(w, w_threshold, exist_trek_pairs)

    values = np.concatenate([
        edge_value,
        path_value,
        trek_value,
    ])

    jacobian = np.vstack([
        np.asarray(edge_grad),
        np.asarray(path_grad),
        np.asarray(trek_grad),
    ])

    return values, jacobian


####just for test, to delete later #######################################

def softplus(x, sharpness=10.0):
    return np.logaddexp(0.0, sharpness * x) / sharpness

def notears_linear(X, lambda1, loss_type, prior_knowledge=None, max_iter=100, violation_tol=1e-8, rho_max=1e+16, w_threshold=0.3, sharpness=10.0, epsilon=1e-1):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        prior_knowledge (dict): prior knowledge
        max_iter (int): max num of dual ascent steps
        violation_tol (float): exit if |violation(w_est)| <= violation_tol
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

    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
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

    def _forbid_edges(W, edge_pairs):
        """forbid edges from the list of index pairs.
        
        Args:
            W (np.ndarray): [d, d] weight matrix
            pairs (list): List of (i, j) edge pairs

        Returns:
            float: Values Sum of W[i, j] for each (i, j) pair
        """
        e = np.sum([(W * W)[i, j] for i, j in edge_pairs]) / len(edge_pairs)
        G_e = np.zeros_like(W)
        for i, j in edge_pairs:
            G_e[i, j] += 2.0 * W[i, j]
        G_e = G_e / len(edge_pairs)
        return e, G_e.reshape(-1)

    def _exist_edges(W, w_thres, edge_pairs):
        """Return a vector of edge residuals for the given index pairs.
        
        Args:
            W (np.ndarray): [d, d] weight matrix
            w_thres (float): threshold for edge existence
            pairs (list): List of (i, j) edge pairs

        Returns:
            np.ndarray (1D): Vector of W[i, j] - w_thres for each (i, j) pair
        """
        residuals = np.array([
            W[i, j]**2 - w_thres**2
            for i, j in edge_pairs
        ])

        jacobian = np.zeros((len(edge_pairs), W.size))

        for k, (i, j) in enumerate(edge_pairs):
            jacobian[k, i * d + j] = 2.0 * W[i, j]
        return residuals, jacobian
    
    def _forbid_paths(W, path_pairs):

        """Compute the forbidden-path penalty and its gradient.
        Args:
            W: Array of shape (d, d).
            path_pairs: Sequence of (start, end) index pairs.
        Returns:
            value: Scalar penalty.
            gradient: Flattened gradient with respect to W.
        """

        if len(path_pairs) == 0:
            raise ValueError("path_pairs must not be empty")
        W = np.asarray(W, dtype=float)
        scale = len(path_pairs)
        A = W * W
        E = slin.expm(A)
        M = np.zeros_like(W, dtype=float)
        for i, j in path_pairs:
            M[i, j] += 1.0
        value = np.sum(M * E) / scale

        # Adjoint of the matrix-exponential Fréchet derivative
        grad_A = slin.expm_frechet(
            A.T,
            M,
            compute_expm=False,
        ) / scale

        # A = W ⊙ W, so dA/dW = 2W
        grad_W = 2.0 * W * grad_A
        return value, grad_W.reshape(-1)

    def _exist_paths(W, w_thres, path_pairs):
        X = W * W - w_thres * w_thres
        A = softplus(X, sharpness)
        E = slin.expm(A)

        residuals = np.array([E[i, j] for i, j in path_pairs])

        dA_dX = sigmoid(sharpness * X)          # derivative of softplus
        dX_dW = 2.0 * W                    # derivative of W^2
        J = np.zeros((len(path_pairs), W.size), dtype=float)

        for k, (i, j) in enumerate(path_pairs):
            M = np.zeros_like(W, dtype=np.float64)
            M[i, j] = 1.0
            grad_A = slin.expm_frechet(A.T, M, compute_expm=False)
            J[k, :] = (dX_dW * dA_dX * grad_A).reshape(-1)

        return residuals, J
    
    def _forbid_trek(W, trek_pairs):
        """Return the mean forbidden-trek penalty and its gradient."""
        E = slin.expm(W * W)
        scale = len(trek_pairs)

        M = np.zeros_like(W, dtype=np.result_type(W, np.float64))
        for i, j in trek_pairs:
            M[i, j] += 1.0

        value = np.sum(M * (E.T @ E)) / scale

        # Gradient with respect to E.
        grad_E = E @ (M + M.T) / scale

        # Adjoint of the matrix-exponential derivative.
        grad_A = slin.expm_frechet(
            (W * W).T,
            grad_E,
            compute_expm=False,
        )

        grad_W = 2.0 * W * grad_A
        return value, grad_W.reshape(-1)
    
    def _exist_trek(W, w_thres, trek_pairs):

        """Return trek residuals and their Jacobian.

        Args:
            W: Weight matrix with shape (d, d).
            w_thres: Threshold for trek existence.
            trek_pairs: Sequence of endpoint pairs (i, j).
            sharpness: Softplus sharpness parameter.

        Returns:
            residuals: Array with shape (len(trek_pairs),).
            J: Jacobian with shape (len(trek_pairs), W.size).
        """

        W = np.asarray(W, dtype=float)
        X = W * W - w_thres * w_thres
        A = softplus(X, sharpness)
        E = slin.expm(A)
        T = E.T @ E

        dA_dX = sigmoid(sharpness * X)
        residuals = np.empty(len(trek_pairs), dtype=float)
        J = np.empty((len(trek_pairs), W.size), dtype=float)
        for k, (i, j) in enumerate(trek_pairs):
            residuals[k] = T[i, j]
            # Gradient of T[i, j] with respect to E
            grad_E = np.zeros_like(E, dtype=float)
            if i == j:
                grad_E[:, i] = 2.0 * E[:, i]
            else:
                grad_E[:, i] = E[:, j]
                grad_E[:, j] = E[:, i]
            # Adjoint derivative through E = expm(A)
            grad_A = slin.expm_frechet(
                A.T,
                grad_E,
                compute_expm=False,
            )

            # Elementwise chain:
            # A = softplus_beta(X)
            # X = W**2 - w_thres**2
            grad_W = grad_A * dA_dX * (2.0 * W)
            J[k, :] = grad_W.reshape(-1)

        return residuals, J

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def combined_equality_constraints(W):
        """Combine active equality constraints and their Jacobians.

        Returns:
            values: Shape (m,), where m is the number of active constraints.
            jacobian: Shape (m, d*d).
        """
        values = []
        gradients = []

        h_value, h_grad = _h(W)
        values.append(h_value)
        gradients.append(np.asarray(h_grad).reshape(-1))

        if forbid_edge_pairs:
            edge_value, edge_grad = _forbid_edges(
                W, forbid_edge_pairs
            )
            values.append(edge_value)
            gradients.append(np.asarray(edge_grad).reshape(-1))

        if forbid_path_pairs:
            path_value, path_grad = _forbid_paths(
                W, forbid_path_pairs
            )
            values.append(path_value)
            gradients.append(np.asarray(path_grad).reshape(-1))

        if forbid_trek_pairs:
            trek_value, trek_grad = _forbid_trek(
                W, forbid_trek_pairs
            )
            values.append(trek_value)
            gradients.append(np.asarray(trek_grad).reshape(-1))

        values = np.asarray(values, dtype=float)
        jacobian = np.vstack(gradients)

        return values, jacobian

    def combined_inequality_constraints(W):
        """Combine active inequality constraints and their Jacobians.

        Returns:
            values: Shape (m,), with one value per pair.
            jacobian: Shape (m, d*d).
        """
        values = []
        jacobians = []

        if exist_edge_pairs:
            edge_values, edge_jacobian = _exist_edges(
                W,
                w_threshold,
                exist_edge_pairs,
            )
            values.append(np.atleast_1d(edge_values))
            jacobians.append(np.atleast_2d(edge_jacobian))

        if exist_path_pairs:
            path_values, path_jacobian = _exist_paths(
                W,
                w_threshold,
                exist_path_pairs,
            )
            values.append(np.atleast_1d(path_values))
            jacobians.append(np.atleast_2d(path_jacobian))

        if exist_trek_pairs:
            trek_values, trek_jacobian = _exist_trek(
                W,
                w_threshold,
                exist_trek_pairs,
            )
            values.append(np.atleast_1d(trek_values))
            jacobians.append(np.atleast_2d(trek_jacobian))

        if not values:
            return (
                np.empty(0, dtype=float),
                np.empty((0, W.size), dtype=float),
            )

        values = np.concatenate(values)
        jacobian = np.vstack(jacobians)

        return values, jacobian

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
        c_e, G_e = combined_equality_constraints(W)
        i_value, i_grad = combined_inequality_constraints(W)
        c_i = epsilon - i_value  
        G_i = -i_grad
        z = beta + rho * c_i
        # value = softplus(z, sharpness)
        # dvalue_dz = sigmoid(sharpness * z)
        positive_part = np.maximum(z, 0.0)
        obj = loss + 0.5 * rho * np.sum(c_e**2) + alpha @ c_e + ( 1 / (2 * rho) ) * (np.sum(positive_part**2) - np.sum(beta**2)) + lambda1 * w.sum()
        G_smooth = G_loss.reshape(-1) + G_e.T @ (alpha + rho * c_e) + G_i.T @ positive_part
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

        all_violations = np.concatenate([
            np.asarray(c_e).reshape(-1),
            active_i.reshape(-1),
        ])

        l2_violation = np.linalg.norm(
            all_violations,
            ord=2,
        )

        max_violation = np.linalg.norm(
            all_violations,
            ord=np.inf,
        )

        return l2_violation, max_violation
    
    n, d = X.shape
    w_est = np.zeros(2 * d * d)  # double w_est into (w_pos, w_neg)
    rho = 1.0
    equality_len = 1 + sum(
    bool(pairs)
    for pairs in (
        forbid_edge_pairs,
        forbid_path_pairs,
        forbid_trek_pairs,
        )
    )
    inequality_len = len(exist_edge_pairs) + len(exist_path_pairs) + len(exist_trek_pairs)
    alpha = np.zeros(equality_len, dtype=float)
    beta = np.zeros(inequality_len, dtype=float)
    l2_violation = np.inf

    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]

    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, c_e_new, c_i_new = None, None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)

            if not sol.success:
                print("L-BFGS-B warning:", sol.message)

            if not np.isfinite(sol.fun):
                raise FloatingPointError(
                    "The augmented objective became non-finite"
                )

            if not np.all(np.isfinite(sol.x)):
                raise FloatingPointError(
                    "The optimizer returned non-finite weights"
                )
            
            w_new = sol.x
            c_e_new, _ = combined_equality_constraints(_adj(w_new))
            i_value_new, _ = combined_inequality_constraints(_adj(w_new))
            c_i_new = epsilon - i_value_new  
            ###############################
            print("equality constraints:", c_e_new)
            print("inequality constraints:", c_i_new)
            ###############################
            # violation_new = _violation(c_e_new, c_i_new)
            l2_violation_new, max_violation_new = _violation(c_e_new, c_i_new,)
            if l2_violation_new > 0.25 * l2_violation:
                rho *= 10
            else:
                break
        w_est, l2_violation = w_new, l2_violation_new
        alpha += rho * c_e_new
        beta = np.maximum( beta + rho * c_i_new, 0.0)
        if max_violation_new <= violation_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est, bool(sol.success)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='linear NOTEARS with prior knowledge',)

    parser.add_argument('-s', '--seed', dest='s',  default=42, type=int)
    parser.add_argument('-d', '--num_nodes', dest='d', default=4, type=int)
    parser.add_argument('-e', '--num_edges', dest='e', default=2, type=int)
    parser.add_argument('-g', '--graph_type', dest='g', default="ER", type=str)
    parser.add_argument('-n', '--noise', dest='n', default="Gaussian", type=str)
    parser.add_argument('-p', '--prior_type', dest='p', default="mix", type=str)
    parser.add_argument('-r', '--prior_rate', dest='r', default=0.25, type=float)
    parser.add_argument('-t', '--w_threshold', dest='t', default=0.3, type=float)
    args = parser.parse_args()

    from prior_notears import utils
    utils.set_random_seed(args.s)
    n, d, s0, graph_type, sem_type = 10*args.d, args.d, args.e*args.d, args.g, args.n
    B_true = utils.simulate_dag(d, s0, graph_type)
    print("B_true:", B_true)
    W_true = utils.simulate_parameter(B_true)
    filename = f"linear_{args.p}_{graph_type}{args.e}_d{d}_{sem_type}_rate{args.r}_seed{args.s}.json"

    X = utils.simulate_linear_sem(W_true, n, sem_type)
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    prior_knowledge = utils.generate_prior_knowledge(
            B_true,
            prior_rate=args.r,
            prior_type=args.p,
        )
    print("prior_knowledge:", prior_knowledge)

    print('>>> Evaluation with prior knowledge <<<')
    W_est_prior, sol_success = notears_linear(X_standardized, lambda1=0.1, loss_type='l2', prior_knowledge=prior_knowledge, w_threshold=args.t)
    assert utils.is_dag(W_est_prior)
    print("W_est_prior:", W_est_prior)
    acc_prior = utils.count_accuracy(B_true, W_est_prior != 0)
    print(acc_prior)

    print('>>> Evaluation without prior knowledge <<<')
    W_est_no_prior = linear.notears_linear(X_standardized, lambda1=0.1, loss_type='l2', w_threshold=args.t)
    assert utils.is_dag(W_est_no_prior)
    print("W_est_no_prior:", W_est_no_prior)
    acc_no_prior = utils.count_accuracy(B_true, W_est_no_prior != 0)
    print(acc_no_prior)

    results = {
        "B_true": B_true.tolist(),
        "W_true": W_true.tolist(),
        "X": X.tolist(),
        "X_standardized": X_standardized.tolist(),
        "prior_knowledge": prior_knowledge,
        "W_est_prior": W_est_prior.tolist(),
        "W_est_no_prior": W_est_no_prior.tolist(),
        "acc_prior": acc_prior,
        "acc_no_prior": acc_no_prior,
        "sol_success": sol_success
    }

    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / f"linear_{args.p}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    with output_path.open("w") as file:
        json.dump(results, file, indent=4)

    print(f"Results saved to: {output_path}")
    ##### add constraints check from ground truth and constraints"
#### check gradient of prior knowledge constraints
# if __name__ == '__main__':
#     d = 4
#     W = np.array([
#         [1.0, 2.0, 0.3, 0.8],
#         [0.2, 1.0, 0.5, 1.0],
#         [0.3, 0.4, 1.0, 0.7],
#         [0.4, 0.3, 0.2, 1.0],
#     ], dtype=float)
#     edge_pairs = [(0, 1), (1, 2), (2, 3)]
#     path_pairs = [(0, 1), (1, 2), (2, 3)]
#     trek_pairs = [(0, 1)]
#     w_threshold = 0.3

#     def test_func(w):
#         c_e, G_e = combined_equality_constraints(w, edge_pairs, path_pairs, trek_pairs)
#         i_value, i_grad = combined_inequality_constraints(w, w_threshold, edge_pairs, path_pairs, trek_pairs)
#         epsilon = 1e-1
#         c_i = epsilon - i_value  
#         G_i = -i_grad
#         l = c_i.shape[0]
#         k = c_e.shape[0]
#         np.random.seed(0)
#         beta = np.random.uniform(0, 1, size=l)
#         rho = 1.0
#         alpha = np.random.uniform(0, 1, size=k)
#         z = beta + rho * c_i
#         positive_part = np.maximum(z, 0.0)
#         obj = 0.5 * rho * np.sum(c_e**2) + alpha @ c_e + ( 1 / (2 * rho) ) * (np.sum(positive_part ** 2) - np.sum(beta**2)) 
#         g_obj = G_e.T @ (alpha + rho * c_e) + G_i.T @ positive_part
#         return obj, g_obj
    
#     def f(w):
#         residuals, _ = test_func(w.reshape(d, d))
#         return residuals

#     def grad(w):
#         _, jacobian = test_func(w.reshape(d, d))
#         return jacobian

#     x0 = W.reshape(-1)

#     print("x0:", x0.shape)
#     print("f:", np.shape(f(x0)))
#     print("grad:", grad(x0).shape)
#     absolute_error = check_grad(f, grad, x0)
#     gradient_norm = np.linalg.norm(grad(x0))
#     relative_error = absolute_error / max(1.0, gradient_norm)

#     print("absolute error:", absolute_error)
#     print("relative error:", relative_error)

#     tests = [
#         (
#             "exist_edges",
#             lambda W: _exist_edges(W, w_threshold, edge_pairs),
#         ),
#         (
#             "exist_paths",
#             lambda W: _exist_paths(W, w_threshold, path_pairs),
#         ),
#         (
#             "exist_trek",
#             lambda W: _exist_trek(W, w_threshold, trek_pairs),
#         ),
#         (
#             "forbid_edges",
#             lambda W: _forbid_edges(W, edge_pairs),
#         ),
#         (
#             "forbid_paths",
#             lambda W: _forbid_paths(W, path_pairs),
#         ),
#         (
#             "forbid_trek",
#             lambda W: _forbid_trek(W, trek_pairs),
#         ),
#     ]

#     x0 = W.reshape(-1)

#     for name, constraint_function in tests:
#         values, jacobian = constraint_function(W)

#         values = np.atleast_1d(values)
#         jacobian = np.asarray(jacobian)

#         # Scalar constraints return a one-dimensional gradient.
#         if jacobian.ndim == 1:
#             jacobian = jacobian.reshape(1, -1)

#         print("\nTesting:", name)
#         print("value shape:", values.shape)
#         print("gradient shape:", jacobian.shape)

#         # check_grad requires a scalar function, so check each component.
#         for k in range(values.size):

#             def f(w):
#                 result, _ = constraint_function(w.reshape(d, d))
#                 return np.atleast_1d(result)[k]

#             def grad(w):
#                 _, result_gradient = constraint_function(
#                     w.reshape(d, d)
#                 )

#                 result_gradient = np.asarray(result_gradient)

#                 if result_gradient.ndim == 1:
#                     return result_gradient

#                 return result_gradient[k]

#             err = check_grad(f, grad, x0)

#             print(
#                 f"constraint {k}, "
#                 f"check_grad difference: {err}"
#             )
