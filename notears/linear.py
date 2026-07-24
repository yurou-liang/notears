import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.optimize import check_grad
from scipy.special import expit as sigmoid


def softplus_beta(x, beta=10.0):
    return np.log1p(np.exp(beta * x)) / beta

def _forbid_paths(W, path_pairs):
        """forbid paths from the list of index pairs.
        
        Args:
            W (np.ndarray): [d, d] weight matrix
            paths (list): List of paths pairs (each pair is the beginning and end of the path)

        Returns:
            float: Sum of exponential weights along each path
        """

        E = slin.expm(W * W)
        M = np.zeros_like(W, dtype=np.result_type(W, np.float64))
        scale = len(path_pairs)

        for i, j in path_pairs:
            M[i, j] += 1.0

        A = W * W
        grad_A = slin.expm_frechet(
        A.T,
        M,
        compute_expm=False,
        ) / scale

        grad_W = 2.0 * W * grad_A 
        return np.sum([E[i, j] for i, j in path_pairs]) / len(path_pairs), grad_W.reshape(-1)

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
            G_e[i, j] = 2 * W[i, j]
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
        residuals = np.array([W[i, j] - w_thres for i, j in edge_pairs])
        d = W.shape[0]
        jacobian = np.zeros((len(edge_pairs), d * d), dtype=float)
        for k, (i, j) in enumerate(edge_pairs):
            jacobian[k, i * d + j] = 1.0
        return residuals, jacobian

def _exist_paths(W, w_thres, path_pairs, beta=10.0):
    X = W * W - w_thres * w_thres
    A = softplus_beta(X, beta)
    E = slin.expm(A)

    residuals = np.array([E[i, j] for i, j in path_pairs])

    dA_dX = sigmoid(beta * X)          # derivative of softplus_beta
    dX_dW = 2.0 * W                    # derivative of W^2
    J = np.zeros((len(path_pairs), W.size), dtype=float)

    for k, (i, j) in enumerate(path_pairs):
        M = np.zeros_like(W, dtype=np.float64)
        M[i, j] = 1.0
        grad_A = slin.expm_frechet(A.T, M, compute_expm=False)
        J[k, :] = (dX_dW * dA_dX * grad_A).reshape(-1)

    return residuals, J

def _forbid_trek(W, trek_pairs):
    """forbid treks from the list of index pairs.
    
    Args:
        W (np.ndarray): [d, d] weight matrix
        trek_pairs (list): List of trek pairs (each pair is the both ends of the trek)

    Returns:
        tuple: (value, grad_W) where value is the scalar objective and grad_W is the gradient matrix
    """
    E = slin.expm(W * W)
    scale = len(trek_pairs)
    value = np.sum([(E.T @ E)[i, j] for i, j in trek_pairs]) / scale

    M = np.zeros_like(W, dtype=np.result_type(W, np.float64))
    for i, j in trek_pairs:
        M[i, j] = 1.0

    A = W * W
    G_E = (E @ M.T + E @ M) / scale
    grad_A = slin.expm_frechet(A.T, G_E, compute_expm=False)
    grad_W = 2.0 * W * grad_A
    return value, grad_W.reshape(-1)

def notears_linear(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
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
            G_e[i, j] = 2 * W[i, j]
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
        i = np.array([W[i, j] - w_thres for i, j in edge_pairs])
        G_i = np.zeros_like(W)
        for i, j in edge_pairs:
            G_i[i, j] = 1.0
        return i, G_i
    
    def _forbid_paths(W, path_pairs):
        """forbid paths from the list of index pairs.
        
        Args:
            W (np.ndarray): [d, d] weight matrix
            paths (list): List of paths pairs (each pair is the beginning and end of the path)

        Returns:
            float: Sum of exponential weights along each path
        """

        E = slin.expm(W * W)
        M = np.zeros_like(W, dtype=np.result_type(W, np.float64))
        scale = len(path_pairs)

        for i, j in path_pairs:
            M[i, j] += 1.0

        A = W * W
        grad_A = slin.expm_frechet(
        A.T,
        M,
        compute_expm=False,
        ) / scale

        grad_W = 2.0 * W * grad_A 
        return np.sum([E[i, j] for i, j in path_pairs]) / len(path_pairs), grad_W
    
    def _exist_paths(W, w_thres, path_pairs):
        """Return a vector of path residuals for the given index pairs.
        
        Args:
            W (np.ndarray): [d, d] weight matrix
            paths (list): List of paths pairs (each pair is the beginning and end of the path)

        Returns:
            np.ndarray (1D): Vector of path (exponential) weights residuals for each (i, j) pair
        """
        A = softplus_beta((W*W) - w_thres*w_thres)
        return np.array([slin.expm(A)[i, j] for i, j in path_pairs])
    
    def _forbid_trek(W, trek_pairs):
        """forbid treks from the list of index pairs.
        
        Args:
            W (np.ndarray): [d, d] weight matrix
            trek_pairs (list): List of trek pairs (each pair is the both ends of the trek)

        Returns:
            tuple: (value, grad_W) where value is the scalar objective and grad_W is the gradient matrix
        """
        E = slin.expm(W * W)
        scale = len(trek_pairs)
        value = np.sum([(E.T @ E)[i, j] for i, j in trek_pairs]) / scale

        M = np.zeros_like(W, dtype=np.result_type(W, np.float64))
        for i, j in trek_pairs:
            M[i, j] = 1.0

        A = W * W
        G_E = (E @ M.T + E @ M) / scale
        grad_A = slin.expm_frechet(A.T, G_E.T, compute_expm=False).T
        grad_W = 2.0 * W * grad_A
        return value, grad_W
    
    def _exist_trek(W, w_thres, trek_pairs):
        """Return a vector of trek residuals for the given index pairs.
        
        Args:
            W (np.ndarray): [d, d] weight matrix
            w_thres (float): threshold for trek existence
            trek_pairs (list): List of trek pairs (each pair is the both ends of the trek)

        Returns:
            np.ndarray (1D): Vector of trek (exponential) weights residuals for each (i, j) pair
        """
        A = slin.expm(softplus_beta((W*W) - w_thres*w_thres))
        return np.array([(A.T@A)[i, j] for i, j in trek_pairs])

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


# if __name__ == '__main__':
#     from notears import utils
#     utils.set_random_seed(1)

#     n, d, s0, graph_type, sem_type = 100, 20, 20, 'ER', 'gauss'
#     B_true = utils.simulate_dag(d, s0, graph_type)
#     W_true = utils.simulate_parameter(B_true)
#     np.savetxt('W_true.csv', W_true, delimiter=',')

#     X = utils.simulate_linear_sem(W_true, n, sem_type)
#     np.savetxt('X.csv', X, delimiter=',')

#     W_est = notears_linear(X, lambda1=0.1, loss_type='l2')
#     assert utils.is_dag(W_est)
#     np.savetxt('W_est.csv', W_est, delimiter=',')
#     acc = utils.count_accuracy(B_true, W_est != 0)
#     print(acc)

if __name__ == '__main__':
    d = 4
    W = np.random.randn(d, d)
    path_pairs = [(0, 1), (1, 2), (2, 3)]
    w_threshold = 0.0

    # def f(w):
    #     return _forbid_paths_obj(w.reshape(d, d), path_pairs)

    # def grad(w):
    #     return _forbid_paths_grad(w.reshape(d, d), path_pairs).reshape(-1)

    def f(w):
        residuals, _ = _forbid_trek(w.reshape(d, d), path_pairs)
        return residuals

    def grad(w):
        _, jacobian = _forbid_trek(w.reshape(d, d), path_pairs)
        return jacobian

    x0 = W.reshape(-1)

    print("x0:", x0.shape)
    print("f:", np.shape(f(W)))
    print("grad:", grad(x0).shape)
    err = check_grad(f, grad, W.reshape(-1))
    # err_alt = check_grad(f, grad_alt, W.reshape(-1))
    print('check_grad with A.T transpose:', err)
    # print('check_grad without transpose:', err_alt)
    # print('f(W) =', f(W.reshape(-1)))
    # print('norm grad transpose:', np.linalg.norm(grad(W.reshape(-1))))
    # print('norm grad no-transpose:', np.linalg.norm(grad_alt(W.reshape(-1))))

