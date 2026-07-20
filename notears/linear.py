import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid


def softplus_beta(x, beta=10.0):
    return np.log1p(np.exp(beta * x)) / beta

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
        return np.sum([(W * W)[i, j] for i, j in edge_pairs]) / len(edge_pairs)
    
    def _exist_edges(W, w_thres, edge_pairs):
        """Return a vector of edge residuals for the given index pairs.
        
        Args:
            W (np.ndarray): [d, d] weight matrix
            w_thres (float): threshold for edge existence
            pairs (list): List of (i, j) edge pairs
            
        Returns:
            np.ndarray (1D): Vector of W[i, j] - w_thres for each (i, j) pair
        """
        return np.array([W[i, j] - w_thres for i, j in edge_pairs])
    
    def _forbid_paths(W, path_pairs):
        """forbid paths from the list of index pairs.
        
        Args:
            W (np.ndarray): [d, d] weight matrix
            paths (list): List of paths pairs (each pair is the beginning and end of the path)

        Returns:
            float: Sum of exponential weights along each path
        """

        E = slin.expm(W * W)
        return np.sum([E[i, j] for i, j in path_pairs]) / len(path_pairs)
    
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
            float: Sum of trek (exponential) weights along each trek
        """
        E = slin.expm(W * W)
        return np.sum([(E.T@E)[i, j] for i, j in trek_pairs]) / len(trek_pairs)
    
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


if __name__ == '__main__':
    from notears import utils
    utils.set_random_seed(1)

    n, d, s0, graph_type, sem_type = 100, 20, 20, 'ER', 'gauss'
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)
    np.savetxt('W_true.csv', W_true, delimiter=',')

    X = utils.simulate_linear_sem(W_true, n, sem_type)
    np.savetxt('X.csv', X, delimiter=',')

    W_est = notears_linear(X, lambda1=0.1, loss_type='l2')
    assert utils.is_dag(W_est)
    np.savetxt('W_est.csv', W_est, delimiter=',')
    acc = utils.count_accuracy(B_true, W_est != 0)
    print(acc)

