import argparse
import json
from prior_notears.locally_connected import LocallyConnected
from prior_notears.lbfgsb_scipy import LBFGSBScipy
from prior_notears.trace_expm import trace_expm
import torch
import torch.nn as nn
import numpy as np
from notears import nonlinear
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from Varsortability.src.varsortability import varsortability
import time


class NotearsMLP(nn.Module):
    def __init__(self, dims, bias=True):
        super(NotearsMLP, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        h = trace_expm(A) - d  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        # M = torch.eye(d) + A / d  # (Yu et al. 2019)
        # E = torch.matrix_power(M, d - 1)
        # h = (E.t() * M).sum() - d
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    # @torch.no_grad()
    def fc1_to_adj(self) -> torch.Tensor:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        W_squared = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        # W = torch.sqrt(A)  # [i, j]
        # W = W.cpu().detach().numpy()  # [i, j]
        return W_squared

def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss

def likelihood_loss(output, target):
    """likelihood loss for tensors with shape [n, d]."""

    residual_mean = torch.mean(
        (target - output) ** 2,
        dim=0,
    )
    residual_mean = residual_mean.clamp_min(
        torch.finfo(residual_mean.dtype).tiny
    )

    return 0.5 * torch.sum(torch.log(residual_mean))

def forbid_edges(W_squared, edge_pairs):
    """forbid edges from the list of index pairs.
    
    Args:
        W_squared (torch.Tensor): [d, d] squared weight matrix
        pairs (list): List of (i, j) edge pairs

    Returns:
        float: Values Sum of W[i, j] for each (i, j) pair
    """
    if not edge_pairs:
        raise ValueError("edge_pairs must not be empty")
    return torch.stack([W_squared[i, j] for i, j in edge_pairs]).mean()

def exist_edges(W_squared, w_thres, edge_pairs):
    """Return a vector of edge residuals for the given index pairs.
    
    Args:
        W_squared (torch.Tensor): [d, d] squared weight matrix
        w_thres (float): threshold for edge existence
        pairs (list): List of (i, j) edge pairs

    Returns:
        torch.Tensor (1D): Vector of W_squared[i, j] - w_thres for each (i, j) pair
    """
    if not edge_pairs:
        return W_squared.new_empty((0,))
    residuals = torch.stack([
        W_squared[i, j] - w_thres**2
        for i, j in edge_pairs
    ])

    return residuals

def forbid_paths(W_squared, path_pairs):

    """Compute the forbidden-path penalty and its gradient.
    Args:
        W_squared: Array of shape (d, d).
        path_pairs: Sequence of (start, end) index pairs.
    Returns:
        value: Scalar penalty.
        gradient: Flattened gradient with respect to W.
    """

    if len(path_pairs) == 0:
        raise ValueError("path_pairs must not be empty")
    E = torch.matrix_exp(W_squared)
    return torch.stack([E[i, j] for i, j in path_pairs]).mean()

def exist_paths(W_squared, w_thres, path_pairs):
    if not path_pairs:
        return W_squared.new_empty((0,))
    X = W_squared - w_thres * w_thres
    A = torch.relu(X)
    E = torch.matrix_exp(A)
    residuals = torch.stack([E[i, j] for i, j in path_pairs])
    return residuals

def forbid_trek(W_squared, trek_pairs):
    """Return the mean forbidden-trek penalty and its gradient."""
    if not trek_pairs:
        raise ValueError("trek_pairs must not be empty")
    E = torch.matrix_exp(W_squared)
    T = E.T @ E
    return torch.stack([T[i, j] for i, j in trek_pairs]).mean()

def exist_trek(W_squared, w_thres, trek_pairs):

    """Return trek residuals and their Jacobian.

    Args:
        W_squared: Squared weight matrix with shape (d, d).
        w_thres: Threshold for trek existence.
        trek_pairs: Sequence of endpoint pairs (i, j).
        sharpness: Softplus sharpness parameter.

    Returns:
        residuals: Array with shape (len(trek_pairs),).
        J: Jacobian with shape (len(trek_pairs), W.size).
    """
    X = W_squared - w_thres * w_thres
    A = torch.relu(X)
    E = torch.matrix_exp(A)
    T = E.T @ E

    if not trek_pairs:
        return W_squared.new_empty((0,))

    residuals = torch.stack([
        T[i, j]
        for i, j in trek_pairs
    ])

    return residuals

def combined_equality_constraints(W_squared, forbid_edge_pairs, forbid_path_pairs, forbid_trek_pairs):
    """Combine active equality constraints and their Jacobians.

    Returns:
        values: Shape (m,), where m is the number of active constraints.
        jacobian: Shape (m, d*d).
    """
    values = []

    if forbid_edge_pairs:
        edge_value = forbid_edges(
            W_squared, forbid_edge_pairs
        )
        values.append(edge_value)

    if forbid_path_pairs:
        path_value = forbid_paths(
            W_squared, forbid_path_pairs
        )
        values.append(path_value)

    if forbid_trek_pairs:
        trek_value = forbid_trek(
            W_squared, forbid_trek_pairs
        )
        values.append(trek_value)

    if not values:
        return W_squared.new_empty((0,))
    return torch.stack(values)

def combined_inequality_constraints(W_squared, exist_edge_pairs, exist_path_pairs, exist_trek_pairs, w_threshold):
    """Combine active inequality constraints and their Jacobians.

    Returns:
        values: Shape (m,), with one value per pair.
        jacobian: Shape (m, d*d).
    """
    values = []

    if exist_edge_pairs:
        edge_values = exist_edges(
            W_squared,
            w_threshold,
            exist_edge_pairs,
        )
        values.append(edge_values.reshape(-1))

    if exist_path_pairs:
        path_values = exist_paths(
            W_squared,
            w_threshold,
            exist_path_pairs,
        )
        values.append(path_values.reshape(-1))

    if exist_trek_pairs:
        trek_values = exist_trek(
            W_squared,
            w_threshold,
            exist_trek_pairs,
        )
        values.append(trek_values.reshape(-1))

    if not values:
        return W_squared.new_empty((0,))
    return torch.cat(values)

def violation(c_e, c_i):
    active_i = torch.relu(c_i)

    all_violations = torch.cat([
        c_e.reshape(-1),
        active_i.reshape(-1),
    ])

    if all_violations.numel() == 0:
        zero = c_e.new_zeros(())
        return zero, zero

    l2_violation = torch.linalg.vector_norm(
        all_violations,
        ord=2,
    )

    max_violation = torch.linalg.vector_norm(
        all_violations,
        ord=float("inf"),
    )

    return l2_violation, max_violation

def dual_ascent_step(model, X_torch, lambda1, lambda2, rho, alpha, beta, rho_max, prior_knowledge, loss_type, w_threshold, epsilon, l2_violation):
    """Perform one step of dual ascent in augmented Lagrangian."""
    if prior_knowledge is None:
        prior_knowledge = {}

    forbid_edge_pairs = prior_knowledge.get("forbid_edge_pairs", [])
    forbid_path_pairs = prior_knowledge.get("forbid_path_pairs", [])
    forbid_trek_pairs = prior_knowledge.get("forbid_trek_pairs", [])

    exist_edge_pairs = prior_knowledge.get("exist_edge_pairs", [])
    exist_path_pairs = prior_knowledge.get("exist_path_pairs", [])
    exist_trek_pairs = prior_knowledge.get("exist_trek_pairs", [])

    optimizer = LBFGSBScipy(model.parameters())
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            X_hat = model(X_torch)
            if loss_type == 'l2':
                loss = squared_loss(X_hat, X_torch)
            elif loss_type == 'likelihood':
                loss = likelihood_loss(X_hat, X_torch)
            else:
                raise ValueError('unknown loss type')
            W_est = model.fc1_to_adj()
            h_val = model.h_func()
            c_e = combined_equality_constraints(
                W_est,
                forbid_edge_pairs,
                forbid_path_pairs,
                forbid_trek_pairs,
            )

            c_e = torch.cat([
                h_val.reshape(1),
                c_e.reshape(-1),
            ])
            i_value = combined_inequality_constraints(W_est, exist_edge_pairs, exist_path_pairs, exist_trek_pairs, w_threshold)
            c_i = epsilon - i_value
            z = beta + rho * c_i
            positive_part = torch.relu(z)

            penalty = (
                0.5 * rho * torch.sum(c_e ** 2)
                + torch.dot(alpha, c_e)
                + (1.0 / (2.0 * rho))
                * (
                    torch.sum(positive_part ** 2)
                    - torch.sum(beta ** 2)
                )
            )
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj = loss + penalty + l2_reg + l1_reg
            primal_obj.backward()
            return primal_obj
        optimizer.step(closure)  # NOTE: updates model in-place
        W_est_new = model.fc1_to_adj()
        h_val_new = model.h_func()
        c_e_new = combined_equality_constraints(
            W_est_new,
            forbid_edge_pairs,
            forbid_path_pairs,
            forbid_trek_pairs,
        )

        c_e_new = torch.cat([
            h_val_new.reshape(1),
            c_e_new.reshape(-1),
        ])
        i_value_new = combined_inequality_constraints(W_est_new, exist_edge_pairs, exist_path_pairs, exist_trek_pairs, w_threshold)
        c_i_new = epsilon - i_value_new
        ###############################
        print("equality constraints:", c_e_new)
        print("inequality constraints:", c_i_new)
        ###############################
        with torch.no_grad():
            l2_violation_new, max_violation_new = violation(c_e_new, c_i_new,)
            if l2_violation_new > 0.25 * l2_violation:
                rho *= 10
            else:
                break
    l2_violation = l2_violation_new
    with torch.no_grad():
        alpha.add_(rho * c_e_new)
        beta.copy_(torch.relu(beta + rho * c_i_new))
    return rho, l2_violation, max_violation_new


def notears_nonlinear(model: nn.Module,
                      X: np.ndarray,
                      prior_knowledge=None,
                      loss_type: str = 'likelihood',                     
                      lambda1: float = 0.01,
                      lambda2: float = 0.01,
                      max_iter: int = 100,
                      violation_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3,
                      epsilon: float = 1e-1):   
    if prior_knowledge is None:
        prior_knowledge = {}

    forbid_edge_pairs = prior_knowledge.get("forbid_edge_pairs", [])
    forbid_path_pairs = prior_knowledge.get("forbid_path_pairs", [])
    forbid_trek_pairs = prior_knowledge.get("forbid_trek_pairs", [])

    exist_edge_pairs = prior_knowledge.get("exist_edge_pairs", [])
    exist_path_pairs = prior_knowledge.get("exist_path_pairs", [])
    exist_trek_pairs = prior_knowledge.get("exist_trek_pairs", [])

    rho = 1.0
    parameter = next(model.parameters())
    l2_violation = torch.tensor(
    float("inf"), dtype=parameter.dtype, device=parameter.device
    )
    equality_len = 1 + sum(
    bool(pairs)
    for pairs in (
        forbid_edge_pairs,
        forbid_path_pairs,
        forbid_trek_pairs,
        )
    )
    inequality_len = len(exist_edge_pairs) + len(exist_path_pairs) + len(exist_trek_pairs)
    alpha = torch.zeros(equality_len, dtype=parameter.dtype, device=parameter.device)
    beta = torch.zeros(inequality_len, dtype=parameter.dtype, device=parameter.device)
    X_torch = torch.as_tensor(
            X, dtype=parameter.dtype, device=parameter.device
        )
    for _ in range(max_iter):
        rho, l2_violation, max_violation = dual_ascent_step(model, X_torch, lambda1, lambda2, rho, alpha, beta, rho_max, 
                                                              prior_knowledge, loss_type, w_threshold, epsilon, l2_violation)
        if max_violation <= violation_tol or rho >= rho_max:
            break
    with torch.no_grad():        
        W_squared_est = model.fc1_to_adj()
        W_est = torch.sqrt(W_squared_est)
        W_est[torch.abs(W_est) < w_threshold] = 0
    return W_est

def evaluate_prior_values(W, prior_knowledge, w_threshold):
    W = torch.as_tensor(W, dtype=torch.get_default_dtype())
    W_squared = W * W
    constraint_values = {}

    forbid_functions = {
        "forbid_edge_pairs": forbid_edges,
        "forbid_path_pairs": forbid_paths,
        "forbid_trek_pairs": forbid_trek,
    }

    exist_functions = {
        "exist_edge_pairs": exist_edges,
        "exist_path_pairs": exist_paths,
        "exist_trek_pairs": exist_trek,
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
                float(constraint_function(W_squared, [pair]).item())
                for pair in pairs
            ]

        elif prior_key in exist_functions:
            values = exist_functions[prior_key](
                W_squared,
                w_threshold,
                pairs,
            )

            values = values.detach().cpu().reshape(-1).tolist()

        else:
            raise ValueError(
                f"Unknown prior-knowledge key: {prior_key}"
            )

        constraint_values[prior_key] = values

    return constraint_values

def main():
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)
    parser = argparse.ArgumentParser(prog='nonlinear NOTEARS with prior knowledge',)
    parser.add_argument('-s', '--seed', dest='s',  default=42, type=int)
    parser.add_argument('-d', '--num_nodes', dest='d', default=4, type=int)
    parser.add_argument('-e', '--num_edges_per_node', dest='e', default=1, type=int)
    parser.add_argument('-g', '--graph_type', dest='g', default="ER", type=str)
    parser.add_argument('-l', '--loss_type', dest='l', default="both", type=str)
    parser.add_argument('-m', '--sem_type', dest='sem_type', default="mlp", type=str)
    parser.add_argument('-p', '--prior_type', dest='p', default="mix", type=str)
    parser.add_argument('-r', '--prior_rate', dest='r', default=0.25, type=float)
    parser.add_argument('-t', '--w_threshold', dest='t', default=0.3, type=float)
    parser.add_argument('-ep', '--epsilon', dest='ep', default=1e-1, type=float)
    args = parser.parse_args()

    from prior_notears import utils
    utils.set_random_seed(args.s)

    n, d, s0, graph_type, sem_type = 10*args.d, args.d, args.e*args.d, args.g, args.sem_type
    B_true = utils.simulate_dag(d, s0, graph_type)
    print("B_true:", B_true)
    filename = f"nonlinear_{args.p}_{graph_type}{args.e}_d{d}_{sem_type}_rate{args.r}_epsilon{args.ep}_seed{args.s}.json"

    noise_scale = np.exp(np.random.uniform(np.log(0.5), np.log(2.0), size=d,))
    X, W_true = utils.simulate_nonlinear_sem(
        B_true,
        n,
        sem_type,
        noise_scale,
        return_weighted_adjacency=True,
    )
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    varsortability_score = varsortability(X_std, B_true)
    prior_knowledge = utils.generate_prior_knowledge(
            B_true,
            prior_rate=args.r,
            prior_type=args.p,
        )
    print("prior_knowledge:", prior_knowledge)

    if args.l in ('both', 'likelihood'):
        print(f'>>> Evaluation with prior knowledge and likelihood loss <<<')
        start_time = time.perf_counter()
        torch.manual_seed(args.s)
        model = NotearsMLP(dims=[d, 10, 1], bias=True)
        W_est_prior_ll = notears_nonlinear(model, X_std, prior_knowledge=prior_knowledge, loss_type="likelihood", w_threshold=args.t, epsilon=args.ep)
        running_time_prior_ll = time.perf_counter() - start_time
        assert utils.is_dag(W_est_prior_ll)
        print("W_est_prior_ll:", W_est_prior_ll)
        acc_prior_ll = utils.count_accuracy(B_true, W_est_prior_ll != 0)
        constraint_values_prior_ll = evaluate_prior_values(W_est_prior_ll, prior_knowledge, args.t)
        print(acc_prior_ll)

    if args.l in ('both', 'l2'):
        print(f'>>> Evaluation with prior knowledge and l2 loss <<<')
        start_time = time.perf_counter()
        torch.manual_seed(args.s)
        model = NotearsMLP(dims=[d, 10, 1], bias=True)
        W_est_prior_l2 = notears_nonlinear(model, X_std, prior_knowledge=prior_knowledge, loss_type="l2", w_threshold=args.t, epsilon=args.ep)
        running_time_prior_l2 = time.perf_counter() - start_time
        assert utils.is_dag(W_est_prior_l2)
        print("W_est_prior_l2:", W_est_prior_l2)
        acc_prior_l2 = utils.count_accuracy(B_true, W_est_prior_l2 != 0)
        constraint_values_prior_l2 = evaluate_prior_values(W_est_prior_l2, prior_knowledge, args.t)
        print(acc_prior_l2)

    if args.l in ('both', 'likelihood'):
        print(f'>>> Evaluation without prior knowledge and likelihood loss <<<')
        start_time = time.perf_counter()
        torch.manual_seed(args.s)
        model = nonlinear.NotearsMLP(dims=[d, 10, 1], bias=True)
        W_est_no_prior_ll = nonlinear.notears_nonlinear(model, X_std, loss_type="likelihood", w_threshold=args.t)
        running_time_no_prior_ll = time.perf_counter() - start_time
        assert utils.is_dag(W_est_no_prior_ll)
        print("W_est_no_prior:", W_est_no_prior_ll)
        acc_no_prior_ll = utils.count_accuracy(B_true, W_est_no_prior_ll != 0)
        constraint_values_no_prior_ll = evaluate_prior_values(W_est_no_prior_ll, prior_knowledge, args.t)
        print(acc_no_prior_ll)

    if args.l in ('both', 'l2'):
        print('>>> Evaluation without prior knowledge and l2 loss <<<')
        start_time = time.perf_counter()
        torch.manual_seed(args.s)
        model = nonlinear.NotearsMLP(dims=[d, 10, 1], bias=True)
        W_est_no_prior_l2 = nonlinear.notears_nonlinear(model, X_std, loss_type="l2", w_threshold=args.t)
        running_time_no_prior_l2 = time.perf_counter() - start_time
        assert utils.is_dag(W_est_no_prior_l2)
        print("W_est_no_prior:", W_est_no_prior_l2)
        acc_no_prior_l2 = utils.count_accuracy(B_true, W_est_no_prior_l2 != 0)
        constraint_values_no_prior_l2 = evaluate_prior_values(W_est_no_prior_l2, prior_knowledge, args.t)
        print(acc_no_prior_l2)

    results = {
        "B_true": B_true.tolist(),
        "W_true": W_true.tolist(),
        "X": X.tolist(),
        "X_std": X_std.tolist(),
        "varsortability_score": varsortability_score,
        "prior_knowledge": prior_knowledge,
    }

    optional_result_names = (
        "W_est_prior_ll",
        "W_est_prior_l2",
        "W_est_no_prior_ll",
        "W_est_no_prior_l2",
        "acc_prior_ll",
        "acc_prior_l2",
        "acc_no_prior_ll",
        "acc_no_prior_l2",
        "constraint_values_prior_ll",
        "constraint_values_prior_l2",
        "constraint_values_no_prior_ll",
        "constraint_values_no_prior_l2",
        "running_time_prior_ll",
        "running_time_prior_l2",
        "running_time_no_prior_ll",
        "running_time_no_prior_l2"
    )
    current_scope = locals()
    for name in optional_result_names:
        if name in current_scope:
            value = current_scope[name]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().tolist()
            elif isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, np.generic):
                value = value.item()
            results[name] = value

    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / f"nonlinear_{args.p}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    with output_path.open("w") as file:
        json.dump(results, file, indent=4)

    print(f"Results saved to: {output_path}")

if __name__ == '__main__':
    main()
