import numpy as np
from scipy.special import expit as sigmoid
import igraph as ig
import random


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


def generate_prior_knowledge(B, prior_rate, prior_type, random_state=None):
    """Sample correct prior knowledge from a ground-truth DAG.

    The requested number of prior statements is ``ceil(prior_rate * d)`` and
    the returned number is capped by the number of eligible candidates. Edge
    and path pairs are directed. Trek pairs are represented once with ``i < j``
    because trek existence is symmetric.

    Args:
        B (np.ndarray): Ground-truth adjacency matrix with shape ``[d, d]``;
            ``B[i, j] != 0`` means ``i -> j``.
        prior_rate (float): Requested number of prior statements per node.
        prior_type (str): ``mix`` or one of ``forbid_edge_pairs``, ``forbid_path_pairs``,
            ``forbid_trek_pairs``, ``exist_edge_pairs``,
            ``exist_path_pairs``, or ``exist_trek_pairs``. The corresponding
            name without the ``_pairs`` suffix is also accepted.
        random_state (int, optional): Seed for local reproducible sampling. If
            omitted, sampling uses NumPy's global random state.

    Returns:
        dict: For a single type, ``{prior_type: pairs}``. For ``mix``, the
        dictionary contains all six canonical type keys. Sampling stops after
        ``ceil(prior_rate * d)`` statements or after all candidate pools are
        exhausted.
    """
    rng = np.random if random_state is None else np.random.RandomState(random_state)

    B = np.asarray(B)
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError('B must be a square adjacency matrix')
    if not np.isfinite(prior_rate) or prior_rate < 0:
        raise ValueError('prior_rate must be a finite nonnegative number')

    canonical_types = (
        'forbid_edge_pairs',
        'forbid_path_pairs',
        'forbid_trek_pairs',
        'exist_edge_pairs',
        'exist_path_pairs',
        'exist_trek_pairs',
    )
    aliases = {
        name.removesuffix('_pairs'): name
        for name in canonical_types
    }
    prior_type = aliases.get(prior_type, prior_type)
    if prior_type not in canonical_types and prior_type != 'mix':
        raise ValueError(
            f'unknown prior type {prior_type!r}; expected one of '
            f'{canonical_types} or \'mix\''
        )

    adjacency = B != 0
    d = adjacency.shape[0]
    np.fill_diagonal(adjacency, False)

    # Boolean transitive closure: reachability[i, j] is true iff there is a
    # directed path from i to j.
    reachability = adjacency.copy()
    for k in range(d):
        reachability |= (
            reachability[:, [k]] & reachability[[k], :]
        )
    np.fill_diagonal(reachability, False)

    if prior_type == 'mix':
        ancestor_relation = reachability.copy()
        np.fill_diagonal(ancestor_relation, True)
        trek_relation = (
            ancestor_relation.T.astype(np.int64)
            @ ancestor_relation.astype(np.int64)
        ) > 0

        candidate_pools = {}
        for candidate_type in canonical_types:
            if candidate_type.endswith('edge_pairs'):
                candidate_relation = adjacency
                candidate_is_directed = True
            elif candidate_type.endswith('path_pairs'):
                candidate_relation = reachability
                candidate_is_directed = True
            else:
                candidate_relation = trek_relation
                candidate_is_directed = False

            candidate_is_existing = candidate_type.startswith('exist_')
            pool = []
            for i in range(d):
                j_start = 0 if candidate_is_directed else i + 1
                for j in range(j_start, d):
                    if i == j:
                        continue
                    if bool(candidate_relation[i, j]) == candidate_is_existing:
                        pool.append((i, j))
            candidate_pools[candidate_type] = pool

        requested_count = int(np.ceil(prior_rate * d))
        prior_knowledge = {name: [] for name in canonical_types}
        available_types = [
            name for name, pool in candidate_pools.items() if pool
        ]

        selected_count = 0
        while selected_count < requested_count and available_types:
            selected_type = rng.choice(available_types)
            selected_pool = candidate_pools[selected_type]
            selected_index = int(rng.randint(len(selected_pool)))
            selected_pair = selected_pool.pop(selected_index)
            prior_knowledge[selected_type].append(selected_pair)
            selected_count += 1

            if not selected_pool:
                available_types.remove(selected_type)

        return prior_knowledge

    if prior_type.endswith('edge_pairs'):
        relation = adjacency
        directed = True
    elif prior_type.endswith('path_pairs'):
        relation = reachability
        directed = True
    else:
        # A trek exists between two nodes iff they share an ancestor. Each
        # node is included as its own ancestor, so each path is also a trek.
        ancestor_relation = reachability.copy()
        np.fill_diagonal(ancestor_relation, True)
        relation = (
            ancestor_relation.T.astype(np.int64)
            @ ancestor_relation.astype(np.int64)
        ) > 0
        directed = False

    want_existing = prior_type.startswith('exist_')
    candidates = []
    for i in range(d):
        j_start = 0 if directed else i + 1
        for j in range(j_start, d):
            if i == j:
                continue
            if bool(relation[i, j]) == want_existing:
                candidates.append((i, j))

    requested_count = int(np.ceil(prior_rate * d))
    actual_count = min(requested_count, len(candidates))
    if actual_count == 0:
        return {prior_type: []}

    selected_indices = rng.choice(
        len(candidates),
        size=actual_count,
        replace=False,
    )
    selected_pairs = [candidates[index] for index in selected_indices]
    return {prior_type: selected_pairs}


def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """Simulate SEM parameters for a DAG.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges

    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W


def simulate_linear_sem(W, n, sem_type, noise_scale=None):
    """Simulate samples from linear SEM with specified type of noise.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """
    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
    return X


def simulate_nonlinear_sem(B, n, sem_type, noise_scale=None):
    """Simulate samples from nonlinear SEM.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        n (int): num of samples
        sem_type (str): mlp, mim, gp, gp-add
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix
    """
    def _simulate_single_equation(X, scale):
        """X: [n, num of parents], x: [n]"""
        z = np.random.normal(scale=scale, size=n)
        pa_size = X.shape[1]
        if pa_size == 0:
            return z
        if sem_type == 'mlp':
            hidden = 100
            W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            x = sigmoid(X @ W1) @ W2 + z
        elif sem_type == 'mim':
            w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w1[np.random.rand(pa_size) < 0.5] *= -1
            w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w2[np.random.rand(pa_size) < 0.5] *= -1
            w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w3[np.random.rand(pa_size) < 0.5] *= -1
            x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
        elif sem_type == 'gp':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = gp.sample_y(X, random_state=None).flatten() + z
        elif sem_type == 'gp-add':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                     for i in range(X.shape[1])]) + z
        else:
            raise ValueError('unknown sem type')
        return x

    d = B.shape[0]
    scale_vec = noise_scale if noise_scale else np.ones(d)
    X = np.zeros([n, d])
    G = ig.Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j])
    return X


def count_accuracy(B_true, B_est):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
        if not is_dag(B_est):
            raise ValueError('B_est should be a DAG')
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}

if __name__ == '__main__':
    # Example DAG: 0 -> 1 -> 2, with node 3 disconnected.
    B_example = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 1],
        [0, 0, 0, 0],
    ])
    prior_rate = 0.5
    requested_count = int(np.ceil(prior_rate * B_example.shape[0]))
    prior_types = (
        'forbid_edge_pairs',
        'forbid_path_pairs',
        'forbid_trek_pairs',
        'exist_edge_pairs',
        'exist_path_pairs',
        'exist_trek_pairs',
    )

    print('Ground-truth adjacency matrix:')
    print(B_example)
    print('Requested prior knowledge per type:', requested_count)

    for seed, prior_type in enumerate(prior_types):
        prior_knowledge = generate_prior_knowledge(
            B_example,
            prior_rate=prior_rate,
            prior_type=prior_type,
            random_state=seed,
        )
        selected_pairs = prior_knowledge[prior_type]

        assert len(selected_pairs) <= requested_count
        assert len(selected_pairs) == len(set(selected_pairs))
        print(f'{prior_type}: {prior_knowledge}')

    mixed_prior_knowledge = generate_prior_knowledge(
        B_example,
        prior_rate=prior_rate,
        prior_type='mix',
        random_state=0,
    )
    mixed_count = sum(len(pairs) for pairs in mixed_prior_knowledge.values())
    assert mixed_count == requested_count
    assert all(
        len(pairs) == len(set(pairs))
        for pairs in mixed_prior_knowledge.values()
    )
    print(f'mix: {mixed_prior_knowledge}')
