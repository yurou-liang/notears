"""Repair a NoCurl potential using path-existence priors."""

from itertools import combinations
from operator import index

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp


def repair_p0(p0, exist_path_pairs):
    """Return potentials that satisfy the priors with the fewest order reversals.

    p0 is a one-dimensional vector in data-column order. Each zero-based pair
    (i, j) requires the returned potential of j to be strictly larger than i's.
    Exact ties in p0 have no reversal cost. If p0 already meets the priors,
    return a copy as supplied. Otherwise anchor the last node's potential at 0.
    """
    p0 = np.asarray(p0, dtype=float)
    if p0.ndim != 1 or not p0.size or not np.isfinite(p0).all():
        raise ValueError("p0 must be a nonempty finite one-dimensional vector")
    d = p0.size
    required = np.zeros((d, d), dtype=bool)
    for source, target in exist_path_pairs:
        i, j = index(source), index(target)
        if not (0 <= i < d and 0 <= j < d):
            raise ValueError("Prior endpoints must be zero-based indices in p0")
        required[i, j] = True

    i, j = np.nonzero(required)
    if np.all(p0[i] < p0[j]):
        return p0.copy()

    for k in range(d):
        required |= required[:, [k]] & required[[k], :]
    if np.any(required.diagonal()):
        raise ValueError("Path-existence priors contain a cycle")

    left, right = np.triu_indices(d, k=1)
    pair_count = left.size
    pair_index = np.empty((d, d), dtype=int)
    pair_index[left, right] = np.arange(pair_count)
    # x[i, j] is 1 if i precedes j, for i < j.
    objective = (p0[left] > p0[right]).astype(float) - (p0[left] < p0[right])
    bounds = Bounds(required[left, right].astype(float), (~required[right, left]).astype(float))

    constraints = ()
    if d >= 3:
        triples = np.array(list(combinations(range(d), 3)), dtype=int)
        i, j, k = triples.T
        columns = np.column_stack((pair_index[i, j], pair_index[j, k], pair_index[i, k]))
        # Enforce 0 <= x[i,j] + x[j,k] - x[i,k] <= 1.
        matrix = sparse.coo_matrix(
            (np.tile([1.0, 1.0, -1.0], len(triples)),
             (np.repeat(np.arange(len(triples)), 3), columns.ravel())),
            shape=(len(triples), pair_count),
        ).tocsc()
        constraints = LinearConstraint(matrix, 0.0, 1.0)

    result = milp(
        objective, integrality=np.ones(pair_count), bounds=bounds,
        constraints=constraints, options={"mip_rel_gap": 0.0},
    )
    if not result.success:
        raise RuntimeError(f"Could not minimize order reversals: {result.message}")

    forward = result.x > 0.5
    rank = np.bincount(np.where(forward, right, left), minlength=d)
    if not np.array_equal(np.sort(rank), np.arange(d)):
        raise RuntimeError("MILP result does not encode a total order")

    values = np.sort(p0)
    anchor = rank[-1]
    values -= values[anchor]
    # Split equal levels by a small, numerically usable gap.
    tie_step = 1e-8 * max(float(np.max(np.abs(values))), 1.0)
    for k in range(anchor + 1, d):
        if values[k] <= values[k - 1]:
            values[k] = values[k - 1] + tie_step
    for k in range(anchor - 1, -1, -1):
        if values[k] >= values[k + 1]:
            values[k] = values[k + 1] - tie_step

    repaired = values[rank]
    sources, targets = np.nonzero(required)
    if not np.isfinite(repaired).all() or not np.all(repaired[sources] < repaired[targets]):
        raise RuntimeError("Could not encode the repaired order as finite potentials")
    return repaired
