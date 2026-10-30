"""CPDAG SHD from saved, already-thresholded NOTEARS matrices.

Conversion: skeleton + unshielded colliders + Meek rules R1--R3.
SHD counts one edit per unordered node pair whose edge state differs
(absent, undirected, forward, backward). Original result files are unchanged.
"""
from pathlib import Path
from itertools import combinations, product
import csv
import json
import re
import numpy as np
import networkx as nx


def cpdag(B):
    B = np.asarray(B, dtype=bool)
    if B.ndim != 2 or B.shape[0] != B.shape[1] or not nx.is_directed_acyclic_graph(nx.from_numpy_array(B, create_using=nx.DiGraph)):
        raise ValueError('Matrix is not a DAG')
    skeleton = B | B.T
    U = skeleton.copy()
    D = np.zeros_like(B)
    def orient(a, b):
        assert B[a, b], 'Orientation inconsistent with input DAG'
        U[a, b] = U[b, a] = False
        D[a, b] = True
    for b in range(len(B)):
        for a, c in combinations(np.flatnonzero(B[:, b]), 2):
            if not skeleton[a, c]:
                orient(a, b); orient(c, b)
    while True:
        changed = False
        for a, b in zip(*np.where(U)):
            # R1: c -> a - b, c nonadjacent to b.
            r1 = np.any(D[:, a] & ~skeleton[:, b])
            # R2: a -> c -> b, a - b.
            r2 = np.any(D[a, :] & D[:, b])
            # R3: a-c, a-d, c->b, d->b, c nonadjacent to d.
            candidates = np.flatnonzero(U[a, :] & D[:, b])
            r3 = any(not skeleton[c, d] for c, d in combinations(candidates, 2))
            if U[a, b] and (r1 or r2 or r3):
                orient(a, b); changed = True
        if not changed:
            return D.astype(int) + U.astype(int) * 2


def shd(A, B):
    return sum((int(A[i,j]), int(A[j,i])) != (int(B[i,j]), int(B[j,i]))
               for i,j in combinations(range(len(A)), 2))


def validate():
    # Independent exhaustive oracle: group every labeled 4-node DAG by its
    # skeleton and unshielded colliders, then find invariant orientations.
    groups = {}
    pairs = list(combinations(range(4), 2))
    for states in product(range(3), repeat=6):
        B = np.zeros((4,4), dtype=bool)
        for (i,j), state in zip(pairs, states):
            if state: B[(i,j) if state==1 else (j,i)] = True
        if not nx.is_directed_acyclic_graph(nx.from_numpy_array(B, create_using=nx.DiGraph)): continue
        sk = B | B.T
        colliders = tuple((a,b,c) for b in range(4) for a,c in combinations(np.flatnonzero(B[:,b]),2) if not sk[a,c])
        key=(tuple(sk.flatten()),colliders)
        groups.setdefault(key,[]).append(B)
    for graphs in groups.values():
        directed = np.logical_and.reduce(graphs)
        sk = graphs[0] | graphs[0].T
        expected = directed.astype(int) + (sk & ~(directed | directed.T)).astype(int)*2
        for B in graphs: assert np.array_equal(cpdag(B),expected)
    print('Validated conversion against all 543 labeled four-node DAGs.')


def main():
    validate()
    methods = {
        'adam_mtl': ('linear_exist_path_pairs_modified_mtl','*_mtl_adam.json'),
        'adam_soft': ('linear_exist_path_pairs_adam_soft','*.json'),
        'lbfgsb_soft': ('linear_exist_path_pairs_soft','*_soft.json'),
    }
    records = {}; audit=[]
    for method,(folder,pattern) in methods.items():
        for p in Path(folder).glob(pattern):
            if 'compare' in p.name: continue
            result=json.loads(p.read_text()); d=len(result['B_true'])
            seed=int(re.search(r'_seed(\d+)',p.name)[1])
            true=cpdag(np.asarray(result['B_true'])!=0)
            for loss in ['ll','l2']:
                key=(d,seed,loss)
                try:
                    est=cpdag(np.asarray(result['W_est_prior_'+loss])!=0)
                    row=dict(method=method,d=d,seed=seed,loss=loss,cpdag_shd=shd(true,est),file=str(p))
                    if (method,key) in records: raise RuntimeError('Duplicate configuration')
                    records[method,key]=(row,result)
                except ValueError as e: audit.append(dict(method=method,d=d,seed=seed,loss=loss,error=str(e)))
    rows=[r for r,_ in records.values()]
    with Path('analysis_results/path_cpdag_per_run.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for d in sorted({r['d'] for r in rows}):
        for loss in ['ll','l2']:
            seeds=set.intersection(*[{k[1] for m,k in records if m==method and k[0]==d and k[2]==loss} for method in methods])
            if not seeds:continue
            for seed in seeds:
                samples=[records[m,(d,seed,loss)][1] for m in methods]
                for field in ['B_true','W_true','X_std','prior_knowledge_'+loss]:
                    assert all(x.get(field)==samples[0].get(field) for x in samples), (d,seed,field)
            for method in methods:
                values=[records[method,(d,seed,loss)][0]['cpdag_shd'] for seed in sorted(seeds)]
                row=dict(d=d,loss=loss,method=method,n=len(values),mean=float(np.mean(values)),std=float(np.std(values,ddof=1)) if len(values)>1 else 0.,seeds=','.join(map(str,sorted(seeds))))
                summary.append(row);print(row)
    with Path('analysis_results/path_cpdag_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    Path('analysis_results/path_cpdag_exclusions.json').write_text(json.dumps(audit,indent=2))
    print('Excluded invalid DAGs:',len(audit))

if __name__=='__main__':main()
