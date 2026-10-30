# Exist-path d=30: saved beta=1 versus beta=0.5 results

Read-only graph analysis; no training or algorithm changes. beta denotes the multiplier on the normalized prior direction, not Softplus sharpness or Adam momentum.

## Comparability

All ten seeds have matching ground-truth adjacency/weights, raw X, X_std, and prior lists for both losses. All twenty runs have valid saved DAG evaluations for both losses; no cycles or missing runs were found. Current linear.py uses Softplus, the original L2>0.1 gate, and a 0.5 prior multiplier. Its Git diff against HEAD changes only that multiplier in the scalar and direction. Defaults are epsilon=0.1, sharpness=50, Adam lr=0.001, max inner updates=10000, max outer iterations=100, threshold=0.3, and parameter-stability tolerance=1e-5 checked over 100-update windows with patience three. The available current shell script passes epsilon but does not override lr/budget/threshold. Filenames record epsilon=0.1 and tau=1. Logs show 10000-update solves. These records support beta as the only intended change. However, the JSONs do not store beta, lr, threshold, budget, or source revision. Historical effective settings and beta as the only actual change cannot be independently certified; current defaults are not immutable run provenance. See configuration_evidence.json and current_source_vs_HEAD.diff.

## Counting

Exact directed TP=|E intersect T|, FP=|E minus T|, FN=|T minus E|. Precision=TP/(TP+FP), recall=TP/(TP+FN), F1=2TP/(2TP+FP+FN). Reversed edges are predicted i->j with true j->i: one FP and one FN, reported separately without being added again. DAG SHD follows NOTEARS: extra skeleton pairs + missing skeleton pairs + reversed-edge count. A reversal costs one DAG SHD edit. This is not CPDAG SHD. Satisfaction is directed reachability of supplied priors in saved thresholded graphs. A beta=0.5-only false edge u->v is on a prior path iff some supplied i->j has i reaching u and v reaching j in the beta=0.5 DAG, including zero-length endpoint segments. This covers all paths without enumerating them.

## Means on matching seeds

| loss | beta | n | F1 | precision | recall | TP | FP | FN | reversed_edges | predicted_edges | DAG_SHD | satisfaction_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ll | 1 | 10 | 0.6420 | 0.6699 | 0.6200 | 18.6000 | 9.9000 | 11.4000 | 1.7000 | 28.5000 | 19.6000 | 97.2077 |
| ll | 0.5 | 10 | 0.6308 | 0.6448 | 0.6600 | 19.8000 | 20.7000 | 10.2000 | 2.1000 | 40.5000 | 28.8000 | 97.1347 |
| l2 | 1 | 10 | 0.5812 | 0.6277 | 0.5433 | 16.3000 | 10.0000 | 13.7000 | 1.4000 | 26.3000 | 22.3000 | 99.5652 |
| l2 | 0.5 | 10 | 0.6104 | 0.6862 | 0.5533 | 16.6000 | 7.7000 | 13.4000 | 1.0000 | 24.3000 | 20.1000 | 98.2960 |

## Per-seed metrics: ll, beta=1

| seed | F1 | precision | recall | TP | FP | FN | reversed_edges | predicted_edges | DAG_SHD | satisfied | prior_count | satisfaction_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.6667 | 0.7037 | 0.6333 | 19 | 8 | 11 | 2 | 27 | 17 | 22 | 23 | 95.6522 |
| 1 | 0.5714 | 0.5455 | 0.6000 | 18 | 15 | 12 | 2 | 33 | 25 | 24 | 26 | 92.3077 |
| 2 | 0.4000 | 0.3714 | 0.4333 | 13 | 22 | 17 | 4 | 35 | 35 | 33 | 36 | 91.6667 |
| 3 | 0.4918 | 0.4839 | 0.5000 | 15 | 16 | 15 | 1 | 31 | 30 | 26 | 27 | 96.2963 |
| 4 | 0.6667 | 0.7500 | 0.6000 | 18 | 6 | 12 | 1 | 24 | 17 | 27 | 27 | 100.0000 |
| 5 | 0.6780 | 0.6897 | 0.6667 | 20 | 9 | 10 | 4 | 29 | 15 | 26 | 26 | 100.0000 |
| 6 | 0.6102 | 0.6207 | 0.6000 | 18 | 11 | 12 | 1 | 29 | 22 | 25 | 26 | 96.1538 |
| 7 | 0.8214 | 0.8846 | 0.7667 | 23 | 3 | 7 | 1 | 26 | 9 | 25 | 25 | 100.0000 |
| 8 | 0.7143 | 0.7692 | 0.6667 | 20 | 6 | 10 | 1 | 26 | 15 | 31 | 31 | 100.0000 |
| 9 | 0.8000 | 0.8800 | 0.7333 | 22 | 3 | 8 | 0 | 25 | 11 | 20 | 20 | 100.0000 |

## Per-seed metrics: ll, beta=0.5

| seed | F1 | precision | recall | TP | FP | FN | reversed_edges | predicted_edges | DAG_SHD | satisfied | prior_count | satisfaction_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.7119 | 0.7241 | 0.7000 | 21 | 8 | 9 | 1 | 29 | 16 | 23 | 23 | 100.0000 |
| 1 | 0.5507 | 0.4872 | 0.6333 | 19 | 20 | 11 | 2 | 39 | 29 | 24 | 26 | 92.3077 |
| 2 | 0.5806 | 0.5625 | 0.6000 | 18 | 14 | 12 | 1 | 32 | 25 | 35 | 36 | 97.2222 |
| 3 | 0.2339 | 0.1418 | 0.6667 | 20 | 121 | 10 | 8 | 141 | 123 | 25 | 27 | 92.5926 |
| 4 | 0.4923 | 0.4571 | 0.5333 | 16 | 19 | 14 | 3 | 35 | 30 | 26 | 27 | 96.2963 |
| 5 | 0.6557 | 0.6452 | 0.6667 | 20 | 11 | 10 | 4 | 31 | 17 | 25 | 26 | 96.1538 |
| 6 | 0.7778 | 0.8750 | 0.7000 | 21 | 3 | 9 | 0 | 24 | 12 | 26 | 26 | 100.0000 |
| 7 | 0.7778 | 0.8750 | 0.7000 | 21 | 3 | 9 | 1 | 24 | 11 | 25 | 25 | 100.0000 |
| 8 | 0.7273 | 0.8000 | 0.6667 | 20 | 5 | 10 | 1 | 25 | 14 | 30 | 31 | 96.7742 |
| 9 | 0.8000 | 0.8800 | 0.7333 | 22 | 3 | 8 | 0 | 25 | 11 | 20 | 20 | 100.0000 |

## Changes at beta=0.5 minus beta=1: ll

| seed | delta_F1 | delta_DAG_SHD | delta_FP | delta_FN | delta_reversed_edges | delta_predicted_edges | delta_skeleton_extra | delta_skeleton_missing |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.0452 | -1 | 0 | -2 | -1 | 2 | 1 | -1 |
| 1 | -0.0207 | 4 | 5 | -1 | 0 | 6 | 5 | -1 |
| 2 | 0.1806 | -10 | -8 | -5 | -3 | -3 | -5 | -2 |
| 3 | -0.2579 | 93 | 105 | -5 | 7 | 110 | 98 | -12 |
| 4 | -0.1744 | 13 | 13 | 2 | 2 | 11 | 11 | 0 |
| 5 | -0.0222 | 2 | 2 | 0 | 0 | 2 | 2 | 0 |
| 6 | 0.1676 | -10 | -8 | -3 | -1 | -5 | -7 | -2 |
| 7 | -0.0437 | 2 | 0 | 2 | 0 | -2 | 0 | 2 |
| 8 | 0.0130 | -1 | -1 | 0 | 0 | -1 | -1 | 0 |
| 9 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Directed edge differences: ll

| seed | half_only_true | half_only_false | one_only_true | one_only_false | half_only_false_on_prior_paths | half_only_false_off_prior_paths |
|---|---|---|---|---|---|---|
| 0 | 3 | 1 | 1 | 1 | 0 | 1 |
| 1 | 2 | 16 | 1 | 11 | 16 | 0 |
| 2 | 13 | 11 | 8 | 19 | 11 | 0 |
| 3 | 6 | 113 | 1 | 8 | 84 | 29 |
| 4 | 1 | 14 | 3 | 1 | 12 | 2 |
| 5 | 2 | 5 | 2 | 3 | 4 | 1 |
| 6 | 4 | 1 | 1 | 9 | 0 | 1 |
| 7 | 0 | 0 | 2 | 0 | 0 | 0 |
| 8 | 0 | 0 | 0 | 1 | 0 | 0 |
| 9 | 0 | 0 | 0 | 0 | 0 | 0 |

## Prior identities: ll

| seed | prior_count | both | only_one | only_half | neither | switched |
|---|---|---|---|---|---|---|
| 0 | 23 | 22 | 0 | 1 | 0 | 1 |
| 1 | 26 | 23 | 1 | 1 | 1 | 2 |
| 2 | 36 | 32 | 1 | 3 | 0 | 4 |
| 3 | 27 | 25 | 1 | 0 | 1 | 1 |
| 4 | 27 | 26 | 1 | 0 | 0 | 1 |
| 5 | 26 | 25 | 1 | 0 | 0 | 1 |
| 6 | 26 | 25 | 0 | 1 | 0 | 1 |
| 7 | 25 | 25 | 0 | 0 | 0 | 0 |
| 8 | 31 | 30 | 1 | 0 | 0 | 1 |
| 9 | 20 | 20 | 0 | 0 | 0 | 0 |

Edge totals: {"half_only_true": 31, "half_only_false": 161, "one_only_true": 19, "one_only_false": 53, "half_only_false_on_prior_paths": 127, "half_only_false_off_prior_paths": 34}

Prior totals: {"prior_count": 267, "both": 253, "only_one": 6, "only_half": 6, "neither": 2, "switched": 12}

Metric change totals: {"delta_F1": -0.1123890479249322, "delta_DAG_SHD": 92, "delta_FP": 108, "delta_FN": -12, "delta_TP": 12, "delta_reversed_edges": 4, "delta_predicted_edges": 120, "delta_skeleton_extra": 104, "delta_skeleton_missing": -16, "delta_satisfaction_pct": -0.7298322165082283}

## Per-seed metrics: l2, beta=1

| seed | F1 | precision | recall | TP | FP | FN | reversed_edges | predicted_edges | DAG_SHD | satisfied | prior_count | satisfaction_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.6667 | 0.7500 | 0.6000 | 18 | 6 | 12 | 0 | 24 | 18 | 22 | 23 | 95.6522 |
| 1 | 0.6667 | 0.7500 | 0.6000 | 18 | 6 | 12 | 0 | 24 | 18 | 26 | 26 | 100.0000 |
| 2 | 0.5424 | 0.5517 | 0.5333 | 16 | 13 | 14 | 0 | 29 | 27 | 36 | 36 | 100.0000 |
| 3 | 0.4912 | 0.5185 | 0.4667 | 14 | 13 | 16 | 2 | 27 | 27 | 27 | 27 | 100.0000 |
| 4 | 0.6552 | 0.6786 | 0.6333 | 19 | 9 | 11 | 1 | 28 | 19 | 27 | 27 | 100.0000 |
| 5 | 0.4828 | 0.5000 | 0.4667 | 14 | 14 | 16 | 3 | 28 | 27 | 26 | 26 | 100.0000 |
| 6 | 0.5283 | 0.6087 | 0.4667 | 14 | 9 | 16 | 0 | 23 | 25 | 26 | 26 | 100.0000 |
| 7 | 0.4333 | 0.4333 | 0.4333 | 13 | 17 | 17 | 7 | 30 | 27 | 25 | 25 | 100.0000 |
| 8 | 0.6667 | 0.7037 | 0.6333 | 19 | 8 | 11 | 0 | 27 | 19 | 31 | 31 | 100.0000 |
| 9 | 0.6792 | 0.7826 | 0.6000 | 18 | 5 | 12 | 1 | 23 | 16 | 20 | 20 | 100.0000 |

## Per-seed metrics: l2, beta=0.5

| seed | F1 | precision | recall | TP | FP | FN | reversed_edges | predicted_edges | DAG_SHD | satisfied | prior_count | satisfaction_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.6182 | 0.6800 | 0.5667 | 17 | 8 | 13 | 2 | 25 | 19 | 22 | 23 | 95.6522 |
| 1 | 0.7037 | 0.7917 | 0.6333 | 19 | 5 | 11 | 1 | 24 | 15 | 25 | 26 | 96.1538 |
| 2 | 0.5185 | 0.5833 | 0.4667 | 14 | 10 | 16 | 1 | 24 | 25 | 36 | 36 | 100.0000 |
| 3 | 0.5000 | 0.5385 | 0.4667 | 14 | 12 | 16 | 2 | 26 | 26 | 27 | 27 | 100.0000 |
| 4 | 0.7925 | 0.9130 | 0.7000 | 21 | 2 | 9 | 0 | 23 | 11 | 27 | 27 | 100.0000 |
| 5 | 0.4912 | 0.5185 | 0.4667 | 14 | 13 | 16 | 1 | 27 | 28 | 26 | 26 | 100.0000 |
| 6 | 0.4906 | 0.5652 | 0.4333 | 13 | 10 | 17 | 1 | 23 | 26 | 25 | 26 | 96.1538 |
| 7 | 0.5926 | 0.6667 | 0.5333 | 16 | 8 | 14 | 1 | 24 | 21 | 25 | 25 | 100.0000 |
| 8 | 0.8136 | 0.8276 | 0.8000 | 24 | 5 | 6 | 0 | 29 | 11 | 31 | 31 | 100.0000 |
| 9 | 0.5833 | 0.7778 | 0.4667 | 14 | 4 | 16 | 1 | 18 | 19 | 19 | 20 | 95.0000 |

## Changes at beta=0.5 minus beta=1: l2

| seed | delta_F1 | delta_DAG_SHD | delta_FP | delta_FN | delta_reversed_edges | delta_predicted_edges | delta_skeleton_extra | delta_skeleton_missing |
|---|---|---|---|---|---|---|---|---|
| 0 | -0.0485 | 1 | 2 | 1 | 2 | 1 | 0 | -1 |
| 1 | 0.0370 | -3 | -1 | -1 | 1 | 0 | -2 | -2 |
| 2 | -0.0239 | -2 | -3 | 2 | 1 | -5 | -4 | 1 |
| 3 | 0.0088 | -1 | -1 | 0 | 0 | -1 | -1 | 0 |
| 4 | 0.1373 | -8 | -7 | -2 | -1 | -5 | -6 | -1 |
| 5 | 0.0085 | 1 | -1 | 0 | -2 | -1 | 1 | 2 |
| 6 | -0.0377 | 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| 7 | 0.1593 | -6 | -9 | -3 | -6 | -6 | -3 | 3 |
| 8 | 0.1469 | -8 | -3 | -5 | 0 | 2 | -3 | -5 |
| 9 | -0.0959 | 3 | -1 | 4 | 0 | -5 | -1 | 4 |

## Directed edge differences: l2

| seed | half_only_true | half_only_false | one_only_true | one_only_false | half_only_false_on_prior_paths | half_only_false_off_prior_paths |
|---|---|---|---|---|---|---|
| 0 | 0 | 2 | 1 | 0 | 0 | 2 |
| 1 | 2 | 2 | 1 | 3 | 2 | 0 |
| 2 | 3 | 4 | 5 | 7 | 4 | 0 |
| 3 | 1 | 1 | 1 | 2 | 1 | 0 |
| 4 | 3 | 0 | 1 | 7 | 0 | 0 |
| 5 | 1 | 3 | 1 | 4 | 3 | 0 |
| 6 | 3 | 3 | 4 | 2 | 3 | 0 |
| 7 | 4 | 4 | 1 | 13 | 3 | 1 |
| 8 | 5 | 0 | 0 | 3 | 0 | 0 |
| 9 | 0 | 2 | 4 | 3 | 2 | 0 |

## Prior identities: l2

| seed | prior_count | both | only_one | only_half | neither | switched |
|---|---|---|---|---|---|---|
| 0 | 23 | 22 | 0 | 0 | 1 | 0 |
| 1 | 26 | 25 | 1 | 0 | 0 | 1 |
| 2 | 36 | 36 | 0 | 0 | 0 | 0 |
| 3 | 27 | 27 | 0 | 0 | 0 | 0 |
| 4 | 27 | 27 | 0 | 0 | 0 | 0 |
| 5 | 26 | 26 | 0 | 0 | 0 | 0 |
| 6 | 26 | 25 | 1 | 0 | 0 | 1 |
| 7 | 25 | 25 | 0 | 0 | 0 | 0 |
| 8 | 31 | 31 | 0 | 0 | 0 | 0 |
| 9 | 20 | 19 | 1 | 0 | 0 | 1 |

Edge totals: {"half_only_true": 22, "half_only_false": 21, "one_only_true": 19, "one_only_false": 44, "half_only_false_on_prior_paths": 18, "half_only_false_off_prior_paths": 3}

Prior totals: {"prior_count": 267, "both": 263, "only_one": 3, "only_half": 0, "neither": 1, "switched": 3}

Metric change totals: {"delta_F1": 0.29172373730504825, "delta_DAG_SHD": -22, "delta_FP": -23, "delta_FN": -3, "delta_TP": 3, "delta_reversed_edges": -4, "delta_predicted_edges": -20, "delta_skeleton_extra": -19, "delta_skeleton_missing": 1, "delta_satisfaction_pct": -12.69230769230768}

## Interpretation limits

These are thresholded final graphs. They establish graph differences and associations with supplied-prior paths, not a causal mechanism or when differences arose during training. No total augmented objectives or unthresholded data-fit comparisons are made.


## Findings

Likelihood mean DAG SHD is independently reproduced as 19.6 (beta=1) and 28.8 (beta=0.5). F1 improves in four seeds, worsens in five, and ties in one; SHD likewise improves in four, worsens in five, and ties in one. Seed 3 contributes +93 of the net +92 SHD edits (81.6% of gross increases of 114). Seed 4 contributes +13; the other seeds collectively offset part of these increases. Seed 3 expands from 31 to 141 predicted edges, with FP 16->121, TP 15->20, reversals 1->8, and SHD 30->123. It remains a DAG. Across all seeds, TP rises by 12 and FP rises by 108; excess false adjacencies, not an aggregate loss of true edges, dominate deterioration. SHD decomposes into +104 extra skeleton pairs, -16 missing skeleton pairs, and +4 reversals, totaling +92.

The edge-difference ledger has 31 newly recovered true edges, 19 removed true edges, 161 added false edges, and 53 removed false edges. Of the 161 beta=0.5-only false edges, 127 (78.88%) lie on at least one supplied-prior path and 34 (21.12%) lie on none. All 20 predicted graphs are DAGs for each loss, and no runs were excluded.

Likelihood satisfaction is nearly equal, but 12 of the 267 prior instances change identity: 253 are satisfied by both, six only at beta=1, six only at beta=0.5, and two by neither. Thus both versions satisfy 259 pooled instances, while their mean per-seed rates differ slightly because seed denominators differ.

L2 F1 improves from 0.5812 to 0.6104 in the mean (six seed improvements, four declines). It gains 22 true edges and loses 19: net +3 TP. It adds 21 false edges and removes 44: net -23 FP. The improvement is chiefly false-edge removal with a smaller true-edge recovery contribution. L2 SHD decreases 22.3->20.1; satisfaction decreases 99.57%->98.30%. Three priors are lost and none newly gained.

As a labeled sensitivity description, retaining nine seeds other than seed 3 gives likelihood means F1 0.6587->0.6749 and SHD 18.4444->18.3333. The primary reported means still use all ten seeds; this is not a reason to discard seed 3. No training timing or causal optimization mechanism is inferred.
