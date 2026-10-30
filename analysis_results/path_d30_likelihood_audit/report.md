# Saved-result audit: exist-path, d=30, likelihood

Only saved JSON artifacts were read. Algorithms and experiment files were not modified; no fitting was run.

## Comparability and limitations

All three methods have seeds 0--9, valid predicted DAGs, valid evaluations, and exactly matching B_true, W_true, raw X, X_std, and supplied prior lists per seed. All supplied path priors are true in the ground-truth graph. Exact per-run edge thresholds and code revisions are not saved. The source default is 0.3 and saved nonzero weights are compatible with this cutoff, but equal historical thresholds cannot be independently confirmed. Method identities are taken from the designated folders; the archived MTL logs record L2 values near the 0.1 gate, but an immutable source snapshot proving the exact historical penalty/gate implementation is not stored with these results.

No unthresholded final weight matrices were found in the JSON artifacts. W_est_prior_ll is a thresholded estimate; source thresholds W_est before returning it. Objective components below are exclusively post-threshold calculations, not final training-objective evidence. Prior loss uses the explicitly assumed defaults threshold=0.3, epsilon=0.1, sharpness=50, tau=1. Raw likelihood is 0.5 sum_j log(mean_n (X-XW)^2_j) - log|det(I-W)|, with centered X_std. No augmented or normalized total objectives are compared.

## Counting convention

Directed exact-match TP=|E_est intersect E_true|, FP=|E_est minus E_true|, FN=|E_true minus E_est|, F1=2TP/(2TP+FP+FN). A reversed edge contributes one FP and one FN under these definitions; reversed_edges is a descriptive subset of errors, never added again to FP/FN. Satisfaction is directed reachability in the saved thresholded graph. Edge-difference counts are directed. On-path membership uses all-path reachability in verified DAGs (i reaches edge source and edge target reaches j), not shortest paths.

## Method means

| method | n | F1 | TP | FP | FN | reversed_edges | predicted_edges | satisfaction_pct |
|---|---|---|---|---|---|---|---|---|
| adam_mtl | 10 | 0.6420 | 18.6000 | 9.9000 | 11.4000 | 1.7000 | 28.5000 | 97.2077 |
| adam_soft | 10 | 0.6697 | 18.0000 | 5.7000 | 12.0000 | 3.0000 | 23.7000 | 75.0077 |
| lbfgsb_soft | 10 | 0.6853 | 18.4000 | 5.1000 | 11.6000 | 2.7000 | 23.5000 | 75.4906 |

## Per-seed metrics: adam_mtl

| seed | F1 | TP | FP | FN | reversed_edges | predicted_edges | satisfied_priors | prior_count | satisfaction_pct |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.6667 | 19 | 8 | 11 | 2 | 27 | 22 | 23 | 95.6522 |
| 1 | 0.5714 | 18 | 15 | 12 | 2 | 33 | 24 | 26 | 92.3077 |
| 2 | 0.4000 | 13 | 22 | 17 | 4 | 35 | 33 | 36 | 91.6667 |
| 3 | 0.4918 | 15 | 16 | 15 | 1 | 31 | 26 | 27 | 96.2963 |
| 4 | 0.6667 | 18 | 6 | 12 | 1 | 24 | 27 | 27 | 100.0000 |
| 5 | 0.6780 | 20 | 9 | 10 | 4 | 29 | 26 | 26 | 100.0000 |
| 6 | 0.6102 | 18 | 11 | 12 | 1 | 29 | 25 | 26 | 96.1538 |
| 7 | 0.8214 | 23 | 3 | 7 | 1 | 26 | 25 | 25 | 100.0000 |
| 8 | 0.7143 | 20 | 6 | 10 | 1 | 26 | 31 | 31 | 100.0000 |
| 9 | 0.8000 | 22 | 3 | 8 | 0 | 25 | 20 | 20 | 100.0000 |

## Per-seed metrics: adam_soft

| seed | F1 | TP | FP | FN | reversed_edges | predicted_edges | satisfied_priors | prior_count | satisfaction_pct |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.6296 | 17 | 7 | 13 | 4 | 24 | 17 | 23 | 73.9130 |
| 1 | 0.8679 | 23 | 0 | 7 | 0 | 23 | 21 | 26 | 80.7692 |
| 2 | 0.5769 | 15 | 7 | 15 | 4 | 22 | 24 | 36 | 66.6667 |
| 3 | 0.6296 | 17 | 7 | 13 | 3 | 24 | 16 | 27 | 59.2593 |
| 4 | 0.6667 | 19 | 8 | 11 | 5 | 27 | 25 | 27 | 92.5926 |
| 5 | 0.7857 | 22 | 4 | 8 | 2 | 26 | 24 | 26 | 92.3077 |
| 6 | 0.7037 | 19 | 5 | 11 | 3 | 24 | 20 | 26 | 76.9231 |
| 7 | 0.6250 | 15 | 3 | 15 | 1 | 18 | 13 | 25 | 52.0000 |
| 8 | 0.5818 | 16 | 9 | 14 | 4 | 25 | 25 | 31 | 80.6452 |
| 9 | 0.6296 | 17 | 7 | 13 | 4 | 24 | 15 | 20 | 75.0000 |

## Per-seed metrics: lbfgsb_soft

| seed | F1 | TP | FP | FN | reversed_edges | predicted_edges | satisfied_priors | prior_count | satisfaction_pct |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.6545 | 18 | 7 | 12 | 3 | 25 | 18 | 23 | 78.2609 |
| 1 | 0.8679 | 23 | 0 | 7 | 0 | 23 | 21 | 26 | 80.7692 |
| 2 | 0.5882 | 15 | 6 | 15 | 3 | 21 | 25 | 36 | 69.4444 |
| 3 | 0.6296 | 17 | 7 | 13 | 3 | 24 | 16 | 27 | 59.2593 |
| 4 | 0.7368 | 21 | 6 | 9 | 4 | 27 | 26 | 27 | 96.2963 |
| 5 | 0.8214 | 23 | 3 | 7 | 2 | 26 | 24 | 26 | 92.3077 |
| 6 | 0.7037 | 19 | 5 | 11 | 3 | 24 | 20 | 26 | 76.9231 |
| 7 | 0.5417 | 13 | 5 | 17 | 2 | 18 | 14 | 25 | 56.0000 |
| 8 | 0.6296 | 17 | 7 | 13 | 4 | 24 | 25 | 31 | 80.6452 |
| 9 | 0.6792 | 18 | 5 | 12 | 3 | 23 | 13 | 20 | 65.0000 |

## Paired directed edge changes: MTL versus Adam Soft

MTL-only true = added true edges; MTL-only false = added false edges. Soft-only true = removed true edges; Soft-only false = removed false edges.

| seed | MTL_only_true | MTL_only_false | Soft_only_true | Soft_only_false | TP_change | FP_change |
|---|---|---|---|---|---|---|
| 0 | 4 | 4 | 2 | 3 | 2 | 1 |
| 1 | 3 | 15 | 8 | 0 | -5 | 15 |
| 2 | 4 | 19 | 6 | 4 | -2 | 15 |
| 3 | 5 | 14 | 7 | 5 | -2 | 9 |
| 4 | 3 | 4 | 4 | 6 | -1 | -2 |
| 5 | 3 | 7 | 5 | 2 | -2 | 5 |
| 6 | 4 | 9 | 5 | 3 | -1 | 6 |
| 7 | 8 | 1 | 0 | 1 | 8 | 0 |
| 8 | 5 | 2 | 1 | 5 | 4 | -3 |
| 9 | 7 | 1 | 2 | 5 | 5 | -4 |

Totals: {"MTL_only_true": 46, "MTL_only_false": 76, "Soft_only_true": 40, "Soft_only_false": 34, "TP_change": 6, "FP_change": 42}

## Newly satisfied path priors and location of MTL-only false edges

| seed | newly_satisfied_priors | true_only_realizations | false_edge_necessary | false_direct_added | MTL_only_false | on_prior_paths | off_prior_paths |
|---|---|---|---|---|---|---|---|
| 0 | 5 | 3 | 2 | 1 | 4 | 4 | 0 |
| 1 | 5 | 3 | 2 | 1 | 15 | 15 | 0 |
| 2 | 12 | 2 | 10 | 3 | 19 | 18 | 1 |
| 3 | 11 | 6 | 5 | 3 | 14 | 14 | 0 |
| 4 | 2 | 2 | 0 | 0 | 4 | 4 | 0 |
| 5 | 2 | 1 | 1 | 1 | 7 | 7 | 0 |
| 6 | 6 | 6 | 0 | 0 | 9 | 9 | 0 |
| 7 | 12 | 12 | 0 | 0 | 1 | 1 | 0 |
| 8 | 6 | 5 | 1 | 2 | 2 | 2 | 0 |
| 9 | 5 | 5 | 0 | 0 | 1 | 1 | 0 |

Totals: {"newly_satisfied_priors": 66, "true_only_realizations": 45, "false_edge_necessary": 21, "false_direct_added": 11, "MTL_only_false": 76, "on_prior_paths": 75, "off_prior_paths": 1}

Each newly satisfied prior is listed in newly_satisfied_priors.csv. Each MTL-only false edge and all supplied priors it can help realize are listed in mtl_only_false_edge_locations.csv.

## Post-threshold components (not training objectives)

| seed | method | raw_data_loss | sum_abs_W | h | prior_loss_assumed_t03_eps01_beta50_tau1 |
|---|---|---|---|---|---|
| 6 | adam_mtl | -10.1315 | 21.1049 | 0.0000 | 0.0998 |
| 2 | adam_mtl | -9.4095 | 38.1976 | -0.0000 | 0.0929 |
| 9 | adam_mtl | -6.5892 | 14.3466 | 0.0000 | 0.0986 |
| 5 | adam_mtl | -10.5723 | 18.2114 | 0.0000 | 0.0623 |
| 8 | adam_mtl | -14.0847 | 18.7329 | 0.0000 | 0.0133 |
| 1 | adam_mtl | -0.7439 | 38.2891 | 0.0000 | 0.0868 |
| 4 | adam_mtl | -9.8714 | 17.1599 | 0.0000 | 0.0266 |
| 7 | adam_mtl | -8.2838 | 17.5011 | 0.0000 | 0.0878 |
| 3 | adam_mtl | -8.1963 | 26.5095 | 0.0000 | 0.0989 |
| 0 | adam_mtl | -10.5512 | 17.8541 | 0.0000 | 0.1007 |
| 6 | adam_soft | -15.1304 | 16.0459 | 0.0000 | 0.8199 |
| 2 | adam_soft | -19.0006 | 16.3975 | 0.0000 | 1.7290 |
| 9 | adam_soft | -7.5976 | 13.2171 | 0.0000 | 0.7865 |
| 5 | adam_soft | -15.2690 | 17.5210 | 0.0000 | 0.7071 |
| 8 | adam_soft | -16.9975 | 17.7587 | 0.0000 | 0.7361 |
| 1 | adam_soft | -14.9233 | 14.7815 | 0.0000 | 1.1270 |
| 4 | adam_soft | -18.9729 | 18.7944 | 0.0000 | 0.8281 |
| 7 | adam_soft | -8.7914 | 11.2979 | 0.0000 | 1.3389 |
| 3 | adam_soft | -16.0187 | 16.1543 | 0.0000 | 1.3434 |
| 0 | adam_soft | -15.1129 | 18.0952 | 0.0000 | 0.6724 |
| 1 | lbfgsb_soft | -14.9226 | 14.7857 | 0.0000 | 1.1261 |
| 2 | lbfgsb_soft | -18.7989 | 16.2572 | 0.0000 | 1.6613 |
| 3 | lbfgsb_soft | -16.0186 | 16.1546 | 0.0000 | 1.3433 |
| 7 | lbfgsb_soft | -8.6299 | 11.9479 | 0.0000 | 1.2235 |
| 6 | lbfgsb_soft | -15.1295 | 16.0502 | 0.0000 | 0.8190 |
| 9 | lbfgsb_soft | -7.5394 | 13.1314 | 0.0000 | 0.8110 |
| 4 | lbfgsb_soft | -19.1325 | 19.0064 | 0.0000 | 0.6832 |
| 5 | lbfgsb_soft | -15.5647 | 17.7701 | 0.0000 | 0.6843 |
| 0 | lbfgsb_soft | -15.0750 | 17.1887 | 0.0000 | 0.5766 |
| 8 | lbfgsb_soft | -17.0740 | 17.6263 | 0.0000 | 0.7943 |

## F1 changes and limits

| seed | F1_MTL | F1_Soft | delta |
|---|---|---|---|
| 0 | 0.6667 | 0.6296 | 0.0370 |
| 1 | 0.5714 | 0.8679 | -0.2965 |
| 2 | 0.4000 | 0.5769 | -0.1769 |
| 3 | 0.4918 | 0.6296 | -0.1378 |
| 4 | 0.6667 | 0.6667 | 0.0000 |
| 5 | 0.6780 | 0.7857 | -0.1077 |
| 6 | 0.6102 | 0.7037 | -0.0935 |
| 7 | 0.8214 | 0.6250 | 0.1964 |
| 8 | 0.7143 | 0.5818 | 0.1325 |
| 9 | 0.8000 | 0.6296 | 0.1704 |

No conclusion about whether errors arose early or late in training, or about unthresholded final data fit, follows from these artifacts.


## Findings

MTL mean F1 is 0.6420 versus 0.6697 for Adam Soft and 0.6853 for L-BFGS-B Soft. Across seeds it adds 46 true and 76 false directed edges relative to Adam Soft, while removing 40 true and 34 false edges: net +6 TP but +42 FP. Therefore the aggregate deficit is primarily excess false edges; the five losing seeds individually show both net TP losses and FP increases (combined -12 TP, +50 FP). Four seeds improve, five worsen, and one ties. Seeds 1--3 account for 75.2% of the total negative F1 differences before improvements offset them.

Of 66 newly satisfied prior instances, 45 (68.18%) have a true-edge-only realization and 21 (31.82%) necessarily use false edges. Eleven (16.67%) also have a newly added false direct edge; these categories are not mutually exclusive. All ten seeds have at least one newly satisfied prior. Of 76 MTL-only false edges, 75 (98.68%) lie on at least one supplied-prior path and one (1.32%) lies on none. Membership is a structural association, not proof the optimizer added an edge for that purpose.

Saved thresholded matrices show worse average raw likelihood data fit for MTL (-8.8434 versus -14.7814 for Adam Soft), but no conclusion about final unthresholded training data fit is possible. Thresholding can change data fit substantially. No total augmented-objective comparison or training-timing inference is made.
