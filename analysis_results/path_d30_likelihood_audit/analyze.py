"""Read-only audit of saved d30 likelihood path-prior experiments."""
from pathlib import Path
import csv, hashlib, json, re
import numpy as np
import networkx as nx
from scipy.linalg import expm

OUT=Path(__file__).resolve().parent
CONFIG={
 'adam_mtl':('linear_exist_path_pairs_modified_mtl','*_d30_*epsilon0.1_*_mtl_adam.json'),
 'adam_soft':('linear_exist_path_pairs_adam_soft','*_d30_*epsilon0.1_*.json'),
 'lbfgsb_soft':('linear_exist_path_pairs_soft','*_d30_*epsilon0.1_*_soft.json'),
}

def save(name, rows):
 if rows:
  with (OUT/name).open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def digest(v):return hashlib.sha256(json.dumps(v,separators=(',',':')).encode()).hexdigest()
def edges(B):return set(zip(*np.where(B)))
def graph(E):
 g=nx.DiGraph();g.add_nodes_from(range(30));g.add_edges_from(E);return g

def reach(g):
 a=np.eye(30,dtype=bool)
 for i in range(30):
  for j in nx.descendants(g,i):a[i,j]=True
 return a

def table(rows,cols):
 lines=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
 for r in rows:
  lines.append('| '+' | '.join(f'{r[c]:.4f}' if isinstance(r[c],float) else str(r[c]) for c in cols)+' |')
 return '\n'.join(lines)

runs={};verify=[];metric=[];components=[];details=[]
for method,(folder,pattern) in CONFIG.items():
 for p in Path(folder).glob(pattern):
  if 'compare' in p.name or 'diagnostic' in p.name:continue
  seed=int(re.search(r'_seed(\d+)',p.name)[1]);assert (method,seed) not in runs
  r=json.loads(p.read_text());B=np.asarray(r['B_true'])!=0;T=edges(B)
  W=np.asarray(r.get('W_est_prior_ll'),dtype=float)
  valid_matrix=W.shape==(30,30) and np.isfinite(W).all()
  E=edges(W!=0) if valid_matrix else set();g=graph(E);cyclic=not nx.is_directed_acyclic_graph(g)
  priors=[tuple(x) for x in r['prior_knowledge_ll'].get('exist_path_pairs',[])]
  R=reach(g);saved_acc=r.get('acc_prior_ll');eval_valid=isinstance(saved_acc,dict) and 'f1' in saved_acc
  runs[method,seed]=dict(raw=r,E=E,T=T,g=g,R=R,priors=priors,W=W,cyclic=cyclic,valid=valid_matrix and not cyclic and eval_valid)
  nonzero=np.abs(W[W!=0]) if valid_matrix else np.array([])
  verify.append(dict(method=method,seed=seed,file=str(p),matrix_valid=valid_matrix,cyclic=cyclic,evaluation_valid=eval_valid,evaluation_status=json.dumps(r.get('evaluation_status',{}).get('prior_ll')),optimizer_success=r.get('sol_success_ll'),threshold_metadata=r.get('w_threshold','not saved'),minimum_saved_nonzero=float(nonzero.min()) if len(nonzero) else None,B_true_sha256=digest(r['B_true']),X_sha256=digest(r['X']),X_std_sha256=digest(r['X_std']),priors_sha256=digest(r['prior_knowledge_ll']),unthresholded_keys=','.join(k for k in r if any(s in k.lower() for s in ['unthreshold','w_raw','w_star','w_final']))))
  if not valid_matrix:continue
  tp=len(E&T);fp=len(E-T);fn=len(T-E);rev=sum((j,i) in T for i,j in E-T)
  f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
  satisfied=sum(R[i,j] for i,j in priors);sat=100*satisfied/len(priors) if priors else float('nan')
  if eval_valid:assert abs(f1-saved_acc['f1'])<1e-10,(method,seed,f1,saved_acc)
  assert abs(sat-r['satisfied_percentage_prior_ll'])<1e-7
  metric.append(dict(seed=seed,method=method,F1=f1,TP=tp,FP=fp,FN=fn,reversed_edges=rev,predicted_edges=len(E),prior_count=len(priors),satisfied_priors=int(satisfied),satisfaction_pct=sat,cyclic=cyclic))
  # Post-threshold components only; prior surrogate parameters assumed defaults.
  X=np.asarray(r['X_std']);X=X-X.mean(axis=0,keepdims=True)
  residual_var=((X-X@W)**2).mean(axis=0);sign,ld=np.linalg.slogdet(np.eye(30)-W)
  ld_loss=float(.5*np.log(residual_var).sum()-ld) if sign and np.all(residual_var>0) else None
  A=np.logaddexp(0,50*(W*W-.3**2))/50;P=expm(A)
  prior=float(sum(np.logaddexp(0,50*(.1-P[i,j]))/50 for i,j in priors))
  components.append(dict(seed=seed,method=method,weights='saved thresholded only',raw_data_loss=ld_loss,sum_abs_W=float(np.abs(W).sum()),h=float(np.trace(expm(W*W))-30),prior_loss_assumed_t03_eps01_beta50_tau1=prior,saved_loss=r.get('loss_prior_ll'),loss_matches_saved=bool(ld_loss is not None and abs(ld_loss-r['loss_prior_ll'])<1e-8)))

comparability=[]
for seed in range(10):
 available=[m for m in CONFIG if (m,seed) in runs]
 checks={}
 for field in ['B_true','W_true','X','X_std','prior_knowledge_ll']:
  vals=[runs[m,seed]['raw'].get(field) for m in available]
  checks[field+'_exact_match']=len(vals)==3 and all(v==vals[0] for v in vals)
 comparability.append(dict(seed=seed,available_methods=','.join(available),**checks,threshold_match='not independently verifiable; no per-run threshold metadata',all_priors_true=all(runs[available[0],seed]['R'][i,j] for i,j in []) if not available else all(reach(graph(runs[available[0],seed]['T']))[i,j] for i,j in runs[available[0],seed]['priors'])))
 assert all(checks.values()),comparability[-1]

paired=[s for s in range(10) if all(runs.get((m,s),{}).get('valid',False) for m in CONFIG)]
deltas=[];prior_rows=[];locations=[];per_edge=[]
for seed in paired:
 m=runs['adam_mtl',seed];s=runs['adam_soft',seed];T=m['T'];M=m['E'];S=s['E']
 added=M-S;removed=S-M
 at=len(added&T);af=len(added-T);rt=len(removed&T);rf=len(removed-T)
 gains=[p for p in m['priors'] if m['R'][p] and not s['R'][p]]
 intersection_R=reach(graph(M&T))
 for i,j in gains:
  true_realization=bool(intersection_R[i,j]);shortcut=(i,j) in added-T
  prior_rows.append(dict(seed=seed,i=i,j=j,true_only_realization=true_realization,false_edge_necessary=not true_realization,added_false_direct_edge=shortcut))
 false_only=added-T;on=0
 for u,v in sorted(false_only):
  # In a DAG, i reaches u and v reaches j implies a simple i->...u->v...->j path.
  supported=[(i,j) for i,j in m['priors'] if m['R'][i,u] and m['R'][v,j]]
  on+=bool(supported)
  per_edge.append(dict(seed=seed,u=int(u),v=int(v),on_any_supplied_prior_path=bool(supported),prior_pairs=json.dumps(supported)))
 deltas.append(dict(seed=seed,MTL_only_true=at,MTL_only_false=af,Soft_only_true=rt,Soft_only_false=rf,TP_change=at-rt,FP_change=af-rf,F1_MTL=next(r['F1'] for r in metric if r['seed']==seed and r['method']=='adam_mtl'),F1_Soft=next(r['F1'] for r in metric if r['seed']==seed and r['method']=='adam_soft')))
 g_rows=[r for r in prior_rows if r['seed']==seed]
 locations.append(dict(seed=seed,newly_satisfied_priors=len(gains),true_only_realizations=sum(r['true_only_realization'] for r in g_rows),false_edge_necessary=sum(r['false_edge_necessary'] for r in g_rows),false_direct_added=sum(r['added_false_direct_edge'] for r in g_rows),MTL_only_false=len(false_only),on_prior_paths=on,off_prior_paths=len(false_only)-on))
 for kind,E in [('MTL_only',added),('Soft_only',removed)]:
  for i,j in sorted(E):details.append(dict(seed=seed,kind=kind,i=int(i),j=int(j),true_edge=(i,j) in T,reversal=(j,i) in T))

save('verification.csv',verify);save('comparability.csv',comparability);save('per_seed_metrics.csv',metric);save('edge_changes.csv',deltas);save('newly_satisfied_priors.csv',prior_rows);save('path_summary.csv',locations);save('mtl_only_false_edge_locations.csv',per_edge);save('edge_change_details.csv',details);save('thresholded_components.csv',components)
summary=[]
for method in CONFIG:
 rows=[r for r in metric if r['method']==method and r['seed'] in paired]
 summary.append(dict(method=method,n=len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in ['F1','TP','FP','FN','reversed_edges','predicted_edges','satisfaction_pct']}))
save('method_summary.csv',summary)
totals={k:sum(r[k] for r in deltas) for k in ['MTL_only_true','MTL_only_false','Soft_only_true','Soft_only_false','TP_change','FP_change']}
loc_totals={k:sum(r[k] for r in locations) for k in locations[0] if k!='seed'}
F=np.array([r['F1_MTL']-r['F1_Soft'] for r in deltas]);print('MATCHED',paired);print('SUMMARY',summary);print('EDGE TOTALS',totals);print('PATH TOTALS',loc_totals);print('F1_DELTAS',list(zip(paired,F.tolist())));print('NO_GAIN_SEEDS',[r['seed'] for r in locations if r['newly_satisfied_priors']==0])
# Descriptive paired replacement diagnostics, not causal attributions.
counter=[]
for seed in paired:
 m=runs['adam_mtl',seed];s=runs['adam_soft',seed];T=m['T'];M=m['E'];S=s['E']
 def f1(E):return 2*len(E&T)/(len(E)+len(T))
 counter.append(dict(seed=seed,MTL_F1=f1(M),remove_MTL_only_false_F1=f1(M-((M-S)-T)),restore_Soft_only_true_F1=f1(M|((S-M)&T))))
save('descriptive_edge_repairs.csv',counter)
report='''# Saved-result audit: exist-path, d=30, likelihood\n\nOnly saved JSON artifacts were read. Algorithms and experiment files were not modified; no fitting was run.\n\n## Comparability and limitations\n\nAll three methods have seeds 0--9, valid predicted DAGs, valid evaluations, and exactly matching B_true, W_true, raw X, X_std, and supplied prior lists per seed. All supplied path priors are true in the ground-truth graph. Exact per-run edge thresholds and code revisions are not saved. The source default is 0.3 and saved nonzero weights are compatible with this cutoff, but equal historical thresholds cannot be independently confirmed. Method identities are taken from the designated folders; logs confirm the archived MTL run is Softplus-based and ends near L2=0.1, but no immutable source snapshot is stored in JSON.\n\nNo unthresholded final weight matrices were found in the JSON artifacts. W_est_prior_ll is a thresholded estimate; source thresholds W_est before returning it. Objective components below are exclusively post-threshold calculations, not final training-objective evidence. Prior loss uses the explicitly assumed defaults threshold=0.3, epsilon=0.1, sharpness=50, tau=1. Raw likelihood is 0.5 sum_j log(mean_n (X-XW)^2_j) - log|det(I-W)|, with centered X_std. No augmented or normalized total objectives are compared.\n\n## Counting convention\n\nDirected exact-match TP=|E_est intersect E_true|, FP=|E_est minus E_true|, FN=|E_true minus E_est|, F1=2TP/(2TP+FP+FN). A reversed edge contributes one FP and one FN under these definitions; reversed_edges is a descriptive subset of errors, never added again to FP/FN. Satisfaction is directed reachability in the saved thresholded graph. Edge-difference counts are directed. On-path membership uses all-path reachability in verified DAGs (i reaches edge source and edge target reaches j), not shortest paths.\n\n## Method means\n\n'''
report+=table(summary,['method','n','F1','TP','FP','FN','reversed_edges','predicted_edges','satisfaction_pct'])
for method in CONFIG:
 report+='\n\n## Per-seed metrics: '+method+'\n\n'+table(sorted([r for r in metric if r['method']==method],key=lambda r:r['seed']),['seed','F1','TP','FP','FN','reversed_edges','predicted_edges','satisfied_priors','prior_count','satisfaction_pct'])
report+='\n\n## Paired directed edge changes: MTL versus Adam Soft\n\nMTL-only true = added true edges; MTL-only false = added false edges. Soft-only true = removed true edges; Soft-only false = removed false edges.\n\n'+table(deltas,['seed','MTL_only_true','MTL_only_false','Soft_only_true','Soft_only_false','TP_change','FP_change'])+'\n\nTotals: '+json.dumps(totals)
report+='\n\n## Newly satisfied path priors and location of MTL-only false edges\n\n'+table(locations,list(locations[0]))+'\n\nTotals: '+json.dumps(loc_totals)+'\n\nEach newly satisfied prior is listed in newly_satisfied_priors.csv. Each MTL-only false edge and all supplied priors it can help realize are listed in mtl_only_false_edge_locations.csv.\n\n## Post-threshold components (not training objectives)\n\n'+table(components,['seed','method','raw_data_loss','sum_abs_W','h','prior_loss_assumed_t03_eps01_beta50_tau1'])
report+='\n\n## F1 changes and limits\n\n'+table([dict(seed=r['seed'],F1_MTL=r['F1_MTL'],F1_Soft=r['F1_Soft'],delta=r['F1_MTL']-r['F1_Soft']) for r in deltas],['seed','F1_MTL','F1_Soft','delta'])+'\n\nNo conclusion about whether errors arose early or late in training, or about unthresholded final data fit, follows from these artifacts.\n'
(OUT/'report.md').write_text(report)
