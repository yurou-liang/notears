"""Saved graph audit only: beta=1 versus beta=0.5, exist-path d=30."""
from pathlib import Path
import csv, json, re, hashlib, subprocess
from collections import defaultdict
import numpy as np
import networkx as nx

OUT=Path(__file__).resolve().parent
FOLDERS={'1':'linear_exist_path_pairs_modified_mtl','0.5':'linear_exist_path_pairs'}

def write(name,rows):
 if not rows:return
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def table(rows,cols):
 def val(v):return f'{v:.4f}' if isinstance(v,float) else str(v)
 return '\n'.join(['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']+['| '+' | '.join(val(r[c]) for c in cols)+' |' for r in rows])

def edge_set(B):return {(int(i),int(j)) for i,j in zip(*np.where(B))}
def graph(E):
 g=nx.DiGraph();g.add_nodes_from(range(30));g.add_edges_from(E);return g

def reach(g):
 R=np.eye(30,dtype=bool)
 for i in g:
  for j in nx.descendants(g,i):R[i,j]=True
 return R

def skeleton(E):return {tuple(sorted(e)) for e in E}
def sha(v):return hashlib.sha256(json.dumps(v,separators=(',',':')).encode()).hexdigest()

runs={};manifest=[];audit=[];metrics=[];missing=[]
for beta,folder in FOLDERS.items():
 for p in sorted(Path(folder).glob('*_d30_*epsilon0.1_*_mtl_adam.json')):
  if 'compare' in p.name:continue
  seed=int(re.search(r'_seed(\d+)',p.name)[1]);assert (beta,seed) not in runs
  r=json.loads(p.read_text());assert len(r['B_true'])==30
  manifest.append(dict(file=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
  run={'raw':r,'file':str(p)};runs[beta,seed]=run
  for loss in ['ll','l2']:
   T=edge_set(np.asarray(r['B_true'])!=0);W=np.asarray(r.get('W_est_prior_'+loss),dtype=float)
   matrix_valid=W.shape==(30,30) and np.isfinite(W).all()
   E=edge_set(W!=0) if matrix_valid else set();g=graph(E);cyclic=not nx.is_directed_acyclic_graph(g)
   acc=r.get('acc_prior_'+loss);valid=matrix_valid and not cyclic and isinstance(acc,dict) and 'f1' in acc
   priors=[tuple(p) for p in r['prior_knowledge_'+loss].get('exist_path_pairs',[])];R=reach(g)
   run[loss]=dict(T=T,E=E,R=R,g=g,valid=valid,cyclic=cyclic,priors=priors)
   minimum=float(np.min(np.abs(W[W!=0]))) if matrix_valid and np.any(W!=0) else None
   audit.append(dict(beta=beta,seed=seed,loss=loss,matrix_valid=matrix_valid,cyclic=cyclic,valid_evaluation=valid,status=json.dumps(r.get('evaluation_status',{}).get('prior_'+loss)),optimizer_success=r.get('sol_success_'+loss),minimum_nonzero=minimum))
   if not valid:continue
   tp=len(E&T);fp=len(E-T);fn=len(T-E);rev=sum((j,i) in T for i,j in E-T)
   extra=len(skeleton(E)-skeleton(T));absent=len(skeleton(T)-skeleton(E));shd=extra+absent+rev
   sat=sum(R[i,j] for i,j in priors);precision=tp/len(E) if E else 0;recall=tp/len(T) if T else 0;f1=2*tp/(len(E)+len(T))
   assert abs(f1-acc['f1'])<1e-10 and shd==acc['shd'],(beta,seed,loss)
   assert abs(100*sat/len(priors)-r['satisfied_percentage_prior_'+loss])<1e-8
   metrics.append(dict(beta=beta,seed=seed,loss=loss,F1=f1,precision=precision,recall=recall,TP=tp,FP=fp,FN=fn,reversed_edges=rev,predicted_edges=len(E),DAG_SHD=shd,skeleton_extra=extra,skeleton_missing=absent,satisfied=int(sat),prior_count=len(priors),satisfaction_pct=100*sat/len(priors)))
 for seed in range(10):
  if (beta,seed) not in runs:missing.append(dict(beta=beta,seed=seed))

comparability=[]
for seed in range(10):
 if any((b,seed) not in runs for b in FOLDERS):continue
 a,b=[runs[x,seed]['raw'] for x in FOLDERS]
 checks={f+'_equal':a.get(f)==b.get(f) for f in ['B_true','W_true','X','X_std','prior_knowledge_ll','prior_knowledge_l2']}
 assert all(checks.values()),(seed,checks)
 comparability.append(dict(seed=seed,**checks))

lookup={(r['beta'],r['seed'],r['loss']):r for r in metrics}
summary=[];changes=[];diffs=[];locations=[];prior_changes=[];prior_details=[];edge_details=[]
matched={}
for loss in ['ll','l2']:
 seeds=[s for s in range(10) if all((b,s,loss) in lookup for b in FOLDERS)];matched[loss]=seeds
 for beta in FOLDERS:
  rows=[lookup[beta,s,loss] for s in seeds]
  summary.append(dict(beta=beta,loss=loss,n=len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in ['F1','precision','recall','TP','FP','FN','reversed_edges','predicted_edges','DAG_SHD','skeleton_extra','skeleton_missing','satisfaction_pct']}))
 for seed in seeds:
  one=runs['1',seed][loss];half=runs['0.5',seed][loss];T=one['T'];A=one['E'];B=half['E']
  changes.append(dict(seed=seed,loss=loss,**{'delta_'+k:lookup['0.5',seed,loss][k]-lookup['1',seed,loss][k] for k in ['F1','DAG_SHD','FP','FN','TP','reversed_edges','predicted_edges','skeleton_extra','skeleton_missing','satisfaction_pct']}))
  only_half=B-A;only_one=A-B;false_half=only_half-T
  on=[];off=[]
  for u,v in sorted(false_half):
   pairs=[(i,j) for i,j in half['priors'] if half['R'][i,u] and half['R'][v,j]]
   (on if pairs else off).append((u,v))
   locations.append(dict(seed=seed,loss=loss,u=u,v=v,on_prior_path=bool(pairs),prior_pairs=json.dumps(pairs)))
  diffs.append(dict(seed=seed,loss=loss,half_only_true=len(only_half&T),half_only_false=len(false_half),one_only_true=len(only_one&T),one_only_false=len(only_one-T),half_only_false_on_prior_paths=len(on),half_only_false_off_prior_paths=len(off)))
  for kind,E in [('half_only',only_half),('one_only',only_one)]:
   for i,j in sorted(E):edge_details.append(dict(seed=seed,loss=loss,kind=kind,i=i,j=j,true_edge=(i,j) in T,reversal=(j,i) in T))
  counts=defaultdict(int)
  for i,j in one['priors']:
   x,y=bool(one['R'][i,j]),bool(half['R'][i,j]);status='both' if x and y else 'only_one' if x else 'only_half' if y else 'neither';counts[status]+=1
   prior_details.append(dict(seed=seed,loss=loss,i=i,j=j,status=status))
  n=len(one['priors']);prior_changes.append(dict(seed=seed,loss=loss,prior_count=n,**{k:counts[k] for k in ['both','only_one','only_half','neither']},switched=counts['only_one']+counts['only_half']))

# Configuration evidence from current source and its Git baseline, not historical provenance.
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
diff=subprocess.check_output(['git','diff','--','prior_notears/linear.py'],text=True)
(OUT/'current_source_vs_HEAD.diff').write_text(diff)
source=Path('prior_notears/linear.py').read_text()
config={
 'interpretation':'beta is the multiplier on the normalized prior direction, not Softplus sharpness or Adam momentum beta',
 'current_source_HEAD':head,
 'current_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
 'current_source':{'softplus':'c_i = softplus(residual, sharpness)' in source,'original_gate':'prior_active = float(L2 > 0.1)' in source,'prior_multiplier_half':'0.5 * a2 * prior_active * (g2 / n2)' in source,'defaults':{'epsilon':0.1,'sharpness':50,'adam_lr':0.001,'adam_max_steps':10000,'inner_param_tol':1e-5,'w_threshold':0.3,'max_iter':100}},
 'historical_provenance_limit':'JSONs contain no beta/config/lr/budget/threshold metadata or immutable source revision. Current source vs HEAD changes only the prior multiplier in scalar and direction; this confirms the intended code change, not that every archived run used that exact source and defaults.',
 'filename_evidence':{'epsilon':0.1,'tau':1.0,'d':30},
 'missing_runs':missing,'matched_valid_seeds':matched,
}
log_summary=[]
for beta,folder in FOLDERS.items():
 for seed in range(10):
  matches=list(Path(folder,'log').glob(f'*_d30_*epsilon0.1_seed{seed}_tau1.0_mtl_adam.log'))
  if not matches:
   log_summary.append(dict(beta=beta,seed=seed,present=False));continue
  p=matches[0];t=p.read_text();steps=[int(x) for x in re.findall(r'Adam \[.*?steps=(\d+)',t)]
  log_summary.append(dict(beta=beta,seed=seed,present=True,has_10000_step_solve=10000 in steps,max_observed_steps=max(steps) if steps else None,contains_parameter_stability='parameter_stable' in t,completed='Results saved to:' in t,traceback='Traceback' in t))
config['log_summary']=log_summary
(OUT/'configuration_evidence.json').write_text(json.dumps(config,indent=2)+'\n')
(OUT/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
write('comparability.csv',comparability);write('validity.csv',audit);write('per_seed_metrics.csv',metrics);write('method_summary.csv',summary);write('per_seed_changes.csv',changes);write('edge_differences.csv',diffs);write('half_only_false_edge_locations.csv',locations);write('edge_difference_details.csv',edge_details);write('prior_changes.csv',prior_changes);write('prior_change_details.csv',prior_details)

report='''# Exist-path d=30: saved beta=1 versus beta=0.5 results\n\nRead-only graph analysis; no training or algorithm changes. beta denotes the multiplier on the normalized prior direction, not Softplus sharpness or Adam momentum.\n\n## Comparability\n\nAll ten seeds have matching ground-truth adjacency/weights, raw X, X_std, and prior lists for both losses. All twenty runs have valid saved DAG evaluations for both losses; no cycles or missing runs were found. Current linear.py uses Softplus, the original L2>0.1 gate, and a 0.5 prior multiplier. Its Git diff against HEAD changes only that multiplier in the scalar and direction. Defaults are epsilon=0.1, sharpness=50, Adam lr=0.001, max inner updates=10000, max outer iterations=100, threshold=0.3, and parameter-stability tolerance=1e-5 checked over 100-update windows with patience three. The available current shell script passes epsilon but does not override lr/budget/threshold. Filenames record epsilon=0.1 and tau=1. Logs show 10000-update solves. These records support beta as the only intended change. However, the JSONs do not store beta, lr, threshold, budget, or source revision. Historical effective settings and beta as the only actual change cannot be independently certified; current defaults are not immutable run provenance. See configuration_evidence.json and current_source_vs_HEAD.diff.\n\n## Counting\n\nExact directed TP=|E intersect T|, FP=|E minus T|, FN=|T minus E|. Precision=TP/(TP+FP), recall=TP/(TP+FN), F1=2TP/(2TP+FP+FN). Reversed edges are predicted i->j with true j->i: one FP and one FN, reported separately without being added again. DAG SHD follows NOTEARS: extra skeleton pairs + missing skeleton pairs + reversed-edge count. A reversal costs one DAG SHD edit. This is not CPDAG SHD. Satisfaction is directed reachability of supplied priors in saved thresholded graphs. A beta=0.5-only false edge u->v is on a prior path iff some supplied i->j has i reaching u and v reaching j in the beta=0.5 DAG, including zero-length endpoint segments. This covers all paths without enumerating them.\n\n## Means on matching seeds\n\n'''
report+=table(summary,['loss','beta','n','F1','precision','recall','TP','FP','FN','reversed_edges','predicted_edges','DAG_SHD','satisfaction_pct'])
for loss in ['ll','l2']:
 for beta in FOLDERS:
  report+=f'\n\n## Per-seed metrics: {loss}, beta={beta}\n\n'+table([r for r in metrics if r['loss']==loss and r['beta']==beta],['seed','F1','precision','recall','TP','FP','FN','reversed_edges','predicted_edges','DAG_SHD','satisfied','prior_count','satisfaction_pct'])
 report+=f'\n\n## Changes at beta=0.5 minus beta=1: {loss}\n\n'+table([r for r in changes if r['loss']==loss],['seed','delta_F1','delta_DAG_SHD','delta_FP','delta_FN','delta_reversed_edges','delta_predicted_edges','delta_skeleton_extra','delta_skeleton_missing'])
 report+=f'\n\n## Directed edge differences: {loss}\n\n'+table([r for r in diffs if r['loss']==loss],['seed','half_only_true','half_only_false','one_only_true','one_only_false','half_only_false_on_prior_paths','half_only_false_off_prior_paths'])
 report+=f'\n\n## Prior identities: {loss}\n\n'+table([r for r in prior_changes if r['loss']==loss],['seed','prior_count','both','only_one','only_half','neither','switched'])
 drows=[r for r in diffs if r['loss']==loss];prows=[r for r in prior_changes if r['loss']==loss];crows=[r for r in changes if r['loss']==loss]
 totals={k:sum(r[k] for r in drows) for k in drows[0] if k not in ['seed','loss']}
 ptot={k:sum(r[k] for r in prows) for k in prows[0] if k not in ['seed','loss']}
 delta={k:sum(r[k] for r in crows) for k in crows[0] if k not in ['seed','loss']}
 report+='\n\nEdge totals: '+json.dumps(totals)+'\n\nPrior totals: '+json.dumps(ptot)+'\n\nMetric change totals: '+json.dumps(delta)
 print(loss,'EDGE',totals,'PRIOR',ptot,'DELTA',delta)
 print('F1 improve/worse/tie',sum(r['delta_F1']>1e-10 for r in crows),sum(r['delta_F1']< -1e-10 for r in crows),sum(abs(r['delta_F1'])<=1e-10 for r in crows))
report+='\n\n## Interpretation limits\n\nThese are thresholded final graphs. They establish graph differences and associations with supplied-prior paths, not a causal mechanism or when differences arose during training. No total augmented objectives or unthresholded data-fit comparisons are made.\n'
(OUT/'report.md').write_text(report)
print('SUMMARY',summary)
print('LIKELIHOOD CHANGES',[r for r in changes if r['loss']=='ll'])
