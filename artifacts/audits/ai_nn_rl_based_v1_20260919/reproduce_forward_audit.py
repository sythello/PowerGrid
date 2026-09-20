import json, hashlib
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from powergrid.ai.nn_rl_based.model import NumpyRlPolicyQNetwork, build_policy_targets
from powergrid.ai.nn_rl_based.dataset import iter_rl_parquet_batches
from powergrid.ai.nn_rl_based.training import _rows_to_arrays
root=Path('/Users/mac/Desktop/syt/Projects/PowerGrid')
out={'checkpoints':{},'labels':{}}
for p in [root/'src/powergrid/data/ai_models/ai_nn_rl_based_v1.npz', *sorted((root/'artifacts/experiments').glob('*/models/*.npz'))]:
 m=NumpyRlPolicyQNetwork.load(p); d=m.metadata
 keys=['training_dataset','training_seed','training_epochs','training_batch_decisions','learning_rate','loss_weights','policy_target_mode','improved_action_weight','min_search_advantage','training_sampling','training_sampling_source_counts','source_search_configuration','final_train_metrics','final_validation_metrics','training_sampling_epoch_counts','release_status','training_iteration']
 out['checkpoints'][str(p.relative_to(root))]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),**{k:d.get(k) for k in keys}}
for folder in ['paired_mc_pi_20260918/data_medium_500','paired_mc_confirm_20260919/data_smoke_k0','paired_mc_confirm_20260919/data_smoke_k8','paired_mc_confirm_20260919/data_smoke_k16']:
 p=root/'artifacts/experiments'/folder
 splits={}
 for split in ['train','validation','test']:
  counter=Counter(); phases=defaultdict(Counter)
  for batch in iter_rl_parquet_batches(p,split,batch_size=2048,columns=('has_search_targets','teacher_action_index','candidate_action_features','search_q_values','search_policy_advantages','search_policy_confirmed','decision_type','game_id')):
   for row in batch.to_pylist():
    counter['rows']+=1
    if not row['has_search_targets']:continue
    counter['searched']+=1
    acts=np.asarray(row['candidate_action_features'],dtype=np.float32); teacher=row['teacher_action_index']; n=len(acts)
    adv=row.get('search_policy_advantages') or None; conf=row.get('search_policy_confirmed') or None
    target,accepted,_=build_policy_targets(np.array([0,n]),np.array([teacher]),np.array([True]),np.asarray(row['search_q_values']),acts,policy_target_mode='advantage_weighted',search_policy_advantages=adv,search_policy_confirmed=conf)
    if accepted[0]:
     counter['accepted']+=1; phases[row['decision_type']]['accepted']+=1
     if np.argmax(target)==teacher:counter['accepted_target_argmax_is_teacher']+=1
     if float(target[teacher])>=float(max(np.delete(target,teacher))):counter['accepted_target_teacher_at_least_joint_max']+=1
     counter['positive_target_actions']+=int(np.sum(target>0))-1
     if len(set(tuple(a) for i,a in enumerate(acts) if target[i]>0 and i!=teacher))<int(np.sum(target>0))-1:counter['accepted_with_duplicate_improved_features']+=1
  splits[split]={'counts':dict(counter),'by_type':{k:dict(v) for k,v in phases.items()}}
 out['labels'][folder]=splits
 print(folder,json.dumps(splits),flush=True)
# One representative validation set: compare the incumbent and K16 on every state.
p=root/'artifacts/experiments/paired_mc_confirm_20260919/data_smoke_k16'
inc=NumpyRlPolicyQNetwork.load(root/'src/powergrid/data/ai_models/ai_nn_rl_based_v1.npz')
cand=NumpyRlPolicyQNetwork.load(root/'artifacts/experiments/paired_mc_confirm_20260919/models/k16_weighted075.npz')
stats=Counter(); types=defaultdict(Counter)
for batch in iter_rl_parquet_batches(p,'validation',batch_size=512):
 rows=batch.to_pylist(); a=_rows_to_arrays(rows)
 old=inc.predict(a['states'],a['actions'],a['offsets']);new=cand.predict(a['states'],a['actions'],a['offsets'])
 targets,accept,_=build_policy_targets(a['offsets'],a['teacher'],a['searched'],a['search_q'],a['actions'],policy_target_mode='advantage_weighted',search_policy_advantages=a['search_policy_advantages'],search_policy_confirmed=a['search_policy_confirmed'])
 for i,(s,e) in enumerate(zip(a['offsets'][:-1],a['offsets'][1:])):
  b=old.policy_probabilities[s:e];c=new.policy_probabilities[s:e];t=int(a['teacher'][i]);chosen=int(np.argmax(c)); typ=rows[i]['decision_type']
  stats['decisions']+=1;types[typ]['decisions']+=1
  stats['incumbent_teacher_mismatches']+=int(np.argmax(b)!=t)
  stats['kl_old_new_sum']+=float(np.sum(b*np.log((b+1e-12)/(c+1e-12))))
  stats['hard_bc_ce_at_incumbent_sum']+=float(-np.log(b[t]+1e-12))
  stats['accepted']+=int(accept[i])
  stats['changed']+=int(chosen!=t);types[typ]['changed']+=int(chosen!=t)
  if not a['searched'][i]:
   stats['non_search']+=1;stats['non_search_changed']+=int(chosen!=t)
  if chosen!=t:
   stats['changed_on_accepted']+=int(accept[i])
   stats['changed_to_confirmed_positive']+=int(a['search_policy_confirmed'][s+chosen])
   stats['changed_q_disagrees']+=int(new.q_values[s+chosen,0]<new.q_values[s+t,0])
   stats['changed_on_same_feature']+=int(np.array_equal(a['actions'][s+chosen],a['actions'][s+t]))
out['k16_validation_forward_audit']={'totals':dict(stats),'by_type':{k:dict(v) for k,v in types.items()}}
print('k16_validation',json.dumps(out['k16_validation_forward_audit']),flush=True)
(root/'artifacts/audits/ai_nn_rl_based_v1_20260919/diagnostics.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
