"""Paired one-update runtime/coverage check; NOT a convergence experiment."""
from argparse import ArgumentParser
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    parser.add_argument('--repair',default='both',choices=['both','none','conditional-replay-v1'])
    options=parser.parse_args()
    import torch
    import train_dual_ppo as training
    from edge_drl.comparison.checkpoint import CRITICAL_ENV_KEYS
    checkpoint=torch.load(options.checkpoint,map_location='cpu')
    args=training.checkpoint_run_args(options.checkpoint,checkpoint)
    keys=['seed','pressure_profile','training_repair','slow_deployment_context','training_design','train_mode',
          'rollout_unit','episode_minutes','deployment_interval_minutes','fast_windows_per_update',
          'slow_windows_per_update','sampled_seconds_per_window','sampling_max_interval_seconds',
          'num_users','num_edge_nodes','num_service_types','slow_allocation_mode','replicas_per_stage',
          'reward_mode','reward_scale','fast_policy_kind','fast_chain_context','fast_counterfactual_credit_coef',
          'fast_counterfactual_credit_final_coef','slow_counterfactual_credit_coef','slow_replica_ablation_samples',
          'fast_oracle_diagnostic_requests','fast_k_epochs','slow_k_epochs','slow_stage_k_epochs',
          'slow_placement_k_epochs','fast_minibatch_size','slow_minibatch_size','slow_critic_k_epochs',
          'slow_critic_replay_windows','slow_critic_replay_decay']
    common=[]
    for k in keys:
        if args.get(k) is not None:
            common.extend(['--'+k.replace('_','-'),str(args[k])])
    common.extend(['--load-checkpoint',str(options.checkpoint.resolve()),'--updates','1','--device',options.device,
                   '--eval-interval','0','--run-root','runs_tmp','--progress-interval-seconds','0'])
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    reports=[]
    repairs=['none','conditional-replay-v1'] if options.repair=='both' else [options.repair]
    for repair in repairs:
        name='slow_learning_'+repair.replace('-','_')+'_paired_'+stamp
        sys.argv=['train_dual_ppo.py',*common,'--run-name',name,'--slow-learning-repair',repair]
        t=time.perf_counter()
        training.main()
        root=ROOT/'runs_tmp'/name
        rows=list(csv.DictReader((root/'logs/training.csv').open(encoding='utf-8-sig')))
        meta=json.loads((root/'metadata.json').read_text(encoding='utf-8'))['args']
        assert all(meta.get(k)==args.get(k) for k in CRITICAL_ENV_KEYS), 'physical settings changed'
        assert float(rows[-1]['invalid_actions'])==0
        reports.append({'repair':repair,'run':str(root),'wall_seconds':time.perf_counter()-t,
                        'training_row':rows[-1]})
    # The first collection precedes the changed update, so exact equality is
    # expected. This does NOT establish either policy's learning superiority.
    if len(reports)==2:
        for key in ['avg_latency_s','avg_reward','request_events','load_multiplier_mean']:
            assert reports[0]['training_row'][key]==reports[1]['training_row'][key], key
    target=ROOT/'runs_tmp'/('slow_learning_paired_'+stamp+'.json')
    target.write_text(json.dumps(reports,indent=2),encoding='utf-8')
    print('PAIRED_VERIFICATION_REPORT',target)
    print('WALL_SECONDS',[(r['repair'],r['wall_seconds']) for r in reports])


if __name__=='__main__':
    main()
