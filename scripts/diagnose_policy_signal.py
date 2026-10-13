"""Bounded, isolated diagnostic; does not change user checkpoints or processes."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_dual_ppo import build_env, rollout
from edge_drl.agents.drl import HierarchicalPPOAgent


def main():
    torch.set_num_threads(2)
    root = Path(sys.argv[1])
    args = Namespace(**json.loads((root / "metadata.json").read_text(encoding="utf-8"))["args"])
    args.device = "cpu"
    args.sampled_seconds_per_window = 2
    args.sampling_max_interval_seconds = 300
    torch.manual_seed(104)
    np.random.seed(104)
    env = build_env(args)
    env.reset()
    agent = HierarchicalPPOAgent.from_env(
        env, replicas_per_stage=32, training_repair="batch-credit-v1",
        fast_chain_context="chain-kkt-v3", slow_allocation_mode="budget-continuous-v2",
        slow_deployment_context="deployment-aware-v2", fast_k_epochs=4,
        fast_lr=args.fast_lr, fast_minibatch_size=64,
    )
    saved = torch.load(root / "checkpoints/latest.pt", map_location="cpu", weights_only=False)
    agent.fast_agent.ppo.policy.load_state_dict(saved["fast_agent"])
    agent.fast_agent.ppo.load_value_normalization_state(saved["fast_value_normalization"])
    for name in ("count", "stage", "placement"):
        getattr(agent.slow_agent, name + "_ppo").policy.load_state_dict(saved["slow_" + name + "_agent"])
    for window in range(3):
        rollout(env, agent, max_requests=0, args=args, rollout_unit="window", reset_env=window == 0,
                reward_scale=10, record_slow=False)
    source = agent.fast_agent.ppo
    states = torch.as_tensor(np.stack(source.buffer.states))
    masks = torch.as_tensor(np.stack(source.buffer.masks))
    actions = torch.as_tensor(source.buffer.actions)
    adv, returns = source._gae_advantages_and_returns()
    normalized = (adv - adv.mean()) / max(adv.std(), 1e-8)
    distribution, value = source.policy(states, masks)
    probabilities = distribution.probs.detach()
    loss = -(distribution.log_prob(actions) * torch.as_tensor(normalized)).mean()
    loss.backward()
    grad = sum(float(p.grad.square().sum()) for name, p in source.policy.named_parameters()
               if not name.startswith("critic.") and p.grad is not None) ** .5
    report = {"checkpoint_update": saved["metadata"]["update"], "samples": len(adv),
              "return_std": float(returns.std()), "actor_gradient_norm": grad,
              "return_position_correlation": float(np.corrcoef(returns, states[:, source.policy.global_dim - 6].numpy())[0, 1])}
    for name, guarded in (("full_rollout_step", True), ("minibatch_steps", False), ("atomic_guarded_minibatches", True)):
        trial = deepcopy(source)
        trial.guarded_epochs = guarded
        trial.guarded_minibatch_epochs = name == "atomic_guarded_minibatches"
        trial.critic_extra_epochs = 0
        metrics = trial.update()
        with torch.no_grad():
            dist = trial.policy(states, masks)[0]
            ratio = (dist.log_prob(actions) - distribution.log_prob(actions).detach()).exp()
            report[name] = {"steps": metrics["optimizer_steps"], "kl": metrics["approx_kl"],
                            "tv": float((dist.probs - probabilities).abs().sum(dim=1).mean() / 2),
                            "surrogate_gain": float(((ratio - 1) * torch.as_tensor(normalized)).mean())}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
