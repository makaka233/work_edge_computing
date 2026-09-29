from __future__ import annotations

import argparse

from edge_drl.comparison.kkt_ablation import run_kkt_action_replay
from edge_drl.comparison.runner import FORMAL_SCENARIO_FAMILIES


def parse_args():
    parser = argparse.ArgumentParser(description="Replay frozen Proposed actions under KKT and equal-share allocation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--phase", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", default="results/kkt_comparison")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--scenario-families", nargs="+", choices=FORMAL_SCENARIO_FAMILIES, default=None)
    parser.add_argument("--sampled-seconds-per-window", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    output = run_kkt_action_replay(
        checkpoint=args.checkpoint, phase=args.phase, device=args.device,
        output_root=args.output_root, run_id=args.run_id,
        eval_seeds_override=None if args.eval_seeds is None else tuple(args.eval_seeds),
        scenario_families=None if args.scenario_families is None else tuple(args.scenario_families),
        sampled_seconds_per_window=args.sampled_seconds_per_window,
    )
    print(f"KKT comparison results: {output.resolve()}")


if __name__ == "__main__":
    main()
