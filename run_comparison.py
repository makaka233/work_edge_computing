from __future__ import annotations

import argparse

from edge_drl.comparison.runner import ACCEPTED_SCHEMES, FORMAL_SCENARIO_FAMILIES, FORMAL_SCHEMES, run_comparison
from edge_drl.comparison.greedy import BASELINE_IMPLEMENTATIONS, GREEDY_IMPLEMENTATION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated IEEE scheme comparisons")
    parser.add_argument("--checkpoint", required=True, help="Explicit Proposed best.pt checkpoint")
    parser.add_argument(
        "--mso-checkpoint", "--monolithic-checkpoint",
        dest="monolithic_checkpoint",
        default=None,
        help="Explicit separately trained MSO PPO checkpoint",
    )
    parser.add_argument(
        "--unified-checkpoint",
        default=None,
        help="Explicit independently trained UnifiedPPO checkpoint",
    )
    parser.add_argument(
        "--mso-checkpoint-mode", "--monolithic-checkpoint-mode",
        dest="monolithic_checkpoint_mode",
        choices=("fixed", "per-point"),
        default="fixed",
        help=(
            "Use one frozen MSO checkpoint across every scenario (formal default), "
            "or explicitly opt into exploratory per-point checkpoints"
        ),
    )
    parser.add_argument("--phase", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", default="results/comparison")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--phase2-validation-run",
        default=None,
        help="Required for Phase 3; path to a successfully completed Phase 2 result directory",
    )
    parser.add_argument("--schemes", nargs="+", choices=ACCEPTED_SCHEMES, default=list(FORMAL_SCHEMES))
    parser.add_argument(
        "--baseline-implementation", choices=BASELINE_IMPLEMENTATIONS,
        default=GREEDY_IMPLEMENTATION,
        help="DARP/DPMS design version; legacy-v1 reproduces the archived MILP/path schemes",
    )
    parser.add_argument(
        "--eval-seeds",
        nargs="+",
        type=int,
        default=None,
        help="Optional evaluation-only seed override; does not alter formal phase defaults",
    )
    parser.add_argument(
        "--scenario-families",
        nargs="+",
        choices=FORMAL_SCENARIO_FAMILIES,
        default=None,
        help="Optional subset of the five scenario families",
    )
    parser.add_argument(
        "--sampled-seconds-per-window",
        type=int,
        default=0,
        help=(
            "Evaluation temporal sampling budget per deployment window; 0 preserves "
            "the formal second-by-second default"
        ),
    )
    parser.add_argument(
        "--learned-fast-eval-mode",
        choices=("deterministic", "stochastic"),
        default="deterministic",
        help="Action mode for learned Fast policies; stochastic runs use explicit routing seeds",
    )
    parser.add_argument(
        "--routing-repeats",
        type=int,
        default=None,
        help="Optional number of independently seeded routing repeats per trace",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_comparison(
        checkpoint=args.checkpoint,
        phase=args.phase,
        device=args.device,
        output_root=args.output_root,
        run_id=args.run_id,
        schemes=tuple(args.schemes),
        baseline_implementation=args.baseline_implementation,
        phase2_validation_run=args.phase2_validation_run,
        monolithic_checkpoint=args.monolithic_checkpoint,
        monolithic_checkpoint_mode=args.monolithic_checkpoint_mode,
        unified_checkpoint=args.unified_checkpoint,
        eval_seeds_override=None if args.eval_seeds is None else tuple(args.eval_seeds),
        scenario_families=None if args.scenario_families is None else tuple(args.scenario_families),
        sampled_seconds_per_window=args.sampled_seconds_per_window,
        learned_fast_eval_mode=args.learned_fast_eval_mode,
        routing_repeats_override=args.routing_repeats,
    )
    print(f"comparison results: {output.resolve()}")


if __name__ == "__main__":
    main()
