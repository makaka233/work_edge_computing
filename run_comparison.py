from __future__ import annotations

import argparse

from edge_drl.comparison.runner import FORMAL_SCENARIO_FAMILIES, FORMAL_SCHEMES, run_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated IEEE scheme comparisons")
    parser.add_argument("--checkpoint", required=True, help="Explicit Proposed best.pt checkpoint")
    parser.add_argument(
        "--monolithic-checkpoint",
        default=None,
        help="Explicit separately trained Monolithic PPO checkpoint",
    )
    parser.add_argument(
        "--monolithic-checkpoint-mode",
        choices=("fixed", "per-point"),
        default="fixed",
        help=(
            "Use one frozen Monolithic checkpoint across every scenario (formal default), "
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
    parser.add_argument("--schemes", nargs="+", choices=FORMAL_SCHEMES, default=list(FORMAL_SCHEMES))
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
        phase2_validation_run=args.phase2_validation_run,
        monolithic_checkpoint=args.monolithic_checkpoint,
        monolithic_checkpoint_mode=args.monolithic_checkpoint_mode,
        eval_seeds_override=None if args.eval_seeds is None else tuple(args.eval_seeds),
        scenario_families=None if args.scenario_families is None else tuple(args.scenario_families),
        sampled_seconds_per_window=args.sampled_seconds_per_window,
    )
    print(f"comparison results: {output.resolve()}")


if __name__ == "__main__":
    main()
