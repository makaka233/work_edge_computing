from __future__ import annotations

import argparse

from edge_drl.comparison.runner import FORMAL_SCHEMES, run_comparison


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
    )
    print(f"comparison results: {output.resolve()}")


if __name__ == "__main__":
    main()
