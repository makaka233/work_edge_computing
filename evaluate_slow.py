"""Compare learned Slow deployment with DARP v2 using one frozen Fast."""
import argparse
import os

from edge_drl.comparison.slow_components import evaluate_slow_components


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fast-checkpoint', required=True)
    parser.add_argument('--slow-checkpoints', nargs='+', default=[])
    parser.add_argument('--quota-decoders', nargs='+', default=['checkpoint'],
                        choices=['checkpoint','mean','integer-mode','categorical-mode','categorical-mean'])
    parser.add_argument('--seeds', default='2030,2031,2032')
    parser.add_argument('--loads', default='0.7,1.0,1.35')
    parser.add_argument('--episode-minutes', type=int, default=60)
    parser.add_argument('--sampled-seconds-per-window', type=int, default=6)
    parser.add_argument('--sampling-max-interval-seconds', type=float, default=100.)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    import torch
    torch.set_num_threads(args.threads)
    output, report = evaluate_slow_components(fast_checkpoint=args.fast_checkpoint,
        slow_checkpoints=args.slow_checkpoints, quota_decoders=args.quota_decoders,
        seeds=tuple(int(x) for x in args.seeds.split(',')), loads=tuple(float(x) for x in args.loads.split(',')),
        minutes=args.episode_minutes, samples=args.sampled_seconds_per_window,
        hold_cap=args.sampling_max_interval_seconds, device=args.device, output=args.output)
    print(f'Results: {output}')
    if report['failed_evaluations']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
