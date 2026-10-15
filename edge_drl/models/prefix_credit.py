"""Terminal-return prefix GAE for a composite Slow deployment action.

All critic targets remain the REAL window return. No proxy rewards or expert
labels. Lambda < 1 trades bias for variance; lambda=1 is exactly MC R-V.
"""
import numpy as np


def prefix_advantages(returns, baselines, window_ids, trace_lambda):
    returns = np.asarray(returns, dtype=np.float32)
    baselines = np.asarray(baselines, dtype=np.float32)
    ids = np.asarray(window_ids)
    if returns.ndim != 1 or baselines.shape != returns.shape or ids.shape != returns.shape:
        raise ValueError("prefix returns, baselines and windows must be aligned")
    if not 0 <= trace_lambda <= 1:
        raise ValueError("trace lambda must be within [0,1]")
    result = np.empty_like(returns)
    for window in np.unique(ids):
        indices = np.flatnonzero(ids == window)
        if not np.allclose(returns[indices], returns[indices[0]]):
            raise ValueError("prefix credit requires one real terminal return per window")
        advantage = 0.
        for position in range(len(indices)-1, -1, -1):
            i = indices[position]
            following = returns[i] if position == len(indices)-1 else baselines[indices[position+1]]
            delta = following - baselines[i]
            advantage = delta + trace_lambda * advantage
            result[i] = advantage
    return result
