"""Inference decoding for a latent Beta policy over executable integer quotas."""
from __future__ import annotations

import numpy as np
from scipy.special import betainc


def beta_quota_probabilities(alpha: float, beta: float, maximum: int) -> np.ndarray:
    if (not np.isfinite([alpha, beta]).all() or alpha <= 0 or beta <= 0
            or not isinstance(maximum, (int, np.integer)) or maximum < 0):
        raise ValueError("Beta parameters must be positive and maximum a nonnegative integer")
    edges = np.linspace(0.0, 1.0, maximum + 2)
    probabilities = np.maximum(np.diff(betainc(alpha, beta, edges)), 0.0)
    if not np.isfinite(probabilities).all() or probabilities.sum() <= 0:
        raise FloatingPointError("Could not decode Beta integer quota probabilities")
    return probabilities / probabilities.sum()


def beta_modal_fraction(alpha: float, beta: float, maximum: int) -> float:
    quota = int(np.argmax(beta_quota_probabilities(alpha, beta, maximum)))
    # Midpoint is strictly inside the selected bin, including both endpoints.
    return (quota + 0.5) / (maximum + 1)
