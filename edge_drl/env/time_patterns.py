from __future__ import annotations

import math


def gaussian_peak(t: float, center: float, width: float, amplitude: float) -> float:
    return amplitude * math.exp(-((t - center) ** 2) / (2.0 * width**2))


def tidal_factor(t: int, pattern: str, slots_per_day: int = 86400) -> float:
    """Return a positive request/activity multiplier for one node."""
    day_t = float(t % slots_per_day)
    hour = day_t / 3600.0

    if pattern == "business":
        value = (
            0.55
            + gaussian_peak(hour, 10.0, 2.0, 0.85)
            + gaussian_peak(hour, 15.0, 2.2, 0.70)
        )
    elif pattern == "residential":
        value = (
            0.45
            + gaussian_peak(hour, 7.5, 1.5, 0.40)
            + gaussian_peak(hour, 20.0, 2.5, 1.10)
        )
    elif pattern == "industrial":
        value = (
            0.25
            + gaussian_peak(hour, 9.5, 2.0, 1.00)
            + gaussian_peak(hour, 14.5, 2.0, 0.90)
        )
    elif pattern == "traffic":
        value = (
            0.35
            + gaussian_peak(hour, 8.0, 1.0, 1.35)
            + gaussian_peak(hour, 18.0, 1.2, 1.50)
        )
    else:
        value = 1.0 + 0.2 * math.sin(2.0 * math.pi * day_t / float(slots_per_day))

    return max(0.1, min(2.5, value))

