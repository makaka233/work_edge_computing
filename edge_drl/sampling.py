"""Shared temporal sampling mechanics; never scales instantaneous KKT work."""
def interval_seconds(window_seconds, sampled_seconds, hold_cap=0.):
    if window_seconds <= 0 or sampled_seconds < 0 or hold_cap < 0:
        raise ValueError("invalid temporal sampling configuration")
    hold = 1.0
    if 0 < sampled_seconds < window_seconds:
        hold = window_seconds / float(sampled_seconds)
        if hold_cap > 0:
            hold = min(hold, hold_cap)
    return max(hold, 1.0)
