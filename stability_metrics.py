import numpy as np

def compute_stability(Z):
    """
    Z: numpy array of shape [T, 3]
    returns: dict of stability metrics
    """

    # ---------- 1. VARIANCE ----------
    # High variance = unstable
    variance = np.var(Z, axis=0)  # [3]

    # ---------- 2. MEAN ABSOLUTE DELTA ----------
    # Frame-to-frame volatility
    delta = np.diff(Z, axis=0)
    mad = np.mean(np.abs(delta), axis=0)  # [3]

    # ---------- 3. BASELINE DRIFT ----------
    # Distance from initial emotional state
    baseline = Z[0]
    drift = np.mean(np.linalg.norm(Z - baseline, axis=1))

    # ---------- 4. AGGREGATE STABILITY SCORE ----------
    # Lower = more stable
    stability_score = (
        variance.mean() +
        mad.mean() +
        drift
    )

    return {
        "variance": variance,
        "mean_abs_delta": mad,
        "drift": drift,
        "stability_score": stability_score
    }
