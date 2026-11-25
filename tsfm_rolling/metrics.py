import numpy as np


def calculate_metrics(y_true, y_pred, y_quantiles=None, quantile_levels=None) -> dict:
    """Calculate legacy point metrics and, when supplied, probabilistic metrics."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if y_true.size == 0 or y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must be non-empty arrays of equal length")

    error = y_true - y_pred
    abs_error = np.abs(error)
    mse = float(np.mean(error ** 2))

    nz = np.abs(y_true) > 1e-12
    mape = (
        float(np.mean(abs_error[nz] / np.abs(y_true[nz])) * 100)
        if nz.any() else None
    )

    denom = np.abs(y_true) + np.abs(y_pred)
    valid = denom > 1e-12
    smape = (
        float(np.mean(2 * abs_error[valid] / denom[valid]) * 100)
        if valid.any() else None
    )

    metrics = {
        "MAE": float(np.mean(abs_error)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAPE_percent": mape,
        "sMAPE_percent": smape,
        "n": int(y_true.size),
    }

    if y_quantiles is None:
        return metrics
    if quantile_levels is None:
        raise ValueError("quantile_levels is required when y_quantiles is provided")

    levels = np.asarray(quantile_levels, dtype=np.float64).reshape(-1)
    quantiles = np.asarray(y_quantiles, dtype=np.float64)
    if quantiles.shape != (y_true.size, levels.size):
        raise ValueError(
            f"y_quantiles must have shape {(y_true.size, levels.size)}, "
            f"got {quantiles.shape}"
        )
    if levels.size == 0 or np.any(levels <= 0) or np.any(levels >= 1):
        raise ValueError("quantile_levels must be non-empty and strictly between 0 and 1")
    if np.any(np.diff(levels) <= 0):
        raise ValueError("quantile_levels must be unique and increasing")
    if not np.isfinite(quantiles).all():
        raise ValueError("y_quantiles contains NaN/Inf values")
    if not np.any(np.isclose(levels, 0.5, atol=1e-8)):
        raise ValueError("quantile_levels must include 0.5")

    # Pinball loss: rho_q(y - qhat) = max(q * error, (q - 1) * error).
    quantile_error = y_true[:, None] - quantiles
    pinball = np.maximum(
        levels[None, :] * quantile_error,
        (levels[None, :] - 1.0) * quantile_error,
    )
    mean_pinball_by_level = np.mean(pinball, axis=0)
    for level, value in zip(levels, mean_pinball_by_level):
        metrics[f"PinballLoss_q{int(round(level * 100)):02d}"] = float(value)
    metrics["MeanPinballLoss"] = float(np.mean(pinball))

    # Mean weighted quantile loss, normalized by the mean absolute target scale.
    target_scale = float(np.sum(np.abs(y_true)))
    metrics["WQL"] = (
        float(2.0 * np.sum(pinball) / (levels.size * target_scale))
        if target_scale > 1e-12 else None
    )

    median_index = int(np.argmin(np.abs(levels - 0.5)))
    median = quantiles[:, median_index]
    interval_scores = []
    wis_numerator = 0.5 * np.abs(y_true - median)

    # Symmetric deciles provide central 80%, 60%, 40% and 20% intervals.
    for lower_index, lower_level in enumerate(levels):
        if lower_level >= 0.5:
            continue
        upper_level = 1.0 - lower_level
        matches = np.flatnonzero(np.isclose(levels, upper_level, atol=1e-8))
        if matches.size == 0:
            continue

        upper_index = int(matches[0])
        lower = quantiles[:, lower_index]
        upper = quantiles[:, upper_index]
        alpha = 2.0 * float(lower_level)
        nominal_coverage = int(round((1.0 - alpha) * 100))

        below = y_true < lower
        above = y_true > upper
        interval_score = (
            (upper - lower)
            + (2.0 / alpha) * (lower - y_true) * below
            + (2.0 / alpha) * (y_true - upper) * above
        )
        coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
        width = float(np.mean(upper - lower))

        metrics[f"PICP_{nominal_coverage}"] = coverage
        metrics[f"MPIW_{nominal_coverage}"] = width
        metrics[f"IntervalScore_{nominal_coverage}"] = float(np.mean(interval_score))

        # Weighted Interval Score combines the median and all central intervals.
        wis_numerator += (alpha / 2.0) * interval_score
        interval_scores.append(interval_score)

    if interval_scores:
        metrics["WIS"] = float(np.mean(wis_numerator / (len(interval_scores) + 0.5)))

    # Share of adjacent quantile pairs that cross (q_(i+1) < q_i).
    if levels.size > 1:
        metrics["QuantileCrossingRate"] = float(
            np.mean(np.diff(quantiles, axis=1) < 0.0)
        )

    return metrics
