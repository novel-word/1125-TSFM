import numpy as np
import pandas as pd

from tsfm_rolling.adapters.base import (
    DEFAULT_QUANTILE_LEVELS,
    quantile_column_name,
    validate_quantile_levels,
)


def rolling_forecast(model, history_df: pd.DataFrame, test_df: pd.DataFrame, cfg: dict):
    """Predict H points with quantiles, then append actual observations.

    If the final test remainder is shorter than H, only that remainder is predicted.
    """
    dcfg = cfg["data"]
    fcfg = cfg["forecast"]

    time_col = dcfg["timestamp_col"]
    target_col = dcfg["target_col"]
    horizon = int(fcfg["horizon"])
    update_with_actual = bool(fcfg.get("update_with_actual", True))
    quantile_levels, _ = validate_quantile_levels(
        fcfg.get("quantiles", DEFAULT_QUANTILE_LEVELS),
        model_name="rolling_forecast",
    )

    if horizon < 1:
        raise ValueError("forecast.horizon must be >= 1")

    history_values = history_df[target_col].to_numpy(dtype=np.float32)
    history_dates = pd.Series(pd.to_datetime(history_df[time_col])).reset_index(drop=True)
    test_values = test_df[target_col].to_numpy(dtype=np.float32)
    test_dates = pd.Series(pd.to_datetime(test_df[time_col])).reset_index(drop=True)

    rows = []
    for round_id, start in enumerate(range(0, len(test_values), horizon), start=1):
        end = min(start + horizon, len(test_values))
        current_h = end - start

        point_pred, quantile_pred = model.predict_with_quantiles(
            context_values=history_values,
            context_dates=history_dates,
            horizon=current_h,
            freq=dcfg["freq"],
            quantile_levels=quantile_levels,
        )
        point_pred = np.asarray(point_pred, dtype=np.float32).reshape(-1)
        quantile_pred = np.asarray(quantile_pred, dtype=np.float32)
        expected_shape = (current_h, len(quantile_levels))

        if point_pred.size != current_h:
            raise RuntimeError(
                f"{cfg['model']['name']} returned {point_pred.size} point values, "
                f"but this round requires {current_h}"
            )
        if quantile_pred.shape != expected_shape:
            raise RuntimeError(
                f"{cfg['model']['name']} returned quantile shape {quantile_pred.shape}, "
                f"expected {expected_shape}"
            )
        if not np.isfinite(point_pred).all() or not np.isfinite(quantile_pred).all():
            raise RuntimeError(f"{cfg['model']['name']} returned NaN/Inf predictions")

        actual = test_values[start:end]
        dates = test_dates.iloc[start:end]

        for i in range(current_h):
            row = {
                "round": round_id,
                "timestamp": dates.iloc[i],
                "y_true": float(actual[i]),
                "y_pred": float(point_pred[i]),
            }
            for j, level in enumerate(quantile_levels):
                row[quantile_column_name(level)] = float(quantile_pred[i, j])
            rows.append(row)

        # Preserve the existing evaluation protocol: append actuals by default.
        append_values = actual if update_with_actual else point_pred
        history_values = np.concatenate(
            [history_values, append_values.astype(np.float32)]
        )
        history_dates = pd.concat(
            [history_dates, dates.reset_index(drop=True)],
            ignore_index=True,
        )

    columns = ["round", "timestamp", "y_true", "y_pred"] + [
        quantile_column_name(level) for level in quantile_levels
    ]
    return pd.DataFrame(rows, columns=columns)
