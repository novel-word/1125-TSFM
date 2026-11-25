import numpy as np
import pandas as pd


def rolling_forecast(model, history_df: pd.DataFrame, test_df: pd.DataFrame, cfg: dict):
    """Predict H points, then append those H ground-truth observations.

    If the final test remainder is shorter than H, only that remainder is predicted.
    """
    dcfg = cfg["data"]
    fcfg = cfg["forecast"]

    time_col = dcfg["timestamp_col"]
    target_col = dcfg["target_col"]
    horizon = int(fcfg["horizon"])
    update_with_actual = bool(fcfg.get("update_with_actual", True))

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

        pred = model.predict(
            context_values=history_values,
            context_dates=history_dates,
            horizon=current_h,
            freq=dcfg["freq"],
        )
        pred = np.asarray(pred, dtype=np.float32).reshape(-1)

        if pred.size != current_h:
            raise RuntimeError(
                f"{cfg['model']['name']} returned {pred.size} values, "
                f"but this round requires {current_h}"
            )
        if not np.isfinite(pred).all():
            raise RuntimeError(f"{cfg['model']['name']} returned NaN/Inf predictions")

        actual = test_values[start:end]
        dates = test_dates.iloc[start:end]

        for i in range(current_h):
            rows.append({
                "round": round_id,
                "timestamp": dates.iloc[i],
                "y_true": float(actual[i]),
                "y_pred": float(pred[i]),
            })

        # User-requested evaluation: append ground truth, not the previous prediction.
        append_values = actual if update_with_actual else pred
        history_values = np.concatenate(
            [history_values, append_values.astype(np.float32)]
        )
        history_dates = pd.concat(
            [history_dates, dates.reset_index(drop=True)],
            ignore_index=True,
        )

    return pd.DataFrame(rows)
