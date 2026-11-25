import numpy as np


def calculate_metrics(y_true, y_pred) -> dict:
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

    return {
        "MAE": float(np.mean(abs_error)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAPE_percent": mape,
        "sMAPE_percent": smape,
        "n": int(y_true.size),
    }
