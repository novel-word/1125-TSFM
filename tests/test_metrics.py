import numpy as np

from tsfm_rolling.metrics import calculate_metrics


def test_legacy_point_metrics_remain_available_without_quantiles():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])
    metrics = calculate_metrics(y_true, y_pred)
    assert np.isclose(metrics["MAE"], 1.0 / 3.0)
    assert metrics["RMSE"] > 0.0
    assert "WIS" not in metrics


def test_probabilistic_metrics_are_calculated_for_deciles():
    y_true = np.array([1.0, 2.0, 3.0])
    offsets = np.arange(-4.0, 5.0)
    quantiles = y_true[:, None] + offsets[None, :]
    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    metrics = calculate_metrics(
        y_true, y_true, y_quantiles=quantiles, quantile_levels=levels
    )
    assert metrics["MAE"] == 0.0
    assert metrics["PinballLoss_q10"] >= 0.0
    assert metrics["MeanPinballLoss"] >= 0.0
    assert metrics["WQL"] >= 0.0
    assert metrics["WIS"] >= 0.0
    assert metrics["PICP_80"] == 1.0
    assert metrics["MPIW_80"] == 8.0
    assert metrics["QuantileCrossingRate"] == 0.0
