# 1125-TSFM

A rolling-evaluation framework for comparing time-series foundation models.

## Point and probabilistic forecasts

The runner preserves the original point-forecast output (`y_pred`) and point metrics
(MAE, MSE, RMSE, MAPE and sMAPE), and additionally exports probabilistic forecasts
at nine quantile levels: 0.1 through 0.9.

Run the experiment as before:

```bash
python run_multiple_models.py --config config_multiple_models.yaml
```

Each model writes:

- `<model>/<model>_predictions.csv`: `round`, `timestamp`, `y_true`,
  backward-compatible `y_pred`, and `q10` through `q90`.
- `<model>/<model>_metrics.json`: point and probabilistic metrics.
- `summary.json` and `summary.csv`: model-level comparison results.

## Probabilistic metrics

The evaluation includes per-quantile Pinball Loss, Mean Pinball Loss, normalized
Weighted Quantile Loss (WQL), Weighted Interval Score (WIS), prediction interval
coverage (PICP), mean prediction interval width (MPIW), interval score, and
quantile crossing rate.

For the decile grid, symmetric central intervals are 80%, 60%, 40% and 20%.
WQL is defined here as twice the sum of pinball losses divided by the number of
quantiles and the sum of absolute observations. WIS combines the median absolute
error and the available central interval scores.

## Model adapters

Adapters return a common `[forecast_horizon, number_of_quantiles]` matrix.
Native quantile grids are selected where available; no synthetic Gaussian
prediction intervals are constructed. Model APIs and installed checkpoint
versions should be validated with a real inference run before comparing results.

The default rolling evaluation protocol is unchanged: by default, each completed
forecast block is followed by the corresponding actual observations in history.
