from .base import (
    DEFAULT_QUANTILE_LEVELS,
    ForecastAdapter,
    as_numpy,
    one_dimensional,
    require_path,
    validate_quantile_levels,
    validate_quantile_matrix,
)


class TiRex2Adapter(ForecastAdapter):
    """NX-AI official TiRex-2 loader and native quantile forecast API."""

    def __init__(self, checkpoint_path: str, device: str, context_limit: int):
        from tirex2 import load_model

        path = require_path(checkpoint_path)
        if not path.is_dir():
            raise ValueError(
                "TiRex-2 checkpoint_path must be a directory containing "
                "config.json and model.safetensors"
            )

        self.model = load_model(str(path), device=device, compile=False)
        self.context_limit = int(context_limit)

    def predict_quantiles(
        self, context_values, context_dates, horizon, freq,
        quantile_levels=DEFAULT_QUANTILE_LEVELS,
    ):
        import torch
        from tirex2 import TimeseriesType

        levels, _ = validate_quantile_levels(
            quantile_levels,
            supported_levels=DEFAULT_QUANTILE_LEVELS,
            model_name="TiRex-2",
        )
        values = one_dimensional(context_values)[-self.context_limit:]
        target = torch.as_tensor(values, dtype=torch.float32).unsqueeze(0)
        ts = TimeseriesType(
            target=target,
            past_covariates=None,
            future_covariates=None,
        )
        forecasts = self.model.forecast(
            [ts],
            prediction_length=int(horizon),
            output_type="numpy",
        )
        # Official output for one univariate target: [1, 9 quantiles, horizon].
        native = as_numpy(forecasts[0])
        if native.ndim != 3 or native.shape[0] != 1 or native.shape[1] != 9:
            raise RuntimeError(f"Unexpected TiRex-2 output shape: {native.shape}")
        quantiles = native[0, :, :horizon].T
        return validate_quantile_matrix(quantiles, horizon, levels, "TiRex-2")
