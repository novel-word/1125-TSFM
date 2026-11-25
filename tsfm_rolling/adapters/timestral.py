from .base import (
    DEFAULT_QUANTILE_LEVELS,
    ForecastAdapter,
    as_numpy,
    one_dimensional,
    require_path,
    validate_quantile_levels,
    validate_quantile_matrix,
)


class TimestralAdapter(ForecastAdapter):
    """Mistral official Timestral inference implementation."""

    def __init__(self, checkpoint_path: str, device: str, context_limit: int):
        from timestral import TimestralForecaster

        path = require_path(checkpoint_path)
        if not path.is_dir():
            raise ValueError(
                "Timestral checkpoint_path must be a local directory "
                "containing config.json and model.safetensors"
            )

        self.context_limit = min(int(context_limit), 32768)
        self.forecaster = TimestralForecaster.from_pretrained(
            path,
            device=device,
            max_context=self.context_limit,
            local_files_only=True,
        )

    def predict_quantiles(
        self, context_values, context_dates, horizon, freq,
        quantile_levels=DEFAULT_QUANTILE_LEVELS,
    ):
        values = one_dimensional(context_values)[-self.context_limit:]
        forecast = self.forecaster.forecast(values, prediction_length=int(horizon))

        native_levels = tuple(float(q) for q in forecast.quantile_levels)
        levels, indices = validate_quantile_levels(
            quantile_levels,
            supported_levels=native_levels,
            model_name="Timestral",
        )
        native = as_numpy(forecast.quantiles)
        if native.ndim != 4 or native.shape[0] != 1 or native.shape[1] != 1:
            raise RuntimeError(f"Unexpected Timestral quantile output shape: {native.shape}")
        # Native shape: [series, variates, horizon, quantile].
        quantiles = native[0, 0][:, list(indices)]
        return validate_quantile_matrix(quantiles, horizon, levels, "Timestral")
