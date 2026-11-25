from .base import (
    DEFAULT_QUANTILE_LEVELS,
    ForecastAdapter,
    as_numpy,
    one_dimensional,
    require_path,
    validate_quantile_levels,
    validate_quantile_matrix,
)


class TimesFM30Adapter(ForecastAdapter):
    """Google official TimesFM 3.0 evaluator API."""

    def __init__(self, checkpoint_path: str, batch_size: int, device: str, context_limit: int):
        from timesfm3 import TimesFM3Evaluator, ModelConfig

        path = require_path(checkpoint_path)
        if not path.is_dir():
            raise ValueError("TimesFM-3.0 checkpoint_path must be a local checkpoint directory")

        self.context_limit = min(int(context_limit), 15360)
        config = ModelConfig(
            checkpoint_path=str(path),
            per_core_batch_size=int(batch_size),
            device=device,
        )
        self.forecaster = TimesFM3Evaluator(config)

    def predict_quantiles(
        self, context_values, context_dates, horizon, freq,
        quantile_levels=DEFAULT_QUANTILE_LEVELS,
    ):
        levels, _ = validate_quantile_levels(
            quantile_levels,
            supported_levels=DEFAULT_QUANTILE_LEVELS,
            model_name="TimesFM-3.0",
        )
        values = one_dimensional(context_values)[-self.context_limit:]
        outputs = list(
            self.forecaster.predict_batch(
                [values],
                horizon=int(horizon),
                return_quantiles=True,
                use_symmetric_averaging=False,
            )
        )
        if not outputs:
            raise RuntimeError("TimesFM-3.0 returned an empty prediction list")
        quantiles = as_numpy(outputs[0].quantiles)
        return validate_quantile_matrix(quantiles, horizon, levels, "TimesFM-3.0")
