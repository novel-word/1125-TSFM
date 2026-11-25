import numpy as np

from .base import (
    DEFAULT_QUANTILE_LEVELS,
    ForecastAdapter,
    as_numpy,
    one_dimensional,
    require_path,
    validate_quantile_levels,
    validate_quantile_matrix,
)


class TiRexAdapter(ForecastAdapter):
    """NX-AI original TiRex with native nine-quantile forecasts."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str,
        backend: str = "torch",
        compile_model: bool = False,
        context_limit: int = 8192,
    ):
        from tirex.models.tirex import TiRexZero

        path = require_path(checkpoint_path)
        if path.is_dir():
            path = path / "model.ckpt"
        if not path.exists():
            raise FileNotFoundError(f"TiRex checkpoint file not found: {path}")

        self.model = TiRexZero.from_pretrained(
            str(path),
            device=device,
            backend=backend,
            compile=bool(compile_model),
        )
        self.model.eval()
        self.device = device
        self.context_limit = int(context_limit)

    def _forecast(self, context_values, horizon):
        import torch

        values = one_dimensional(context_values)[-self.context_limit:]
        context = torch.as_tensor(values, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            native_quantiles, mean = self.model.forecast(
                context=context,
                prediction_length=int(horizon),
                output_type="numpy",
            )

        quantiles = as_numpy(native_quantiles)
        point = as_numpy(mean).reshape(-1)
        if quantiles.ndim != 3 or quantiles.shape[0] != 1:
            raise RuntimeError(f"Unexpected TiRex quantile output shape: {quantiles.shape}")
        return point[:horizon], quantiles[0, :horizon, :]

    def predict_quantiles(
        self, context_values, context_dates, horizon, freq,
        quantile_levels=DEFAULT_QUANTILE_LEVELS,
    ):
        levels, _ = validate_quantile_levels(
            quantile_levels,
            supported_levels=DEFAULT_QUANTILE_LEVELS,
            model_name="TiRex",
        )
        _, quantiles = self._forecast(context_values, horizon)
        return validate_quantile_matrix(quantiles, horizon, levels, "TiRex")

    def predict_with_quantiles(
        self, context_values, context_dates, horizon, freq,
        quantile_levels=DEFAULT_QUANTILE_LEVELS,
    ):
        # Preserve the previous implementation's y_pred = forecast mean.
        levels, _ = validate_quantile_levels(
            quantile_levels,
            supported_levels=DEFAULT_QUANTILE_LEVELS,
            model_name="TiRex",
        )
        point, quantiles = self._forecast(context_values, horizon)
        quantiles = validate_quantile_matrix(quantiles, horizon, levels, "TiRex")
        if point.size != int(horizon) or not np.isfinite(point).all():
            raise RuntimeError("TiRex returned an invalid point forecast")
        return point, quantiles
