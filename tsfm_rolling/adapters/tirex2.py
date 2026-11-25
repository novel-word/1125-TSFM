import numpy as np
import pandas as pd

from .base import ForecastAdapter, one_dimensional, require_path


class TiRex2Adapter(ForecastAdapter):
    """NX-AI official TiRex-2 loader and forecast API."""

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

    def predict(self, context_values, context_dates, horizon, freq):
        import torch
        from tirex2 import TimeseriesType

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
        # Official shape for one univariate target: [1, 9 quantiles, horizon].
        q = np.asarray(forecasts[0])
        if q.ndim != 3:
            raise RuntimeError(f"Unexpected TiRex-2 output shape: {q.shape}")
        return q[0, 4, :horizon].astype(np.float32)
