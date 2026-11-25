import numpy as np
import pandas as pd

from .base import ForecastAdapter, one_dimensional, require_path


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

    def predict(self, context_values, context_dates, horizon, freq):
        values = one_dimensional(context_values)[-self.context_limit:]
        forecast = self.forecaster.forecast(
            values,
            prediction_length=int(horizon),
        )
        return forecast.median.detach().float().cpu().numpy().reshape(-1)[:horizon]
