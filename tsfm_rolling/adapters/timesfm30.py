import numpy as np
import pandas as pd

from .base import ForecastAdapter, one_dimensional, require_path


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

    def predict(self, context_values, context_dates, horizon, freq):
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
        return np.asarray(outputs[0].forecast, dtype=np.float32).reshape(-1)[:horizon]
