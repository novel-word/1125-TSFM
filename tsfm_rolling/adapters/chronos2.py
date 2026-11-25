import numpy as np
import pandas as pd

from .base import (
    DEFAULT_QUANTILE_LEVELS,
    ForecastAdapter,
    one_dimensional,
    require_path,
    validate_quantile_levels,
    validate_quantile_matrix,
)


class Chronos2Adapter(ForecastAdapter):
    """Amazon official Chronos-2 pipeline."""

    def __init__(self, checkpoint_path: str, device: str, context_limit: int):
        from chronos import Chronos2Pipeline

        path = require_path(checkpoint_path)
        self.device = device
        self.pipeline = Chronos2Pipeline.from_pretrained(
            str(path),
            device_map=device,
            local_files_only=True,
        )

        model_context = getattr(self.pipeline, "model_context_length", None)
        self.context_limit = min(
            int(context_limit),
            int(model_context) if model_context else int(context_limit),
        )

    def predict_quantiles(
        self, context_values, context_dates, horizon, freq,
        quantile_levels=DEFAULT_QUANTILE_LEVELS,
    ):
        values = one_dimensional(context_values)[-self.context_limit:]
        dates = pd.to_datetime(pd.Series(context_dates)).reset_index(drop=True)
        dates = dates.iloc[-len(values):].reset_index(drop=True)
        levels, _ = validate_quantile_levels(quantile_levels, model_name="Chronos-2")

        df = pd.DataFrame({
            "item_id": "series_0",
            "timestamp": dates,
            "target": values,
        })
        forecast_df = self.pipeline.predict_df(
            df,
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
            prediction_length=int(horizon),
            quantile_levels=list(levels),
            context_length=self.context_limit,
            freq=freq,
        )

        quantile_columns = [str(q) for q in levels]
        missing = [column for column in quantile_columns if column not in forecast_df.columns]
        if missing:
            raise KeyError(
                f"Chronos-2 output is missing quantile columns {missing}; "
                f"available columns: {forecast_df.columns.tolist()}"
            )
        quantiles = forecast_df[quantile_columns].to_numpy(dtype=np.float32)
        return validate_quantile_matrix(quantiles, horizon, levels, "Chronos-2")
