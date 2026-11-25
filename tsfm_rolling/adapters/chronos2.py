import numpy as np
import pandas as pd

from .base import ForecastAdapter, one_dimensional, require_path


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

    def predict(self, context_values, context_dates, horizon, freq):
        values = one_dimensional(context_values)[-self.context_limit:]
        dates = pd.to_datetime(pd.Series(context_dates)).reset_index(drop=True)
        dates = dates.iloc[-len(values):].reset_index(drop=True)

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
            quantile_levels=[0.5],
            context_length=self.context_limit,
            freq=freq,
        )

        if "predictions" in forecast_df.columns:
            pred = forecast_df["predictions"].to_numpy(dtype=np.float32)
        elif "0.5" in forecast_df.columns:
            pred = forecast_df["0.5"].to_numpy(dtype=np.float32)
        else:
            raise KeyError(
                "Chronos-2 output has neither 'predictions' nor '0.5' column: "
                f"{forecast_df.columns.tolist()}"
            )
        return pred[:horizon]
