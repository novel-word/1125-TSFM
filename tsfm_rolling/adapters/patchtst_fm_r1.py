import numpy as np
import pandas as pd

from .base import ForecastAdapter, one_dimensional, require_path, resolve_torch_device


class PatchTSTFMR1Adapter(ForecastAdapter):
    """IBM Granite official PatchTST-FM-R1 model implementation."""

    def __init__(self, checkpoint_path: str, device: str, context_limit: int):
        import torch
        from tsfm_public.models.patchtst_fm import PatchTSTFMForPrediction

        path = require_path(checkpoint_path)
        if not path.is_dir():
            raise ValueError(
                "PatchTST-FM-R1 checkpoint_path must be a local model directory"
            )

        self.device = resolve_torch_device(torch, device)
        self.model = PatchTSTFMForPrediction.from_pretrained(
            str(path),
            local_files_only=True,
        ).to(self.device).eval()

        model_context = int(self.model.config.context_length)
        self.model_context = model_context
        self.context_limit = min(int(context_limit), model_context)

    def predict(self, context_values, context_dates, horizon, freq):
        import torch

        if horizon >= self.model_context:
            raise ValueError(
                f"PatchTST-FM-R1 requires context + horizon <= "
                f"{self.model_context}; got horizon={horizon}"
            )

        # In this checkpoint, config.context_length is the full input window,
        # including the forecast section. Keep room for the requested horizon.
        max_history = min(self.context_limit, self.model_context - int(horizon))
        values = one_dimensional(context_values)[-max_history:]
        past_values = torch.as_tensor(
            values, dtype=torch.float32, device=self.device
        ).view(1, -1, 1)

        with torch.inference_mode():
            output = self.model(
                past_values=past_values,
                prediction_length=int(horizon),
                quantile_levels=[0.5],
                return_dict=True,
            )

        # Official quantile output: [batch, quantile, horizon, variate].
        median = output.quantile_outputs[0, 0, :, 0]
        return median.detach().float().cpu().numpy()[:horizon]
