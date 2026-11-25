import numpy as np
import pandas as pd

from .base import ForecastAdapter, one_dimensional, require_path, resolve_torch_device


class Toto2Adapter(ForecastAdapter):
    """Datadog official Toto 2.0 model implementation."""

    def __init__(self, checkpoint_path: str, device: str, context_limit: int):
        import torch
        from toto2 import Toto2Model

        path = require_path(checkpoint_path)
        if not path.is_dir():
            raise ValueError(
                "Toto-2 checkpoint_path must be a local directory "
                "containing its config and weights"
            )

        self.device = resolve_torch_device(torch, device)
        self.model = Toto2Model.from_pretrained(str(path), local_files_only=True)
        self.model = self.model.to(self.device).eval()
        self.context_limit = int(context_limit)

    def predict(self, context_values, context_dates, horizon, freq):
        import torch

        values = one_dimensional(context_values)[-self.context_limit:]
        target = torch.as_tensor(
            values, dtype=torch.float32, device=self.device
        ).view(1, 1, -1)
        target_mask = torch.ones_like(target, dtype=torch.bool)
        series_ids = torch.zeros((1, 1), dtype=torch.long, device=self.device)

        with torch.inference_mode():
            quantiles = self.model.forecast(
                {
                    "target": target,
                    "target_mask": target_mask,
                    "series_ids": series_ids,
                },
                horizon=int(horizon),
                decode_block_size=None,
                has_missing_values=False,
            )

        # Official output: [quantile, batch, variate, horizon], levels 0.1...0.9.
        median = quantiles[4, 0, 0, :]
        return median.detach().float().cpu().numpy()[:horizon]
