import math
import numpy as np
import pandas as pd

from .base import ForecastAdapter, one_dimensional, require_path, resolve_torch_device


class TimesFM25Adapter(ForecastAdapter):
    """Google official TimesFM 2.5 PyTorch implementation."""

    def __init__(
        self,
        checkpoint_path: str,
        context_limit: int,
        max_horizon: int,
        batch_size: int,
        device: str,
        normalize_inputs: bool = True,
        infer_is_positive: bool = False,
    ):
        import torch
        import timesfm
        from timesfm import TimesFM_2p5_200M_torch, ForecastConfig

        path = require_path(checkpoint_path)
        if not path.is_dir():
            raise ValueError(
                "TimesFM-2.5 checkpoint_path must be a local directory "
                "containing config.json and model.safetensors"
            )

        self.model = TimesFM_2p5_200M_torch.from_pretrained(
            str(path),
            local_files_only=True,
            force_download=False,
        )

        device_obj = resolve_torch_device(torch, device)
        self.model.model.to(device_obj)
        self.model.model.device = device_obj

        # TimesFM 2.5 rounds max_context to 32 and max_horizon to 128 internally.
        rounded_horizon = int(math.ceil(max_horizon / 128) * 128)
        safe_context = min(
            int(context_limit),
            int(self.model.model.config.context_limit) - rounded_horizon,
        )
        safe_context = max(32, (safe_context // 32) * 32)
        self.context_limit = safe_context

        self.model.compile(
            ForecastConfig(
                max_context=safe_context,
                max_horizon=rounded_horizon,
                normalize_inputs=normalize_inputs,
                per_core_batch_size=int(batch_size),
                use_continuous_quantile_head=False,
                force_flip_invariance=True,
                infer_is_positive=bool(infer_is_positive),
                fix_quantile_crossing=True,
            )
        )

    def predict(self, context_values, context_dates, horizon, freq):
        values = one_dimensional(context_values)[-self.context_limit:]
        point_forecast, _ = self.model.forecast(
            horizon=int(horizon),
            inputs=[values],
        )
        return np.asarray(point_forecast[0], dtype=np.float32)[:horizon]
