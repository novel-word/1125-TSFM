import numpy as np

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
        if self.context_limit <= 0:
            raise ValueError(
                f"context_limit must be positive, got {self.context_limit}"
            )

        # Toto2Model exposes the patch size through its loaded config.
        self.patch_size = int(self.model.config.patch_size)
        if self.patch_size <= 0:
            raise ValueError(
                f"Invalid TOTO2 patch_size: {self.patch_size}"
            )

    def predict(self, context_values, context_dates, horizon, freq):
        import torch

        values = np.asarray(
            one_dimensional(context_values), dtype=np.float32
        )[-self.context_limit:]

        if values.size == 0:
            raise ValueError(
                "TOTO2 received an empty context window. "
                "Check context_limit and the rolling-forecast input."
            )

        # Mark non-finite observations as missing so the model does not
        # treat NaN/Inf values as real measurements.
        observed_mask = np.isfinite(values)
        if not observed_mask.any():
            raise ValueError(
                "TOTO2 context contains no finite observations."
            )
        values = np.where(observed_mask, values, 0.0)

        # TOTO2's patch-based scaler requires context length to be a
        # multiple of patch_size. Left-pad so the latest real observation
        # remains at the end of a complete patch.
        pad_len = (-values.size) % self.patch_size
        if pad_len:
            values = np.pad(
                values,
                (pad_len, 0),
                mode="constant",
                constant_values=0.0,
            )
            observed_mask = np.pad(
                observed_mask,
                (pad_len, 0),
                mode="constant",
                constant_values=False,
            )

        target = torch.as_tensor(
            values, dtype=torch.float32, device=self.device
        ).view(1, 1, -1)
        target_mask = torch.as_tensor(
            observed_mask, dtype=torch.bool, device=self.device
        ).view(1, 1, -1)
        series_ids = torch.zeros(
            (1, 1), dtype=torch.long, device=self.device
        )

        # TOTO2 requires has_missing_values=True whenever target_mask
        # contains False entries (including the left-padding positions).
        has_missing_values = not bool(observed_mask.all())

        with torch.inference_mode():
            quantiles = self.model.forecast(
                {
                    "target": target,
                    "target_mask": target_mask,
                    "series_ids": series_ids,
                },
                horizon=int(horizon),
                decode_block_size=None,
                has_missing_values=has_missing_values,
            )

        # Official output: [quantile, batch, variate, horizon],
        # quantile levels 0.1 ... 0.9. Index 4 is the median (0.5).
        median = quantiles[4, 0, 0, :]
        return median.detach().float().cpu().numpy()[:horizon]
