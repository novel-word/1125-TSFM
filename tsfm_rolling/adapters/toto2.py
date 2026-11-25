import numpy as np

from .base import (
    DEFAULT_QUANTILE_LEVELS,
    ForecastAdapter,
    as_numpy,
    one_dimensional,
    require_path,
    resolve_torch_device,
    validate_quantile_levels,
    validate_quantile_matrix,
)


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
            raise ValueError(f"context_limit must be positive, got {self.context_limit}")

        self.patch_size = int(self.model.config.patch_size)
        if self.patch_size <= 0:
            raise ValueError(f"Invalid TOTO2 patch_size: {self.patch_size}")

    def predict_quantiles(
        self, context_values, context_dates, horizon, freq,
        quantile_levels=DEFAULT_QUANTILE_LEVELS,
    ):
        levels, _ = validate_quantile_levels(
            quantile_levels,
            supported_levels=DEFAULT_QUANTILE_LEVELS,
            model_name="Toto-2",
        )
        values = one_dimensional(context_values)[-self.context_limit:]
        observed_mask = np.isfinite(values)
        if not observed_mask.any():
            raise ValueError("Toto-2 context contains no finite observations.")
        values = np.where(observed_mask, values, 0.0)

        # Toto-2's patch-based scaler requires context length to be a multiple
        # of patch_size. Left-pad while keeping the latest observation last.
        pad_len = (-values.size) % self.patch_size
        if pad_len:
            values = np.pad(values, (pad_len, 0), mode="constant", constant_values=0.0)
            observed_mask = np.pad(
                observed_mask, (pad_len, 0), mode="constant", constant_values=False
            )

        import torch

        target = torch.as_tensor(values, dtype=torch.float32, device=self.device).view(1, 1, -1)
        target_mask = torch.as_tensor(
            observed_mask, dtype=torch.bool, device=self.device
        ).view(1, 1, -1)
        series_ids = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        has_missing_values = not bool(observed_mask.all())

        with torch.inference_mode():
            output = self.model.forecast(
                {
                    "target": target,
                    "target_mask": target_mask,
                    "series_ids": series_ids,
                },
                horizon=int(horizon),
                decode_block_size=None,
                has_missing_values=has_missing_values,
            )

        # Official shape: [quantile, batch, variate, horizon].
        native = as_numpy(output)
        if native.ndim != 4 or native.shape[0] != 9:
            raise RuntimeError(f"Unexpected Toto-2 output shape: {native.shape}")
        quantiles = native[:, 0, 0, :].T
        return validate_quantile_matrix(quantiles, horizon, levels, "Toto-2")
