import math

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

    def _forecast(self, context_values, horizon):
        values = one_dimensional(context_values)[-self.context_limit:]
        point_forecast, quantile_forecast = self.model.forecast(
            horizon=int(horizon),
            inputs=[values],
        )
        # TimesFM 2.5 returns [batch, horizon, 10]:
        # channel 0 is the mean, channels 1..9 are q10..q90.
        native = as_numpy(quantile_forecast)[0]
        if native.ndim != 2 or native.shape[1] < 10:
            raise RuntimeError(
                f"Unexpected TimesFM-2.5 quantile output shape: {native.shape}"
            )
        point = as_numpy(point_forecast)[0].reshape(-1)[:horizon]
        return point, native[:, 1:10]

    def predict_quantiles(
        self, context_values, context_dates, horizon, freq,
        quantile_levels=DEFAULT_QUANTILE_LEVELS,
    ):
        levels, _ = validate_quantile_levels(
            quantile_levels,
            supported_levels=DEFAULT_QUANTILE_LEVELS,
            model_name="TimesFM-2.5",
        )
        _, quantiles = self._forecast(context_values, horizon)
        return validate_quantile_matrix(quantiles, horizon, levels, "TimesFM-2.5")

    def predict_with_quantiles(
        self, context_values, context_dates, horizon, freq,
        quantile_levels=DEFAULT_QUANTILE_LEVELS,
    ):
        levels, _ = validate_quantile_levels(
            quantile_levels,
            supported_levels=DEFAULT_QUANTILE_LEVELS,
            model_name="TimesFM-2.5",
        )
        point, quantiles = self._forecast(context_values, horizon)
        quantiles = validate_quantile_matrix(quantiles, horizon, levels, "TimesFM-2.5")
        if point.size != int(horizon) or not np.isfinite(point).all():
            raise RuntimeError("TimesFM-2.5 returned an invalid point forecast")
        return point, quantiles
