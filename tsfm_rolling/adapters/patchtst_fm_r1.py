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

        self.model_context = int(self.model.config.context_length)
        self.context_limit = min(int(context_limit), self.model_context)

    def predict_quantiles(
        self, context_values, context_dates, horizon, freq,
        quantile_levels=DEFAULT_QUANTILE_LEVELS,
    ):
        import torch

        levels, _ = validate_quantile_levels(
            quantile_levels, model_name="PatchTST-FM-R1"
        )
        if horizon >= self.model_context:
            raise ValueError(
                f"PatchTST-FM-R1 requires context + horizon <= "
                f"{self.model_context}; got horizon={horizon}"
            )

        max_history = min(self.context_limit, self.model_context - int(horizon))
        values = one_dimensional(context_values)[-max_history:]
        past_values = torch.as_tensor(
            values, dtype=torch.float32, device=self.device
        ).view(1, -1, 1)

        with torch.inference_mode():
            output = self.model(
                past_values=past_values,
                prediction_length=int(horizon),
                quantile_levels=list(levels),
                return_dict=True,
            )

        # Output shape: [batch, quantile, horizon, variate].
        native = as_numpy(output.quantile_outputs)
        if native.ndim != 4 or native.shape[0] != 1 or native.shape[3] != 1:
            raise RuntimeError(
                f"Unexpected PatchTST-FM-R1 quantile output shape: {native.shape}"
            )
        quantiles = native[0, :, :horizon, 0].T
        return validate_quantile_matrix(quantiles, horizon, levels, "PatchTST-FM-R1")
