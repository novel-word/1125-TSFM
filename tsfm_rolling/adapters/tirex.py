import numpy as np
import pandas as pd

from .base import ForecastAdapter, one_dimensional, require_path


class TiRexAdapter(ForecastAdapter):
    """NX-AI original TiRex, loaded directly through its official model class."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str,
        backend: str = "torch",
        compile_model: bool = False,
        context_limit: int = 8192,
    ):
        from tirex.models.tirex import TiRexZero

        path = require_path(checkpoint_path)
        if path.is_dir():
            path = path / "model.ckpt"
        if not path.exists():
            raise FileNotFoundError(f"TiRex checkpoint file not found: {path}")

        self.model = TiRexZero.from_pretrained(
            str(path),
            device=device,
            backend=backend,
            compile=bool(compile_model),
        )
        self.model.eval()
        self.device = device
        self.context_limit = int(context_limit)

    def predict(self, context_values, context_dates, horizon, freq):
        import torch

        values = one_dimensional(context_values)[-self.context_limit:]
        context = torch.as_tensor(values, dtype=torch.float32).unsqueeze(0)

        with torch.inference_mode():
            _, mean = self.model.forecast(
                context=context,
                prediction_length=int(horizon),
                output_type="numpy",
            )
        return np.asarray(mean, dtype=np.float32).reshape(-1)[:horizon]
