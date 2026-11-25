from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class ForecastAdapter(ABC):
    """One model adapter. All adapters return a 1-D point forecast of length H."""

    @abstractmethod
    def predict(
        self,
        context_values: np.ndarray,
        context_dates: pd.Series,
        horizon: int,
        freq: str | None,
    ) -> np.ndarray:
        raise NotImplementedError


def require_path(path: str, description: str = "checkpoint") -> Path:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"{description} path does not exist: {p}")
    return p


def one_dimensional(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("The model context is empty")
    if not np.isfinite(arr).all():
        raise ValueError(
            "The current project requires a finite target context. "
            "Add a shared missing-value preprocessing policy if your data has gaps."
        )
    return arr


def resolve_torch_device(torch, device_name: str):
    """Resolve a requested device and fail clearly if CUDA was requested but unavailable."""
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device '{device_name}' was requested, but CUDA is unavailable. "
            "Install a CUDA-enabled PyTorch build or set model.device: 'cpu'."
        )
    return torch.device(device_name)
