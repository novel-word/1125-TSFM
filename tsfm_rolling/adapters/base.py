from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def as_numpy(values) -> np.ndarray:
    """Convert NumPy or torch output to a CPU float32 NumPy array."""
    if hasattr(values, "detach"):
        values = values.detach().float().cpu().numpy()
    return np.asarray(values, dtype=np.float32)


def validate_quantile_levels(quantile_levels, supported_levels=None, model_name="model"):
    """Validate requested levels and optionally map them to native model levels."""
    levels = tuple(float(q) for q in quantile_levels)
    if not levels:
        raise ValueError("At least one quantile level is required")
    if any(not 0.0 < q < 1.0 for q in levels):
        raise ValueError(f"Quantile levels must be between 0 and 1, got {levels}")
    if any(right <= left for left, right in zip(levels, levels[1:])):
        raise ValueError(f"Quantile levels must be unique and increasing, got {levels}")
    if not any(abs(q - 0.5) < 1e-8 for q in levels):
        raise ValueError("Quantile levels must include 0.5 so y_pred remains available")

    if supported_levels is None:
        return levels, tuple(range(len(levels)))

    native = tuple(float(q) for q in supported_levels)
    indices = []
    for requested in levels:
        if not native:
            raise ValueError(f"{model_name} did not expose native quantile levels")
        index = min(range(len(native)), key=lambda i: abs(native[i] - requested))
        if abs(native[index] - requested) > 1e-6:
            raise ValueError(
                f"{model_name} does not natively support quantile {requested}; "
                f"available levels are {native}"
            )
        indices.append(index)
    return levels, tuple(indices)


def validate_quantile_matrix(values, horizon, quantile_levels, model_name):
    """Require a finite [horizon, number_of_quantiles] matrix."""
    arr = as_numpy(values)
    expected = (int(horizon), len(quantile_levels))
    if arr.shape != expected:
        raise RuntimeError(
            f"{model_name} returned quantile shape {arr.shape}; expected {expected}"
        )
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{model_name} returned NaN/Inf quantile predictions")
    return arr


def quantile_column_name(level):
    """Return stable CSV names such as q10, q50 and q90."""
    return f"q{int(round(float(level) * 100)):02d}"


class ForecastAdapter(ABC):
    """Model adapter with a common probabilistic forecast interface."""

    @abstractmethod
    def predict_quantiles(
        self, context_values, context_dates, horizon, freq,
        quantile_levels=DEFAULT_QUANTILE_LEVELS,
    ):
        """Return quantile forecasts with shape [horizon, number_of_quantiles]."""
        raise NotImplementedError

    def predict_with_quantiles(
        self, context_values, context_dates, horizon, freq,
        quantile_levels=DEFAULT_QUANTILE_LEVELS,
    ):
        levels, _ = validate_quantile_levels(
            quantile_levels, model_name=type(self).__name__
        )
        quantiles = validate_quantile_matrix(
            self.predict_quantiles(
                context_values=context_values,
                context_dates=context_dates,
                horizon=horizon,
                freq=freq,
                quantile_levels=levels,
            ),
            horizon,
            levels,
            type(self).__name__,
        )
        median_index = min(range(len(levels)), key=lambda i: abs(levels[i] - 0.5))
        return quantiles[:, median_index], quantiles

    def predict(self, context_values, context_dates, horizon, freq):
        """Backward-compatible point forecast (the median unless overridden)."""
        point, _ = self.predict_with_quantiles(
            context_values=context_values,
            context_dates=context_dates,
            horizon=horizon,
            freq=freq,
            quantile_levels=DEFAULT_QUANTILE_LEVELS,
        )
        return point


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
