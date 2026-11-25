import json
import os
import sys

import numpy as np

from .base import (
    DEFAULT_QUANTILE_LEVELS,
    ForecastAdapter,
    one_dimensional,
    require_path,
    validate_quantile_levels,
    validate_quantile_matrix,
)


class HART36MAdapter(ForecastAdapter):
    """HART official Keras/RWKV-Time implementation with quantile output."""

    def __init__(
        self,
        checkpoint_path: str,
        repo_path: str,
        kernel_type: str = "cuda",
        context_limit: int = 8192,
    ):
        os.environ["KERAS_BACKEND"] = "torch"
        os.environ["KERNEL_TYPE"] = kernel_type
        os.environ["KERNEL_BACKEND"] = "torch"

        repo = require_path(repo_path, "HART source repository")
        model_dir = require_path(checkpoint_path, "HART checkpoint directory")
        config_path = model_dir / "config.json"
        weights_path = model_dir / "model.weights.h5"

        if not (repo / "model" / "RWKV_Time").is_dir():
            raise FileNotFoundError(
                f"Invalid HART repository: expected {repo / 'model' / 'RWKV_Time'}"
            )
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                "HART checkpoint directory must contain config.json "
                "and model.weights.h5"
            )

        sys.path.insert(0, str(repo))
        import keras
        import torch
        from model.RWKV_Time.rwkv_time_backbone import RWKVTimeBackbone
        from model.RWKV_Time.rwkv_time_causal_lm_preprocessor import (
            RWKV_time_CausalLMPreprocessor,
        )

        with config_path.open("r", encoding="utf-8") as f:
            model_config = json.load(f)

        self.keras = keras
        self.torch = torch
        self.patch_size = int(model_config["patch_size"])
        self.n_quantiles = int(model_config["n_quantiles"])
        native_levels = model_config.get("quantile_levels")
        if native_levels is None:
            native_levels = model_config.get("quantiles")
        if native_levels is None:
            if self.n_quantiles != 9:
                raise ValueError(
                    "HART checkpoint does not declare quantile_levels and "
                    f"n_quantiles={self.n_quantiles}; cannot safely infer quantile labels"
                )
            native_levels = DEFAULT_QUANTILE_LEVELS
        self.native_quantile_levels = tuple(float(q) for q in native_levels)
        if len(self.native_quantile_levels) != self.n_quantiles:
            raise ValueError(
                "HART config quantile level count does not match n_quantiles: "
                f"{len(self.native_quantile_levels)} != {self.n_quantiles}"
            )
        self.context_limit = int(context_limit)

        self.backbone = RWKVTimeBackbone(**model_config)
        self.backbone.compile()
        self.backbone.load_weights(str(weights_path), True)
        self.preprocessor = RWKV_time_CausalLMPreprocessor(
            patch_size=self.patch_size,
            predict_token_num=1,
            sequence_length=2,
        )

    def predict_quantiles(
        self, context_values, context_dates, horizon, freq,
        quantile_levels=DEFAULT_QUANTILE_LEVELS,
    ):
        values = one_dimensional(context_values)[-self.context_limit:]
        levels, indices = validate_quantile_levels(
            quantile_levels,
            supported_levels=self.native_quantile_levels,
            model_name="HART",
        )
        num_tokens = (int(horizon) + self.patch_size - 1) // self.patch_size

        self.preprocessor.predict_token_num = num_tokens
        self.preprocessor.sequence_length = len(values) // self.patch_size + 1
        model_inputs, mean, std = self.preprocessor.generate_preprocess(
            [values],
            sequence_length=num_tokens,
        )

        with self.torch.inference_mode():
            model_outputs = self.backbone.predict(
                model_inputs,
                batch_size=1,
                verbose=0,
            )

        pred = np.transpose(model_outputs[:, -num_tokens:], [0, 1, 3, 2])
        pred = np.reshape(pred, [len(pred), -1, self.n_quantiles])
        pred = (pred * std + mean)[:, :horizon, :]
        quantiles = pred[0][:, list(indices)]
        return validate_quantile_matrix(quantiles, horizon, levels, "HART")
