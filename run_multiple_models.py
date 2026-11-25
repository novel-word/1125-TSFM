import argparse
import copy
import gc
import traceback
from pathlib import Path

import pandas as pd

from tsfm_rolling.adapters.base import (
    DEFAULT_QUANTILE_LEVELS,
    quantile_column_name,
    validate_quantile_levels,
)
from tsfm_rolling.config import load_config
from tsfm_rolling.data import load_train_test
from tsfm_rolling.metrics import calculate_metrics
from tsfm_rolling.registry import build_model
from tsfm_rolling.rolling import rolling_forecast
from tsfm_rolling.utils import save_json


def collect_model_configs(cfg):
    """Normalize cfg into a list of model configs; retain legacy single-model support."""
    if cfg.get("models"):
        base = dict(cfg.get("model") or {})
        base.pop("name", None)
        base.pop("checkpoint_path", None)
        result = []
        for item in cfg["models"]:
            if not item.get("enabled", True):
                continue
            result.append({**base, **item})
        return result
    return [cfg["model"]]


def run_one_model(cfg, history_df, test_df):
    name = cfg["model"]["name"]
    print(f"\n===== Running: {name} =====")
    print(f"checkpoint: {cfg['model'].get('checkpoint_path')}")

    model = build_model(cfg)
    result_df = rolling_forecast(model, history_df, test_df, cfg)

    output_dir = Path(cfg["output"]["dir"]) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"{name}_predictions.csv"
    metric_path = output_dir / f"{name}_metrics.json"
    result_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    quantile_levels = cfg["forecast"].get("quantiles", DEFAULT_QUANTILE_LEVELS)
    quantile_columns = [quantile_column_name(q) for q in quantile_levels]
    metrics = calculate_metrics(
        result_df["y_true"].to_numpy(),
        result_df["y_pred"].to_numpy(),
        y_quantiles=result_df[quantile_columns].to_numpy(),
        quantile_levels=quantile_levels,
    )
    save_json(metrics, metric_path)
    print(f"Predictions: {pred_path}")
    print(f"Metrics: {metric_path}")
    print(metrics)

    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return {"name": name, "metrics": metrics, "pred_path": str(pred_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_multiple_models.yaml")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model names, e.g. timesfm_3_0,tirex")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output"]["dir"] = args.output_dir

    history_df, test_df, inferred_freq = load_train_test(cfg)
    if not cfg["data"].get("freq"):
        cfg["data"]["freq"] = inferred_freq

    quantile_levels, _ = validate_quantile_levels(
        cfg["forecast"].get("quantiles", DEFAULT_QUANTILE_LEVELS),
        model_name="config",
    )
    print(f"History rows: {len(history_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Frequency: {cfg['data']['freq']}")
    print(f"Rolling horizon H: {cfg['forecast']['horizon']}")
    print(f"Quantile levels: {quantile_levels}")

    model_configs = collect_model_configs(cfg)
    if args.models:
        wanted = {s.strip() for s in args.models.split(",") if s.strip()}
        model_configs = [m for m in model_configs if m["name"] in wanted]
    print(f"Models to run: {[m['name'] for m in model_configs]}")

    summary, failed = [], []
    for model_cfg in model_configs:
        run_cfg = copy.deepcopy(cfg)
        run_cfg["model"] = model_cfg
        try:
            summary.append(run_one_model(run_cfg, history_df.copy(), test_df.copy()))
        except Exception as exc:
            traceback.print_exc()
            failed.append({"name": model_cfg["name"], "error": str(exc)})

    output_root = Path(cfg["output"]["dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    summary_csv_path = output_root / "summary.csv"
    save_json({"results": summary, "failed": failed}, summary_path)

    summary_rows = [
        {"name": item["name"], "status": "success", **item["metrics"],
         "pred_path": item["pred_path"]}
        for item in summary
    ]
    summary_rows.extend(
        {"name": item["name"], "status": "failed", "error": item["error"]}
        for item in failed
    )
    pd.DataFrame(summary_rows).to_csv(
        summary_csv_path, index=False, encoding="utf-8-sig"
    )
    print(f"\nSummary saved: {summary_path}")
    print(f"Summary CSV saved: {summary_csv_path}")
    if failed:
        print("Failed:", [item["name"] for item in failed])


import time

if __name__ == "__main__":
    start = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start
    minutes, seconds = divmod(elapsed, 60)
    print(f"耗时: {int(minutes)} 分 {seconds:.2f} 秒")
