import argparse
import copy
import gc
import traceback
from pathlib import Path

from tsfm_rolling.config import load_config
from tsfm_rolling.data import load_train_test
from tsfm_rolling.metrics import calculate_metrics
from tsfm_rolling.registry import build_model
from tsfm_rolling.rolling import rolling_forecast
from tsfm_rolling.utils import save_json


def collect_model_configs(cfg):
    """把 cfg 规范成 [(model_cfg), ...]，兼容旧的单个 model 配置。"""
    if cfg.get("models"):
        base = dict(cfg.get("model") or {})
        base.pop("name", None)
        base.pop("checkpoint_path", None)

        result = []
        for item in cfg["models"]:
            if not item.get("enabled", True):
                continue
            merged = {**base, **item}
            result.append(merged)
        return result

    return [cfg["model"]]


def run_one_model(cfg, history_df, test_df):
    name = cfg["model"]["name"]
    print(f"\n===== Running: {name} =====")
    print(f"checkpoint: {cfg['model'].get('checkpoint_path')}")

    model = build_model(cfg)
    result_df = rolling_forecast(
        model=model,
        history_df=history_df,
        test_df=test_df,
        cfg=cfg,
    )

    output_dir = Path(cfg["output"]["dir"]) / name
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_path = output_dir / f"{name}_predictions.csv"
    metric_path = output_dir / f"{name}_metrics.json"

    result_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    metrics = calculate_metrics(
        result_df["y_true"].to_numpy(),
        result_df["y_pred"].to_numpy(),
    )
    save_json(metrics, metric_path)

    print(f"Predictions: {pred_path}")
    print(f"Metrics: {metric_path}")
    print(metrics)

    # 释放显存，避免多模型连续跑 OOM
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
                        help="逗号分隔，只跑这些模型，例如 timesfm_3_0,tirex")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output"]["dir"] = args.output_dir

    history_df, test_df, inferred_freq = load_train_test(cfg)
    if not cfg["data"].get("freq"):
        cfg["data"]["freq"] = inferred_freq

    print(f"History rows: {len(history_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Frequency: {cfg['data']['freq']}")
    print(f"Rolling horizon H: {cfg['forecast']['horizon']}")

    model_configs = collect_model_configs(cfg)
    if args.models:
        wanted = {s.strip() for s in args.models.split(",") if s.strip()}
        model_configs = [m for m in model_configs if m["name"] in wanted]

    print(f"Models to run: {[m['name'] for m in model_configs]}")

    summary, failed = [], []
    for model_cfg in model_configs:
        run_cfg = copy.deepcopy(cfg)
        run_cfg["model"] = model_cfg

        # 如果 rolling_forecast 会就地修改 history_df/test_df，这里要复制
        h = history_df.copy()
        t = test_df.copy()

        try:
            summary.append(run_one_model(run_cfg, h, t))
        except Exception as e:
            traceback.print_exc()
            failed.append({"name": model_cfg["name"], "error": str(e)})

    summary_path = Path(cfg["output"]["dir"]) / "summary.json"
    save_json({"results": summary, "failed": failed}, summary_path)
    print(f"\nSummary saved: {summary_path}")
    if failed:
        print("Failed:", [f["name"] for f in failed])

import time
if __name__ == "__main__":
    start = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start
    minutes, seconds = divmod(elapsed, 60)
    print(f"耗时: {int(minutes)} 分 {seconds:.2f} 秒")