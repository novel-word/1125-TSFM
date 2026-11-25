from pathlib import Path

import pandas as pd


def load_train_test(cfg: dict):
    """Initial history is strictly before test_start; test interval is inclusive."""
    dcfg = cfg["data"]
    csv_path = Path(dcfg["csv_path"])
    time_col = dcfg["timestamp_col"]
    target_col = dcfg["target_col"]

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for col in (time_col, target_col):
        if col not in df.columns:
            raise KeyError(f"CSV does not contain column: {col}")

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    if df[time_col].isna().any():
        raise ValueError(f"Invalid timestamp exists in {time_col}")
    if df[target_col].isna().any():
        raise ValueError(
            f"Missing/non-numeric values exist in {target_col}. "
            "Please define one consistent missing-value policy before comparing models."
        )

    df = df.sort_values(time_col).reset_index(drop=True)
    if df[time_col].duplicated().any():
        raise ValueError(f"Duplicate timestamps exist in {time_col}")

    test_start = pd.Timestamp(dcfg["test_start"])
    test_end = pd.Timestamp(dcfg["test_end"])
    if test_end < test_start:
        raise ValueError("test_end must not be earlier than test_start")

    history_df = df.loc[df[time_col] < test_start].copy().reset_index(drop=True)
    test_df = df.loc[
        (df[time_col] >= test_start) & (df[time_col] <= test_end)
    ].copy().reset_index(drop=True)

    if history_df.empty:
        raise ValueError("No history exists before test_start")
    if test_df.empty:
        raise ValueError("No observations found in the requested test interval")

    freq = dcfg.get("freq")
    if not freq:
        freq = pd.infer_freq(df[time_col])
        if freq is None:
            raise ValueError(
                "Cannot infer a regular frequency. Set data.freq explicitly, "
                "for example '15min', 'h', 'D', 'W', or 'MS'."
            )

    return history_df, test_df, freq
