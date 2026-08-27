import pandas as pd
from functools import lru_cache
import config


@lru_cache(maxsize=1)
def load_predictions() -> pd.DataFrame:
    """Loads the real prediction data once and keeps it in memory. Reads
    Parquet (smaller, faster) rather than the original CSV."""
    if config.DATA_PATH.lower().endswith(".csv"):
        df = pd.read_csv(config.DATA_PATH)
    else:
        df = pd.read_parquet(config.DATA_PATH)
    df["residual"] = df["predicted_arr_delay"] - df["actual_arr_delay"]
    df["month_name"] = df["month"].map(config.MONTH_NAMES)
    df["day_name"] = df["day_of_week"].map(config.DAY_NAMES)
    return df


@lru_cache(maxsize=1)
def load_metrics() -> pd.DataFrame:
    return pd.read_csv(config.METRICS_PATH)


def apply_filters(df: pd.DataFrame, carriers=None, months=None, days=None) -> pd.DataFrame:
    """Same filtering logic as the Streamlit sidebar: an empty list means
    'include everything' for that category, not 'exclude everything'."""
    out = df
    if carriers:
        out = out[out["op_unique_carrier"].isin(carriers)]
    if months:
        out = out[out["month"].isin(months)]
    if days:
        out = out[out["day_of_week"].isin(days)]
    return out


def severity_tier(value: float) -> str:
    if value < config.SEVERITY_LOW:
        return "good"
    if value < config.SEVERITY_HIGH:
        return "amber"
    return "bad"
