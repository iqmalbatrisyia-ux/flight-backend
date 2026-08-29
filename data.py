import gc
import pandas as pd
from functools import lru_cache
import config

# Only the columns this API actually uses. The original file has ~20
# columns (including origin/dest/distance/model_used and some *_encoded
# duplicates not needed here), and loading all of them was pushing past
# Render's free-tier 512MB memory limit, causing the server to be killed
# and restarted in a loop (visible as repeated 502s). Cutting to just
# what's needed roughly halves the memory footprint on load.
NEEDED_COLUMNS = [
    "month", "day_of_week", "dep_hour", "op_unique_carrier",
    "actual_arr_delay", "predicted_arr_delay",
    "carrier_delay", "weather_delay", "nas_delay",
    "security_delay", "late_aircraft_delay",
]


@lru_cache(maxsize=1)
def load_predictions() -> pd.DataFrame:
    """Loads the real prediction data once and keeps it in memory. Reads
    Parquet (smaller, faster) rather than the original CSV, and only the
    columns actually needed, to fit comfortably in a free-tier host's
    memory limit."""
    if config.DATA_PATH.lower().endswith(".csv"):
        df = pd.read_csv(config.DATA_PATH, usecols=NEEDED_COLUMNS)
    else:
        df = pd.read_parquet(config.DATA_PATH, columns=NEEDED_COLUMNS)

    # Downcast to smaller number types. These delay columns don't need
    # 64-bit precision, and month/day/hour only ever hold small integers.
    float_cols = ["actual_arr_delay", "predicted_arr_delay", "carrier_delay",
                  "weather_delay", "nas_delay", "security_delay", "late_aircraft_delay"]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype("float32")
    for col in ["month", "day_of_week", "dep_hour"]:
        if col in df.columns:
            df[col] = df[col].astype("int16")
    df["op_unique_carrier"] = df["op_unique_carrier"].astype("category")

    df["residual"] = df["predicted_arr_delay"] - df["actual_arr_delay"]
    df["month_name"] = df["month"].map(config.MONTH_NAMES).astype("category")
    df["day_name"] = df["day_of_week"].map(config.DAY_NAMES).astype("category")
    gc.collect()
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
