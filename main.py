from typing import Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd

import config
import data

app = FastAPI(title="Flight Delay Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    # Also accept any vercel.app URL automatically, regardless of the
    # FLIGHT_ALLOWED_ORIGINS env var. This means the frontend just works
    # once deployed to Vercel, no exact string to type/match/redeploy.
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)


def _parse_list(raw: Optional[str]):
    if not raw:
        return None
    return [v for v in raw.split(",") if v]


@app.get("/api/health")
def health():
    try:
        df = data.load_predictions()
        return {"status": "ok", "rows": int(len(df)), "data_path": config.DATA_PATH}
    except FileNotFoundError:
        return {"status": "error", "detail": f"Data file not found at {config.DATA_PATH}"}


@app.get("/api/filter-options")
def filter_options():
    df = data.load_predictions()
    carriers = sorted(df["op_unique_carrier"].dropna().unique().tolist())
    months = sorted(df["month"].dropna().unique().tolist())
    days = sorted(df["day_of_week"].dropna().unique().tolist())
    return {
        "carriers": [{"code": c, "name": config.CARRIER_NAMES.get(c, c)} for c in carriers],
        "months": [{"value": m, "name": config.MONTH_NAMES.get(m, m)} for m in months],
        "days": [{"value": d, "name": config.DAY_NAMES.get(d, d)} for d in days],
    }


@app.get("/api/key-insights")
def key_insights(
    carriers: Optional[str] = Query(None),
    months: Optional[str] = Query(None),
    days: Optional[str] = Query(None),
):
    df = data.load_predictions()
    metrics_df = data.load_metrics()

    carrier_list = _parse_list(carriers)
    month_list = [int(m) for m in _parse_list(months)] if _parse_list(months) else None
    day_list = [int(d) for d in _parse_list(days)] if _parse_list(days) else None

    filtered = data.apply_filters(df, carrier_list, month_list, day_list)
    n = len(filtered)
    if n == 0:
        return {"empty": True}

    actual_avg = float(filtered["actual_arr_delay"].mean())
    predicted_avg = float(filtered["predicted_arr_delay"].mean())
    model_bias = predicted_avg - actual_avg
    typical_error = float(filtered["residual"].abs().mean())
    delayed_share = float((filtered["actual_arr_delay"] >= config.DELAYED_THRESHOLD_MIN).mean() * 100)

    best_row = metrics_df.loc[metrics_df["MAE"].idxmin()]

    # Cause breakdown, for the donut chart
    present_causes = [c for c in config.CAUSE_COLS if c in filtered.columns]
    cause_avg = filtered[present_causes].mean().rename(config.CAUSE_LABELS).sort_values(ascending=False)
    top_cause = cause_avg.index[0] if len(cause_avg) else None
    top_cause_val = float(cause_avg.iloc[0]) if len(cause_avg) else None
    top_cause_share = float(cause_avg.iloc[0] / cause_avg.sum() * 100) if len(cause_avg) and cause_avg.sum() > 0 else None

    # Worst departure hour
    hourly = filtered.groupby("dep_hour")["actual_arr_delay"].mean()
    peak_hour = int(hourly.idxmax()) if len(hourly) else None
    peak_hour_val = float(hourly.max()) if len(hourly) else None

    # Worst carrier (with a minimum sample size, same as Streamlit)
    carrier_stats = filtered.groupby("op_unique_carrier")["actual_arr_delay"].agg(["mean", "count"])
    carrier_stats_elig = carrier_stats[carrier_stats["count"] >= 50]
    worst_carrier = None
    worst_carrier_val = None
    if not carrier_stats_elig.empty:
        worst_row = carrier_stats_elig["mean"].idxmax()
        worst_carrier = worst_row
        worst_carrier_val = float(carrier_stats_elig.loc[worst_row, "mean"])

    # Naive-guess comparison, for the trust section
    naive_pred = float(filtered["actual_arr_delay"].mean())
    naive_mae = float((filtered["actual_arr_delay"] - naive_pred).abs().mean())
    improvement_pct = float((naive_mae - typical_error) / naive_mae * 100) if naive_mae > 0 else 0.0
    acc_15 = float((filtered["residual"].abs() <= 15).mean() * 100)
    acc_30 = float((filtered["residual"].abs() <= 30).mean() * 100)

    return {
        "empty": False,
        "flights_in_view": int(n),
        "actual_delay_avg": actual_avg,
        "predicted_delay_avg": predicted_avg,
        "model_bias": model_bias,
        "typical_error": typical_error,
        "deployed_model": best_row["model"],
        "deployed_model_r2": float(best_row["R2"]) * 100,
        "delayed_share": delayed_share,
        "delayed_status": data.severity_tier(delayed_share if delayed_share < 20 else 20),
        "top_cause": top_cause,
        "top_cause_val": top_cause_val,
        "top_cause_share": top_cause_share,
        "peak_hour": peak_hour,
        "peak_hour_val": peak_hour_val,
        "worst_carrier": worst_carrier,
        "worst_carrier_val": worst_carrier_val,
        "cause_breakdown": [
            {"name": name, "value": float(val)} for name, val in cause_avg.items()
        ],
        "trust": {
            "improvement_pct": improvement_pct,
            "naive_mae": naive_mae,
            "model_mae": typical_error,
            "acc_15": acc_15,
            "acc_30": acc_30,
        },
    }
