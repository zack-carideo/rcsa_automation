import pandas as pd
import numpy as np
import re
from typing import Optional


def profile_df(
    df: pd.DataFrame,
    sample_top_n: int = 3,
    outlier_z_thresh: float = 3.0,
    display: bool = True,
) -> pd.DataFrame:
    """
    Generate a comprehensive column-level profile of a DataFrame.

    Parameters
    ----------
    df              : Input DataFrame.
    sample_top_n    : Number of top values to show in `top_values`.
    outlier_z_thresh: Z-score threshold for outlier flagging (default 3σ).
    display         : If True, render a styled HTML table (Jupyter-friendly).

    Returns
    -------
    pd.DataFrame — one row per column, all profile metrics as columns.
    """
    n_rows = len(df)
    records = []

    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        n_missing = s.isna().sum()
        n_unique  = s.nunique(dropna=True)

        # ── Mode ──────────────────────────────────────────────────────────
        mode_result = s.mode(dropna=True)
        mode_val    = mode_result.iloc[0] if len(mode_result) else pd.NA
        pct_mode    = round(100 * (s == mode_val).sum() / n_rows, 2) if pd.notna(mode_val) else pd.NA

        # ── Top N values ──────────────────────────────────────────────────
        top_counts  = s.value_counts(dropna=True).head(sample_top_n)
        top_values  = ", ".join(f"{v}({c})" for v, c in top_counts.items())

        # ── Inferred semantic type ─────────────────────────────────────────
        inferred = _infer_type(s)

        row = {
            "column"      : col,
            "dtype"       : str(s.dtype),
            "inferred_type": inferred,
            "n_rows"      : n_rows,
            "n_missing"   : n_missing,
            "pct_missing" : round(100 * n_missing / n_rows, 2),
            "n_unique"    : n_unique,
            "pct_unique"  : round(100 * n_unique / n_rows, 2),
            "mode"        : mode_val,
            "pct_mode"    : pct_mode,
            "top_values"  : top_values,
        }

        # ── Numeric extras ────────────────────────────────────────────────
        if pd.api.types.is_numeric_dtype(s) and inferred != "boolean":
            vals = non_null.astype(float)
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr    = q3 - q1

            row.update({
                "mean"         : round(vals.mean(), 4),
                "std"          : round(vals.std(), 4),
                "min"          : vals.min(),
                "p25"          : q1,
                "p50"          : vals.median(),
                "p75"          : q3,
                "max"          : vals.max(),
                "skewness"     : round(vals.skew(), 4),
                "kurtosis"     : round(vals.kurtosis(), 4),
                "pct_zero"     : round(100 * (vals == 0).sum() / n_rows, 2),
                "pct_negative" : round(100 * (vals < 0).sum() / n_rows, 2),
                "pct_positive" : round(100 * (vals > 0).sum() / n_rows, 2),
                "n_outlier_iqr": int(((vals < (q1 - 1.5 * iqr)) | (vals > (q3 + 1.5 * iqr))).sum()),
                "n_outlier_zscore": int(
                    (np.abs((vals - vals.mean()) / (vals.std() + 1e-12)) > outlier_z_thresh).sum()
                ),
            })

        # ── String extras ─────────────────────────────────────────────────
        elif inferred in ("string", "categorical"):
            str_vals = non_null.astype(str)
            row.update({
                "avg_str_len"  : round(str_vals.str.len().mean(), 2),
                "min_str_len"  : str_vals.str.len().min(),
                "max_str_len"  : str_vals.str.len().max(),
                "pct_has_digit": round(100 * str_vals.str.contains(r"\d").sum() / max(len(str_vals), 1), 2),
                "pct_has_space": round(100 * str_vals.str.contains(r"\s").sum() / max(len(str_vals), 1), 2),
                "has_email_pattern": bool(str_vals.str.contains(r"[^@\s]+@[^@\s]+\.[^@\s]+").any()),
                "has_url_pattern"  : bool(str_vals.str.contains(r"https?://").any()),
            })

        # ── Datetime extras ───────────────────────────────────────────────
        elif inferred == "datetime":
            dt_vals = pd.to_datetime(non_null, errors="coerce").dropna()
            if len(dt_vals):
                row.update({
                    "dt_min"       : dt_vals.min(),
                    "dt_max"       : dt_vals.max(),
                    "dt_range_days": (dt_vals.max() - dt_vals.min()).days,
                })

        records.append(row)

    profile = pd.DataFrame(records).set_index("column")

    if display:
        try:
            from IPython.display import display as ipy_display
            ipy_display(_style_profile(profile))
        except ImportError:
            pass  # non-Jupyter environment

    return profile


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infer_type(s: pd.Series) -> str:
    """Infer semantic type beyond pandas dtype."""
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        non_null = s.dropna()
        unique_vals = set(non_null.unique())
        if unique_vals <= {0, 1} or unique_vals <= {0.0, 1.0}:
            return "boolean"
        return "numeric"
    # Object columns — try datetime parse, then classify by cardinality
    non_null = s.dropna().astype(str)
    try:
        parsed = pd.to_datetime(non_null, infer_datetime_format=True, errors="coerce")
        if parsed.notna().mean() > 0.85:
            return "datetime"
    except Exception:
        pass
    # Email / ID heuristics
    if non_null.str.contains(r"[^@\s]+@[^@\s]+\.[^@\s]+").mean() > 0.5:
        return "email"
    n_unique_ratio = s.nunique(dropna=True) / max(len(s.dropna()), 1)
    if n_unique_ratio > 0.95:
        return "identifier"
    if n_unique_ratio < 0.05 or s.nunique(dropna=True) <= 20:
        return "categorical"
    return "string"


def _style_profile(profile: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Apply conditional formatting for quick visual scanning."""

    def highlight_missing(val):
        if pd.isna(val):
            return ""
        if val > 20:
            return "background-color: #f28b82; color: #4a1010"
        if val > 5:
            return "background-color: #fdd663; color: #3a2a00"
        return "background-color: #81c995; color: #0a3a1a"

    def highlight_skew(val):
        if pd.isna(val):
            return ""
        if abs(val) > 2:
            return "background-color: #f28b82; color: #4a1010"
        if abs(val) > 1:
            return "background-color: #fdd663; color: #3a2a00"
        return ""

    styler = profile.style.format(
        {
            "pct_missing" : "{:.2f}%",
            "pct_unique"  : "{:.2f}%",
            "pct_mode"    : "{:.2f}%",
            "mean"        : "{:.4f}",
            "std"         : "{:.4f}",
            "skewness"    : "{:.4f}",
            "kurtosis"    : "{:.4f}",
        },
        na_rep="—",
    )

    if "pct_missing" in profile.columns:
        styler = styler.applymap(highlight_missing, subset=["pct_missing"])
    if "skewness" in profile.columns:
        styler = styler.applymap(highlight_skew, subset=["skewness"])

    styler = styler.set_table_styles([
        {"selector": "th", "props": [("font-size", "11px"), ("text-align", "left"), ("white-space", "nowrap")]},
        {"selector": "td", "props": [("font-size", "11px"), ("white-space", "nowrap")]},
    ])

    return styler