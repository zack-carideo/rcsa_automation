"""
eda_profiler.py
===============
Exhaustive EDA profiling for mixed-type DataFrames.
Generates a self-contained HTML report with embedded plots.

Usage
-----
    from eda_profiler import profile_df, generate_report

    # Quick stats table (Jupyter-friendly styled DataFrame)
    stats = profile_df(df)

    # Full HTML report
    generate_report(df, output_path="eda_report.html", title="Loan Portfolio EDA")
"""

from __future__ import annotations

import base64
import io
import re
import textwrap
import warnings
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Plotting defaults ──────────────────────────────────────────────────────────
PALETTE_NUMERIC   = "#4C72B0"
PALETTE_CAT       = sns.color_palette("muted", 20)
PALETTE_HEAT      = "coolwarm"
FIG_DPI           = 120
sns.set_theme(style="whitegrid", palette="muted", font_scale=0.95)
plt.rcParams.update({
    "figure.dpi"       : FIG_DPI,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.titlesize"   : 10,
    "axes.labelsize"   : 9,
    "xtick.labelsize"  : 8,
    "ytick.labelsize"  : 8,
})

MAX_CAT_BARS      = 20   # max categories shown in bar charts
MAX_PAIR_COLS     = 8    # max numeric cols in pairplot
OUTLIER_Z_THRESH  = 3.0


# ══════════════════════════════════════════════════════════════════════════════
# 1.  TYPE INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def _infer_type(s: pd.Series) -> str:
    """Semantic type beyond pandas dtype."""
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        non_null  = s.dropna()
        u         = set(non_null.unique())
        if u <= {0, 1} or u <= {0.0, 1.0}:
            return "boolean"
        if s.nunique(dropna=True) <= 2:
            return "binary"
        return "numeric"

    non_null = s.dropna().astype(str)
    if len(non_null) == 0:
        return "empty"

    # Try parsing as datetime
    try:
        parsed = pd.to_datetime(non_null, infer_datetime_format=True, errors="coerce")
        if parsed.notna().mean() > 0.85:
            return "datetime"
    except Exception:
        pass

    # Email / URL / ID heuristics
    if non_null.str.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$").mean() > 0.5:
        return "email"
    if non_null.str.match(r"^https?://").mean() > 0.3:
        return "url"

    n_unique_ratio = s.nunique(dropna=True) / max(len(s.dropna()), 1)
    if n_unique_ratio > 0.95:
        return "identifier"
    if n_unique_ratio < 0.05 or s.nunique(dropna=True) <= 20:
        return "categorical"
    return "string"


# ══════════════════════════════════════════════════════════════════════════════
# 2.  COLUMN-LEVEL STATS TABLE
# ══════════════════════════════════════════════════════════════════════════════

def profile_df(
    df: pd.DataFrame,
    sample_top_n: int = 5,
    outlier_z_thresh: float = OUTLIER_Z_THRESH,
    display_styled: bool = True,
) -> pd.DataFrame:
    """
    Returns a column-level profile DataFrame.

    Columns
    -------
    Universal   : dtype, inferred_type, n_rows, n_missing, pct_missing,
                  n_unique, pct_unique, mode, pct_mode, top_values
    Numeric     : mean, std, min, p01, p05, p25, p50, p75, p95, p99, max,
                  range, iqr, cv (coeff. of variation), skewness, excess_kurtosis,
                  pct_zero, pct_negative, pct_positive,
                  n_outlier_iqr, n_outlier_zscore, normality_p (Shapiro/KS)
    Categorical : avg_str_len, min_str_len, max_str_len,
                  pct_has_digit, pct_has_space, pct_upper, pct_lower,
                  has_email_pattern, has_url_pattern
    Datetime    : dt_min, dt_max, dt_range_days, dt_most_common_year,
                  dt_most_common_dow (0=Mon), dt_weekend_pct
    """
    n_rows  = len(df)
    records = []

    for col in df.columns:
        s        = df[col]
        non_null = s.dropna()
        n_miss   = int(s.isna().sum())
        n_uniq   = int(s.nunique(dropna=True))

        mode_result = s.mode(dropna=True)
        mode_val    = mode_result.iloc[0] if len(mode_result) else pd.NA
        pct_mode    = round(100 * (s == mode_val).sum() / n_rows, 2) if pd.notna(mode_val) else pd.NA

        top_counts = s.value_counts(dropna=True).head(sample_top_n)
        top_values = " | ".join(f"{v}({c})" for v, c in top_counts.items())

        inferred = _infer_type(s)

        row: dict = {
            "column"        : col,
            "dtype"         : str(s.dtype),
            "inferred_type" : inferred,
            "n_rows"        : n_rows,
            "n_missing"     : n_miss,
            "pct_missing"   : round(100 * n_miss / n_rows, 2),
            "n_unique"      : n_uniq,
            "pct_unique"    : round(100 * n_uniq / n_rows, 2),
            "mode"          : mode_val,
            "pct_mode"      : pct_mode,
            "top_values"    : top_values,
        }

        # ── Numeric ───────────────────────────────────────────────────────
        if pd.api.types.is_numeric_dtype(s) and inferred not in ("boolean", "binary"):
            v  = non_null.astype(float)
            q1, q3 = v.quantile(0.25), v.quantile(0.75)
            iqr_val = q3 - q1
            rng     = v.max() - v.min()
            cv      = (v.std() / v.mean()) if v.mean() != 0 else pd.NA

            # Normality test: Shapiro if n<=5000 else KS vs normal
            try:
                if len(v) <= 5000:
                    _, norm_p = stats.shapiro(v.sample(min(len(v), 5000), random_state=0))
                else:
                    _, norm_p = stats.kstest(
                        (v - v.mean()) / (v.std() + 1e-12), "norm"
                    )
            except Exception:
                norm_p = pd.NA

            row.update({
                "mean"             : round(float(v.mean()), 6),
                "std"              : round(float(v.std()), 6),
                "min"              : float(v.min()),
                "p01"              : float(v.quantile(0.01)),
                "p05"              : float(v.quantile(0.05)),
                "p25"              : float(q1),
                "p50"              : float(v.median()),
                "p75"              : float(q3),
                "p95"              : float(v.quantile(0.95)),
                "p99"              : float(v.quantile(0.99)),
                "max"              : float(v.max()),
                "range"            : round(float(rng), 6),
                "iqr"              : round(float(iqr_val), 6),
                "cv"               : round(float(cv), 4) if pd.notna(cv) else pd.NA,
                "skewness"         : round(float(v.skew()), 4),
                "excess_kurtosis"  : round(float(v.kurtosis()), 4),
                "pct_zero"         : round(100 * (v == 0).sum() / n_rows, 2),
                "pct_negative"     : round(100 * (v < 0).sum() / n_rows, 2),
                "pct_positive"     : round(100 * (v > 0).sum() / n_rows, 2),
                "n_outlier_iqr"    : int(((v < q1 - 1.5*iqr_val) | (v > q3 + 1.5*iqr_val)).sum()),
                "n_outlier_zscore" : int((np.abs((v - v.mean()) / (v.std() + 1e-12)) > outlier_z_thresh).sum()),
                "normality_p"      : round(float(norm_p), 4) if pd.notna(norm_p) else pd.NA,
            })

        # ── Categorical / String ──────────────────────────────────────────
        elif inferred in ("categorical", "string", "email", "url", "identifier", "binary", "boolean"):
            str_vals = non_null.astype(str)
            lens     = str_vals.str.len()
            row.update({
                "avg_str_len"      : round(float(lens.mean()), 2) if len(lens) else pd.NA,
                "min_str_len"      : int(lens.min()) if len(lens) else pd.NA,
                "max_str_len"      : int(lens.max()) if len(lens) else pd.NA,
                "pct_has_digit"    : round(100 * str_vals.str.contains(r"\d").sum() / max(len(str_vals), 1), 2),
                "pct_has_space"    : round(100 * str_vals.str.contains(r"\s").sum() / max(len(str_vals), 1), 2),
                "pct_upper"        : round(100 * str_vals.str.isupper().sum() / max(len(str_vals), 1), 2),
                "pct_lower"        : round(100 * str_vals.str.islower().sum() / max(len(str_vals), 1), 2),
                "has_email_pattern": bool(str_vals.str.contains(r"[^@\s]+@[^@\s]+\.[^@\s]+").any()),
                "has_url_pattern"  : bool(str_vals.str.contains(r"https?://").any()),
            })

        # ── Datetime ──────────────────────────────────────────────────────
        elif inferred == "datetime":
            dt_vals = pd.to_datetime(non_null, errors="coerce").dropna()
            if len(dt_vals):
                dow_counts = dt_vals.dt.dayofweek.value_counts()
                row.update({
                    "dt_min"              : dt_vals.min(),
                    "dt_max"              : dt_vals.max(),
                    "dt_range_days"       : (dt_vals.max() - dt_vals.min()).days,
                    "dt_most_common_year" : int(dt_vals.dt.year.mode().iloc[0]),
                    "dt_most_common_dow"  : int(dow_counts.index[0]),
                    "dt_weekend_pct"      : round(100 * (dt_vals.dt.dayofweek >= 5).sum() / len(dt_vals), 2),
                })

        records.append(row)

    profile = pd.DataFrame(records).set_index("column")

    if display_styled:
        try:
            from IPython.display import display as ipy_display
            ipy_display(_style_profile(profile))
        except Exception:
            pass

    return profile


def _style_profile(profile: pd.DataFrame):
    """Traffic-light conditional formatting for Jupyter display."""
    def color_missing(v):
        if pd.isna(v): return ""
        if v > 20:  return "background:#f28b82;color:#4a1010"
        if v > 5:   return "background:#fdd663;color:#3a2a00"
        return "background:#81c995;color:#0a3a1a"

    def color_skew(v):
        if pd.isna(v): return ""
        if abs(v) > 2: return "background:#f28b82;color:#4a1010"
        if abs(v) > 1: return "background:#fdd663;color:#3a2a00"
        return ""

    def color_normality(v):
        if pd.isna(v): return ""
        if v < 0.01: return "background:#f28b82;color:#4a1010"
        if v < 0.05: return "background:#fdd663;color:#3a2a00"
        return ""

    fmt = {}
    pct_cols  = [c for c in profile.columns if c.startswith("pct_")]
    stat_cols = ["mean","std","min","p01","p05","p25","p50","p75","p95","p99",
                 "max","range","iqr","cv","skewness","excess_kurtosis","normality_p"]
    for c in pct_cols:
        fmt[c] = "{:.2f}%"
    for c in stat_cols:
        if c in profile.columns:
            fmt[c] = "{:.4f}"

    styler = profile.style.format(fmt, na_rep="—")

    if "pct_missing" in profile.columns:
        styler = styler.applymap(color_missing, subset=["pct_missing"])
    if "skewness" in profile.columns:
        styler = styler.applymap(color_skew, subset=["skewness"])
    if "normality_p" in profile.columns:
        styler = styler.applymap(color_normality, subset=["normality_p"])

    styler.set_table_styles([
        {"selector": "th", "props": [("font-size","11px"),("white-space","nowrap"),("text-align","left")]},
        {"selector": "td", "props": [("font-size","11px"),("white-space","nowrap")]},
    ])
    return styler


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PLOT HELPERS  (all return base64-encoded PNG strings)
# ══════════════════════════════════════════════════════════════════════════════

def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=FIG_DPI)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _b64_img(b64: str, alt: str = "", width: str = "100%") -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="width:{width};max-width:900px">'


# ── 3a. Missing data heatmap + bar ────────────────────────────────────────────

def plot_missing(df: pd.DataFrame) -> dict[str, str]:
    missing = df.isnull()
    miss_pct = missing.mean().sort_values(ascending=False) * 100
    cols_with_miss = miss_pct[miss_pct > 0]

    out = {}

    # Bar chart of missing %
    fig, ax = plt.subplots(figsize=(max(6, len(miss_pct) * 0.35), 4))
    colors = ["#e15759" if p > 20 else "#f28e2b" if p > 5 else "#76b7b2"
              for p in miss_pct]
    ax.bar(miss_pct.index, miss_pct.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(5,  color="#f28e2b", linewidth=1, linestyle="--", label="5% threshold")
    ax.axhline(20, color="#e15759", linewidth=1, linestyle="--", label="20% threshold")
    ax.set_ylabel("% missing")
    ax.set_title("Missing values by column")
    ax.legend(fontsize=7)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out["missing_bar"] = _fig_to_b64(fig)

    # Heatmap matrix (only if there's any missingness)
    if cols_with_miss.empty:
        out["missing_heatmap"] = None
    else:
        sample = missing[cols_with_miss.index]
        if len(sample) > 300:
            sample = sample.sample(300, random_state=42)
        fig, ax = plt.subplots(figsize=(max(6, len(cols_with_miss) * 0.5), 5))
        sns.heatmap(
            sample.T, cbar=False, cmap=["#dce9f5", "#e15759"],
            yticklabels=True, xticklabels=False, ax=ax, linewidths=0,
        )
        ax.set_title("Missing value matrix (sample rows × columns with nulls)")
        ax.set_xlabel("Row index (sampled)")
        plt.tight_layout()
        out["missing_heatmap"] = _fig_to_b64(fig)

    return out


# ── 3b. Correlation heatmaps ──────────────────────────────────────────────────

def plot_correlations(df: pd.DataFrame) -> dict[str, str]:
    num_df = df.select_dtypes(include="number").dropna(axis=1, how="all")
    out    = {}

    if num_df.shape[1] < 2:
        return out

    for method in ("pearson", "spearman"):
        corr = num_df.corr(method=method, numeric_only=True)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(max(6, len(corr) * 0.55), max(5, len(corr) * 0.5)))
        sns.heatmap(
            corr, mask=mask, annot=len(corr) <= 15,
            fmt=".2f", cmap=PALETTE_HEAT, center=0, vmin=-1, vmax=1,
            linewidths=0.4, ax=ax, annot_kws={"size": 7},
            square=True, cbar_kws={"shrink": 0.7},
        )
        ax.set_title(f"{method.capitalize()} correlation matrix")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        out[f"corr_{method}"] = _fig_to_b64(fig)

    # High-correlation pairs table (|r| > 0.7)
    corr_p = num_df.corr(method="pearson", numeric_only=True)
    pairs  = []
    cols   = list(corr_p.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_p.iloc[i, j]
            if abs(r) > 0.7:
                pairs.append({"col_a": cols[i], "col_b": cols[j], "pearson_r": round(r, 4)})
    out["high_corr_pairs"] = pd.DataFrame(pairs).sort_values("pearson_r", key=abs, ascending=False) \
        if pairs else pd.DataFrame(columns=["col_a", "col_b", "pearson_r"])

    return out


# ── 3c. Per-column univariate plots ──────────────────────────────────────────

def plot_univariate(df: pd.DataFrame, profile: pd.DataFrame) -> dict[str, str]:
    """
    For each column, produce the most relevant univariate plot:
    - numeric     : histogram + KDE overlay + box plot + QQ plot (2×2)
    - categorical : horizontal bar (top N) + pie if n_unique <= 6
    - datetime    : time series count + day-of-week + hour distribution
    - boolean     : binary bar
    """
    out = {}

    for col in df.columns:
        s        = df[col]
        inferred = profile.loc[col, "inferred_type"] if col in profile.index else _infer_type(s)

        try:
            if inferred in ("numeric",):
                out[col] = _plot_numeric(s, col)
            elif inferred in ("categorical", "string", "identifier", "email", "url"):
                out[col] = _plot_categorical(s, col)
            elif inferred in ("boolean", "binary"):
                out[col] = _plot_binary(s, col)
            elif inferred == "datetime":
                out[col] = _plot_datetime(s, col)
        except Exception as e:
            out[col] = None  # silently skip broken columns

    return out


def _plot_numeric(s: pd.Series, col: str) -> str:
    v     = s.dropna().astype(float)
    q1    = v.quantile(0.25)
    q3    = v.quantile(0.75)
    iqr_v = q3 - q1

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(f"{col}  (n={len(v):,}  |  {s.isna().sum()} missing)", fontsize=11, y=1.01)

    # Histogram + KDE
    ax = axes[0, 0]
    ax.hist(v, bins=min(60, max(10, len(v) // 50)), color=PALETTE_NUMERIC,
            edgecolor="white", linewidth=0.4, density=True, alpha=0.75)
    try:
        kde = stats.gaussian_kde(v)
        xs  = np.linspace(v.min(), v.max(), 300)
        ax.plot(xs, kde(xs), color="#c44e52", linewidth=1.5, label="KDE")
    except Exception:
        pass
    ax.axvline(v.mean(),   color="#dd8452", linewidth=1.2, linestyle="--", label=f"mean={v.mean():.3g}")
    ax.axvline(v.median(), color="#4c72b0", linewidth=1.2, linestyle=":", label=f"median={v.median():.3g}")
    ax.set_title("Histogram + KDE"); ax.legend(fontsize=7)

    # Box plot
    ax = axes[0, 1]
    bp = ax.boxplot(v, vert=True, patch_artist=True,
                    flierprops=dict(marker=".", markersize=2, alpha=0.4, markerfacecolor="#c44e52"),
                    boxprops=dict(facecolor="#aec7e8", linewidth=0.8),
                    medianprops=dict(color="#c44e52", linewidth=1.5))
    outliers_iqr = v[(v < q1 - 1.5*iqr_v) | (v > q3 + 1.5*iqr_v)]
    ax.set_title(f"Box plot  ({len(outliers_iqr)} IQR outliers)")
    ax.set_xticks([])

    # Empirical CDF
    ax = axes[1, 0]
    sorted_v = np.sort(v)
    cdf      = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
    ax.plot(sorted_v, cdf, color=PALETTE_NUMERIC, linewidth=1.5)
    ax.axhline(0.05, color="gray", linewidth=0.7, linestyle="--", label="p5")
    ax.axhline(0.95, color="gray", linewidth=0.7, linestyle="--", label="p95")
    ax.set_title("Empirical CDF"); ax.set_ylabel("Cumulative probability"); ax.legend(fontsize=7)

    # QQ plot vs normal
    ax = axes[1, 1]
    osm, osr = stats.probplot(v, dist="norm")
    ax.scatter(osm[0], osm[1], s=5, alpha=0.5, color=PALETTE_NUMERIC)
    # fit line
    fit = np.polyfit(osm[0], osm[1], 1)
    ax.plot(osm[0], np.polyval(fit, osm[0]), color="#c44e52", linewidth=1.5)
    skw = v.skew()
    ax.set_title(f"QQ plot (skew={skw:.3f})")
    ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Sample quantiles")

    plt.tight_layout()
    return _fig_to_b64(fig)


def _plot_categorical(s: pd.Series, col: str) -> str:
    vc  = s.value_counts(dropna=False).head(MAX_CAT_BARS)
    n_u = s.nunique(dropna=True)

    if n_u <= 6:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, max(4, len(vc) * 0.35)))
        axes      = [axes]

    fig.suptitle(f"{col}  (n={len(s):,}  |  {n_u} unique)", fontsize=11)

    # Horizontal bar
    ax = axes[0]
    labels = [str(l) if pd.notna(l) else "NaN" for l in vc.index]
    labels = [textwrap.shorten(l, width=30, placeholder="…") for l in labels]
    colors = [sns.color_palette("muted", len(vc))[i] for i in range(len(vc))]
    bars   = ax.barh(labels[::-1], vc.values[::-1], color=colors[::-1],
                     edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, vc.values[::-1]):
        pct = 100 * val / len(s.dropna())
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{pct:.1f}%", va="center", fontsize=7)
    ax.set_title(f"Top {min(MAX_CAT_BARS, n_u)} value counts")
    ax.set_xlabel("Count")

    # Pie chart for low-cardinality
    if n_u <= 6 and len(axes) > 1:
        ax2 = axes[1]
        vc2 = s.value_counts(dropna=True).head(6)
        wedge_labels = [str(l) for l in vc2.index]
        ax2.pie(vc2.values, labels=wedge_labels, autopct="%1.1f%%",
                startangle=90, colors=sns.color_palette("muted", len(vc2)),
                wedgeprops={"edgecolor": "white", "linewidth": 1})
        ax2.set_title("Distribution (excl. NaN)")

    plt.tight_layout()
    return _fig_to_b64(fig)


def _plot_binary(s: pd.Series, col: str) -> str:
    vc  = s.value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    labels  = [str(l) if pd.notna(l) else "NaN" for l in vc.index]
    ax.bar(labels, vc.values, color=["#4c72b0", "#dd8452", "#c44e52"][:len(vc)],
           edgecolor="white", linewidth=0.8, width=0.5)
    for i, (lbl, val) in enumerate(zip(labels, vc.values)):
        ax.text(i, val + 0.5, f"{100*val/len(s.dropna()):.1f}%",
                ha="center", fontsize=9)
    ax.set_title(f"{col}  — binary / boolean distribution")
    ax.set_ylabel("Count")
    plt.tight_layout()
    return _fig_to_b64(fig)


def _plot_datetime(s: pd.Series, col: str) -> str:
    dt  = pd.to_datetime(s, errors="coerce").dropna()
    if len(dt) == 0:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"{col}  (n={len(dt):,}  |  range: {dt.min().date()} → {dt.max().date()})", fontsize=10)

    # Time-series count (monthly)
    ax = axes[0]
    monthly = dt.dt.to_period("M").value_counts().sort_index()
    ax.fill_between(range(len(monthly)), monthly.values, alpha=0.6, color=PALETTE_NUMERIC)
    ax.plot(range(len(monthly)), monthly.values, color=PALETTE_NUMERIC, linewidth=1.2)
    step = max(1, len(monthly) // 6)
    ax.set_xticks(range(0, len(monthly), step))
    ax.set_xticklabels([str(monthly.index[i]) for i in range(0, len(monthly), step)],
                       rotation=45, ha="right", fontsize=7)
    ax.set_title("Monthly frequency"); ax.set_ylabel("Count")

    # Day of week
    ax = axes[1]
    dow_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    dow     = dt.dt.dayofweek.value_counts().sort_index()
    ax.bar([dow_map.get(i, str(i)) for i in dow.index], dow.values,
           color=[("#c44e52" if i >= 5 else PALETTE_NUMERIC) for i in dow.index],
           edgecolor="white", linewidth=0.5)
    ax.set_title("Day of week"); ax.set_ylabel("Count")

    # Hour of day (if time component exists)
    ax = axes[2]
    if dt.dt.hour.std() > 0:
        hour = dt.dt.hour.value_counts().sort_index()
        ax.bar(hour.index, hour.values, color=PALETTE_NUMERIC, edgecolor="white", linewidth=0.5)
        ax.set_title("Hour of day"); ax.set_xlabel("Hour")
    else:
        month = dt.dt.month.value_counts().sort_index()
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        ax.bar([month_names[i-1] for i in month.index], month.values,
               color=PALETTE_NUMERIC, edgecolor="white", linewidth=0.5)
        ax.set_title("Month distribution")
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    return _fig_to_b64(fig)


# ── 3d. Scatter matrix (pairplot) ─────────────────────────────────────────────

def plot_pairplot(df: pd.DataFrame, hue_col: Optional[str] = None) -> Optional[str]:
    num_cols = df.select_dtypes(include="number").columns.tolist()[:MAX_PAIR_COLS]
    if len(num_cols) < 2:
        return None

    sample = df[num_cols].dropna()
    if len(sample) > 2000:
        sample = sample.sample(2000, random_state=42)

    plot_df = sample.copy()
    if hue_col and hue_col in df.columns:
        if hue_col not in plot_df.columns:
            plot_df = plot_df.join(df[[hue_col]], how="left")
        plot_df[hue_col] = plot_df[hue_col].astype(str)
        hue_use = hue_col
    else:
        hue_use = None

    try:
        g = sns.pairplot(plot_df, vars=num_cols, hue=hue_use,
                         diag_kind="kde", plot_kws={"alpha": 0.3, "s": 10},
                         diag_kws={"fill": True})
        g.fig.suptitle("Scatter matrix (numeric columns)", y=1.01, fontsize=11)
        g.fig.set_size_inches(2.5 * len(num_cols), 2.5 * len(num_cols))
        return _fig_to_b64(g.fig)
    except Exception:
        return None


# ── 3e. Outlier summary boxplots ─────────────────────────────────────────────

def plot_outlier_boxplots(df: pd.DataFrame) -> Optional[str]:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return None

    ncols = min(4, len(num_cols))
    nrows = int(np.ceil(len(num_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3))
    axes_flat = np.array(axes).flatten()

    for i, col in enumerate(num_cols):
        ax  = axes_flat[i]
        v   = df[col].dropna().astype(float)
        ax.boxplot(v, vert=True, patch_artist=True, widths=0.6,
                   flierprops=dict(marker=".", markersize=3, alpha=0.4, markerfacecolor="#c44e52"),
                   boxprops=dict(facecolor="#aec7e8", linewidth=0.8),
                   medianprops=dict(color="#c44e52", linewidth=1.5),
                   whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8))
        ax.set_title(col, fontsize=8)
        ax.set_xticks([])

    for j in range(len(num_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Numeric column box plots", fontsize=11)
    plt.tight_layout()
    return _fig_to_b64(fig)


# ── 3f. Value counts table for all columns ────────────────────────────────────

def compute_value_counts(df: pd.DataFrame, top_n: int = 20) -> dict[str, pd.DataFrame]:
    """Returns a dict of col → value_counts DataFrame with count + pct columns."""
    out = {}
    for col in df.columns:
        vc = df[col].value_counts(dropna=False, normalize=False).head(top_n).reset_index()
        vc.columns = [col, "count"]
        vc["pct"]  = (100 * vc["count"] / len(df)).round(2)
        out[col]   = vc
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 4.  HTML REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
  :root {{
    --bg:#f7f8fc; --card:#fff; --accent:#4c72b0;
    --accent2:#dd8452; --danger:#e15759; --warn:#f28e2b; --ok:#59a14f;
    --text:#22283a; --muted:#6b7280; --border:#e2e5ef;
    --mono:'Menlo','Consolas',monospace;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
        background:var(--bg);color:var(--text);font-size:14px;line-height:1.6}}
  h1{{font-size:1.7rem;font-weight:700;color:var(--accent)}}
  h2{{font-size:1.15rem;font-weight:600;color:var(--text);margin:0 0 12px}}
  h3{{font-size:.95rem;font-weight:600;color:var(--muted);margin:0 0 8px}}
  .wrapper{{max-width:1200px;margin:0 auto;padding:28px 20px}}
  .hero{{background:linear-gradient(135deg,#e8edf8 0%,#f7f8fc 100%);
         border-radius:12px;padding:28px 32px;margin-bottom:28px;
         border:1px solid var(--border)}}
  .hero-meta{{display:flex;gap:32px;margin-top:16px;flex-wrap:wrap}}
  .kpi{{text-align:center;background:var(--card);border-radius:8px;
        padding:14px 20px;border:1px solid var(--border);min-width:120px}}
  .kpi-val{{font-size:1.6rem;font-weight:700;color:var(--accent)}}
  .kpi-lbl{{font-size:.78rem;color:var(--muted);margin-top:2px}}
  .card{{background:var(--card);border-radius:10px;border:1px solid var(--border);
         padding:22px 24px;margin-bottom:22px;overflow:hidden}}
  .tabs{{display:flex;gap:6px;margin-bottom:18px;flex-wrap:wrap}}
  .tab{{padding:6px 14px;border-radius:20px;cursor:pointer;font-size:.82rem;
        font-weight:500;border:1px solid var(--border);background:var(--bg);
        color:var(--muted);transition:all .15s}}
  .tab.active,.tab:hover{{background:var(--accent);color:#fff;border-color:var(--accent)}}
  .section{{display:none}}.section.active{{display:block}}
  /* Stats table */
  .tbl-wrap{{overflow-x:auto}}
  table.stats{{width:100%;border-collapse:collapse;font-size:12px;
               font-family:var(--mono)}}
  table.stats thead tr{{background:#f0f3fa}}
  table.stats th{{padding:7px 10px;text-align:left;white-space:nowrap;
                  font-weight:600;border-bottom:2px solid var(--border)}}
  table.stats td{{padding:6px 10px;border-bottom:1px solid #f0f0f0;
                  white-space:nowrap;max-width:260px;overflow:hidden;
                  text-overflow:ellipsis}}
  table.stats tr:hover td{{background:#f7f8fc}}
  .pill{{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:500}}
  .pill-numeric{{background:#dbeafe;color:#1d4ed8}}
  .pill-categorical{{background:#dcfce7;color:#15803d}}
  .pill-datetime{{background:#fef9c3;color:#854d0e}}
  .pill-boolean{{background:#ede9fe;color:#6d28d9}}
  .pill-string{{background:#ffedd5;color:#9a3412}}
  .pill-identifier{{background:#f3f4f6;color:#374151}}
  .pill-binary{{background:#ede9fe;color:#6d28d9}}
  .pill-email,.pill-url{{background:#fce7f3;color:#9d174d}}
  /* Miss heat */
  td.m-ok{{background:#dcfce7;color:#15803d}}
  td.m-warn{{background:#fef9c3;color:#854d0e}}
  td.m-danger{{background:#fee2e2;color:#991b1b}}
  /* Univariate grid */
  .uni-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:16px}}
  .uni-card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px;overflow:hidden}}
  .uni-card h3{{font-size:.85rem;margin-bottom:8px}}
  .uni-card img{{width:100%;border-radius:4px}}
  /* Value counts */
  .vc-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}}
  .vc-card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px}}
  .vc-card h3{{font-size:.85rem;margin-bottom:8px}}
  table.vc{{width:100%;border-collapse:collapse;font-size:11px}}
  table.vc th{{padding:4px 6px;text-align:left;background:#f0f3fa;font-weight:600}}
  table.vc td{{padding:4px 6px;border-top:1px solid #f0f0f0}}
  table.vc td.bar-cell{{width:120px}}
  .bar-bg{{background:#e9ecef;border-radius:4px;height:8px}}
  .bar-fg{{background:var(--accent);height:8px;border-radius:4px}}
  /* High-corr */
  .hc-table{{width:100%;border-collapse:collapse;font-size:12px}}
  .hc-table th{{padding:7px 12px;text-align:left;background:#f0f3fa;font-weight:600}}
  .hc-table td{{padding:7px 12px;border-top:1px solid #f0f0f0}}
  .r-pos{{color:#c44e52;font-weight:600}}
  .r-neg{{color:#4c72b0;font-weight:600}}
  /* TOC sidebar pill */
  .toc-row{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px}}
  .toc-pill{{padding:5px 12px;background:var(--bg);border:1px solid var(--border);
             border-radius:20px;cursor:pointer;font-size:.8rem;color:var(--muted)}}
  .toc-pill:hover{{background:var(--accent);color:#fff;border-color:var(--accent)}}
  a.anchor{{color:inherit;text-decoration:none}}
  .section-label{{font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;
                  color:var(--muted);margin-bottom:6px}}
</style>
</head>
<body>
<div class="wrapper">

<!-- ── Hero ── -->
<div class="hero">
  <h1>{title}</h1>
  <div style="color:var(--muted);font-size:.85rem;margin-top:4px">
    Generated {generated_at} &nbsp;·&nbsp; {n_rows:,} rows &nbsp;·&nbsp; {n_cols} columns
  </div>
  <div class="hero-meta">
    {kpis}
  </div>
</div>

<!-- ── Nav tabs ── -->
<div class="tabs" id="nav">
  <div class="tab active" onclick="showSection('overview')">Overview</div>
  <div class="tab" onclick="showSection('missing')">Missing data</div>
  <div class="tab" onclick="showSection('univariate')">Distributions</div>
  <div class="tab" onclick="showSection('valuecounts')">Value counts</div>
  <div class="tab" onclick="showSection('outliers')">Outliers</div>
  <div class="tab" onclick="showSection('correlations')">Correlations</div>
  <div class="tab" onclick="showSection('pairplot')">Pair plot</div>
</div>

<!-- ══ 1. Overview ══ -->
<div class="section active" id="s-overview">
  <div class="card">
    <h2>Column profile summary</h2>
    <div class="tbl-wrap">{stats_table}</div>
  </div>
</div>

<!-- ══ 2. Missing ══ -->
<div class="section" id="s-missing">
  <div class="card">
    <h2>Missing values — bar chart</h2>
    {missing_bar}
  </div>
  {missing_heatmap_card}
</div>

<!-- ══ 3. Univariate ══ -->
<div class="section" id="s-univariate">
  <div class="card">
    <h2>Univariate distributions</h2>
    <div class="uni-grid">{uni_cards}</div>
  </div>
</div>

<!-- ══ 4. Value counts ══ -->
<div class="section" id="s-valuecounts">
  <div class="card">
    <h2>Value counts (top {top_n} per column)</h2>
    <div class="vc-grid">{vc_cards}</div>
  </div>
</div>

<!-- ══ 5. Outliers ══ -->
<div class="section" id="s-outliers">
  <div class="card">
    <h2>Box plots — numeric columns</h2>
    {outlier_plot}
  </div>
  <div class="card">
    <h2>Outlier count summary</h2>
    <div class="tbl-wrap">{outlier_table}</div>
  </div>
</div>

<!-- ══ 6. Correlations ══ -->
<div class="section" id="s-correlations">
  <div class="card">
    <h2>Pearson correlation matrix</h2>
    {corr_pearson}
  </div>
  <div class="card">
    <h2>Spearman correlation matrix</h2>
    {corr_spearman}
  </div>
  <div class="card">
    <h2>High-correlation pairs  (|r| &gt; 0.70)</h2>
    {high_corr_table}
  </div>
</div>

<!-- ══ 7. Pair plot ══ -->
<div class="section" id="s-pairplot">
  <div class="card">
    <h2>Scatter matrix (up to {max_pair_cols} numeric columns, 2,000 row sample)</h2>
    {pairplot}
  </div>
</div>

</div><!-- /wrapper -->

<script>
function showSection(id){{
  document.querySelectorAll('.section').forEach(s=>s.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('s-'+id).classList.add('active');
  event.target.classList.add('active');
}}
</script>
</body>
</html>
"""


def _make_kpi(val, label):
    return f'<div class="kpi"><div class="kpi-val">{val}</div><div class="kpi-lbl">{label}</div></div>'


def _df_to_html_table(profile: pd.DataFrame, css_class: str = "stats") -> str:
    type_pill = {
        "numeric"    : "pill-numeric",
        "categorical": "pill-categorical",
        "datetime"   : "pill-datetime",
        "boolean"    : "pill-boolean",
        "binary"     : "pill-binary",
        "string"     : "pill-string",
        "identifier" : "pill-identifier",
        "email"      : "pill-email",
        "url"        : "pill-url",
    }

    col_display = [
        "dtype", "inferred_type", "n_missing", "pct_missing",
        "n_unique", "pct_unique", "mode", "pct_mode",
        "mean", "std", "min", "p25", "p50", "p75", "max",
        "skewness", "excess_kurtosis", "n_outlier_iqr", "normality_p",
        "avg_str_len", "dt_min", "dt_max", "dt_range_days",
    ]
    cols = [c for c in col_display if c in profile.columns]
    disp = profile[cols].copy()

    miss_css = {c: ("m-danger" if v > 20 else "m-warn" if v > 5 else "m-ok")
                for c, v in profile["pct_missing"].items()} if "pct_missing" in profile.columns else {}

    html = [f'<table class="{css_class}"><thead><tr>',
            "<th>column</th>"]
    for c in cols:
        html.append(f"<th>{c}</th>")
    html.append("</tr></thead><tbody>")

    for col, row in disp.iterrows():
        html.append(f"<tr><td><strong>{col}</strong></td>")
        for c in cols:
            v = row[c]
            td_class = ""
            if c == "pct_missing" and col in miss_css:
                td_class = f' class="{miss_css[col]}"'
            if pd.isna(v):
                cell = "—"
            elif c == "inferred_type":
                pill_cls = type_pill.get(str(v), "pill-string")
                cell = f'<span class="pill {pill_cls}">{v}</span>'
            elif c in ("pct_missing", "pct_unique", "pct_mode"):
                cell = f"{float(v):.2f}%"
            elif c in ("mean","std","skewness","excess_kurtosis","normality_p","cv"):
                cell = f"{float(v):.4f}"
            elif c in ("min","p25","p50","p75","max","p01","p99"):
                cell = f"{float(v):.4g}"
            elif c in ("dt_min","dt_max"):
                cell = str(v)[:10]
            else:
                cell = str(v)
            html.append(f"<td{td_class}>{cell}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")
    return "".join(html)


def _vc_card(col: str, vc_df: pd.DataFrame) -> str:
    max_count = vc_df["count"].max() if len(vc_df) else 1
    rows = []
    for _, r in vc_df.iterrows():
        val  = str(r.iloc[0]) if pd.notna(r.iloc[0]) else "NaN"
        val  = textwrap.shorten(val, width=30, placeholder="…")
        cnt  = int(r["count"])
        pct  = float(r["pct"])
        pct_bar = int(100 * cnt / max_count)
        rows.append(
            f"<tr><td>{val}</td><td>{cnt:,}</td><td>{pct:.1f}%</td>"
            f'<td class="bar-cell"><div class="bar-bg"><div class="bar-fg" style="width:{pct_bar}%"></div></div></td></tr>'
        )
    return (
        f'<div class="vc-card"><h3>{col}</h3>'
        f'<table class="vc"><thead><tr><th>value</th><th>count</th>'
        f'<th>%</th><th style="width:120px">freq</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _outlier_table_html(profile: pd.DataFrame) -> str:
    cols = ["n_outlier_iqr", "n_outlier_zscore"]
    num_profile = profile[[c for c in cols if c in profile.columns]].dropna(how="all")
    if num_profile.empty:
        return "<p>No numeric columns found.</p>"
    html = ['<table class="stats"><thead><tr><th>column</th>']
    for c in num_profile.columns:
        html.append(f"<th>{c}</th>")
    html.append("</tr></thead><tbody>")
    for col, row in num_profile.iterrows():
        html.append(f"<tr><td><strong>{col}</strong></td>")
        for c in num_profile.columns:
            v = row[c]
            css = ""
            if pd.notna(v) and v > 0:
                css = ' style="color:#c44e52;font-weight:600"'
            html.append(f"<td{css}>{int(v) if pd.notna(v) else '—'}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")
    return "".join(html)


def _high_corr_html(pairs_df: pd.DataFrame) -> str:
    if pairs_df.empty:
        return "<p style='color:var(--muted)'>No pairs with |r| &gt; 0.70 found.</p>"
    rows = []
    for _, r in pairs_df.iterrows():
        rc = "r-pos" if r["pearson_r"] > 0 else "r-neg"
        rows.append(
            f"<tr><td>{r['col_a']}</td><td>{r['col_b']}</td>"
            f'<td class="{rc}">{r["pearson_r"]:.4f}</td></tr>'
        )
    return (
        '<table class="hc-table"><thead><tr>'
        "<th>Column A</th><th>Column B</th><th>Pearson r</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MASTER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(
    df: pd.DataFrame,
    output_path: str = "eda_report.html",
    title: str = "EDA Report",
    hue_col: Optional[str] = None,
    top_n_values: int = 20,
) -> str:
    """
    Run full EDA pipeline and write a self-contained HTML report.

    Parameters
    ----------
    df           : Input DataFrame.
    output_path  : File path for output HTML.
    title        : Report heading.
    hue_col      : Optional column to use as hue in pair plot.
    top_n_values : Max categories to show in value-counts section.

    Returns
    -------
    str : Absolute path of written file.
    """
    import os
    print(f"[EDA] Profiling {df.shape[0]:,} rows × {df.shape[1]} columns …")

    # 1. Stats table
    profile = profile_df(df, sample_top_n=top_n_values, display_styled=False)
    stats_table = _df_to_html_table(profile)

    # 2. Missing
    print("[EDA] Computing missing data plots …")
    miss = plot_missing(df)
    missing_bar = _b64_img(miss["missing_bar"], "missing bar")
    if miss.get("missing_heatmap"):
        missing_heatmap_card = (
            '<div class="card"><h2>Missing value matrix</h2>'
            + _b64_img(miss["missing_heatmap"], "missing heatmap")
            + "</div>"
        )
    else:
        missing_heatmap_card = (
            '<div class="card" style="color:var(--muted);text-align:center;padding:32px">'
            "No missing values detected — matrix not shown.</div>"
        )

    # 3. Univariate
    print("[EDA] Building univariate distribution plots …")
    uni_plots = plot_univariate(df, profile)
    uni_cards = ""
    for col, b64 in uni_plots.items():
        if b64:
            inferred = profile.loc[col, "inferred_type"] if col in profile.index else ""
            uni_cards += (
                f'<div class="uni-card"><h3>{col} '
                f'<span style="font-weight:400;color:var(--muted);font-size:.78rem">({inferred})</span></h3>'
                f"{_b64_img(b64)}</div>"
            )

    # 4. Value counts
    print("[EDA] Computing value counts …")
    vc_all   = compute_value_counts(df, top_n=top_n_values)
    vc_cards = "".join(_vc_card(col, vc_df) for col, vc_df in vc_all.items())

    # 5. Outliers
    print("[EDA] Building outlier plots …")
    outlier_b64 = plot_outlier_boxplots(df)
    outlier_plot = _b64_img(outlier_b64) if outlier_b64 else "<p>No numeric columns.</p>"
    outlier_table = _outlier_table_html(profile)

    # 6. Correlations
    print("[EDA] Computing correlation matrices …")
    corr_out  = plot_correlations(df)
    corr_pearson  = _b64_img(corr_out["corr_pearson"])  if "corr_pearson"  in corr_out else "<p>Insufficient numeric columns.</p>"
    corr_spearman = _b64_img(corr_out["corr_spearman"]) if "corr_spearman" in corr_out else "<p>Insufficient numeric columns.</p>"
    high_corr_table = _high_corr_html(corr_out.get("high_corr_pairs", pd.DataFrame()))

    # 7. Pairplot
    print("[EDA] Generating pair plot …")
    pp_b64   = plot_pairplot(df, hue_col=hue_col)
    pairplot = _b64_img(pp_b64) if pp_b64 else "<p>Fewer than 2 numeric columns — skipped.</p>"

    # KPIs
    n_num  = int(df.select_dtypes(include="number").shape[1])
    n_cat  = int(profile["inferred_type"].isin(["categorical", "string", "identifier"]).sum())
    n_dt   = int((profile["inferred_type"] == "datetime").sum())
    n_miss = int(df.isnull().any(axis=1).sum())
    kpis   = "".join([
        _make_kpi(f"{df.shape[0]:,}", "rows"),
        _make_kpi(df.shape[1], "columns"),
        _make_kpi(n_num, "numeric cols"),
        _make_kpi(n_cat, "categorical cols"),
        _make_kpi(n_dt, "datetime cols"),
        _make_kpi(f"{100*n_miss/max(len(df),1):.1f}%", "rows w/ missing"),
    ])

    # Render
    html = _HTML_TEMPLATE.format(
        title           = title,
        generated_at    = datetime.now().strftime("%Y-%m-%d %H:%M"),
        n_rows          = df.shape[0],
        n_cols          = df.shape[1],
        kpis            = kpis,
        stats_table     = stats_table,
        missing_bar     = missing_bar,
        missing_heatmap_card = missing_heatmap_card,
        uni_cards       = uni_cards,
        vc_cards        = vc_cards,
        top_n           = top_n_values,
        outlier_plot    = outlier_plot,
        outlier_table   = outlier_table,
        corr_pearson    = corr_pearson,
        corr_spearman   = corr_spearman,
        high_corr_table = high_corr_table,
        pairplot        = pairplot,
        max_pair_cols   = MAX_PAIR_COLS,
    )

    abs_path = os.path.abspath(output_path)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[EDA] Report written → {abs_path}")
    return abs_path


# ══════════════════════════════════════════════════════════════════════════════
# 6.  QUICK-USE DEMO  (python eda_profiler.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n   = 2_000

    demo_df = pd.DataFrame({
        "loan_id"        : [f"LN{i:06d}" for i in range(n)],
        "loan_amount"    : rng.lognormal(mean=10.5, sigma=0.8, size=n),
        "interest_rate"  : rng.normal(loc=0.065, scale=0.02, size=n).clip(0.01, 0.25),
        "credit_score"   : rng.integers(300, 850, size=n).astype(float),
        "dti_ratio"      : rng.beta(2, 5, size=n),
        "loan_term_months": rng.choice([12, 24, 36, 48, 60, 72], size=n),
        "loan_purpose"   : rng.choice(["mortgage","auto","personal","business","student"], size=n,
                                       p=[0.35, 0.25, 0.20, 0.12, 0.08]),
        "state"          : rng.choice(["CA","TX","NY","FL","IL","WA","GA"], size=n),
        "is_default"     : rng.choice([0, 1], size=n, p=[0.88, 0.12]),
        "origination_date": pd.date_range("2019-01-01", periods=n, freq="6h"),
        "borrower_email" : [f"user{i}@bank.com" for i in range(n)],
        "annual_income"  : rng.lognormal(mean=11.0, sigma=0.6, size=n),
        "employment_status": rng.choice(["employed","self-employed","unemployed","retired"], size=n,
                                         p=[0.68, 0.18, 0.07, 0.07]),
        "num_open_accounts": rng.integers(0, 20, size=n).astype(float),
        "derogatory_marks" : rng.integers(0, 5, size=n).astype(float),
    })

    # Inject missingness
    for col, frac in [("credit_score", 0.04), ("dti_ratio", 0.08),
                      ("annual_income", 0.12), ("state", 0.02)]:
        mask = rng.random(n) < frac
        demo_df.loc[mask, col] = np.nan

    generate_report(demo_df, output_path="eda_report.html",
                    title="Loan Portfolio — EDA Report", hue_col="is_default")