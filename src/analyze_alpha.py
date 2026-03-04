# src/analyze_alpha.py
# Minimal, robust alpha diagnostics for LOB features.
# - Uses NON-OVERLAP target if available (ret_fwd_{H}_nonoverlap)
# - IC (Pearson corr) for raw + tight-spread subset
# - Alpha decay plots
# - Binned mean-return plots (auto-reduces bins if too few / duplicated quantiles)
# - Toy backtest reports t-stat (NOT annualized Sharpe) to avoid misleading huge numbers
#
# Output figures -> data/figs
# Output report  -> data/report.json

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------- Config -----------------------
IN_PATH = Path("data/features.parquet")
OUT_DIR = Path("data/figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 2, 5, 10, 20]      # for decay curves
H_MAIN = 10                       # your primary horizon (matches build_features non-overlap example)
TIGHT_Q = 0.20                    # tight-spread = bottom q fraction (0.20 = bottom-20%)

# Binning for "quintile" plots (will degrade to 3/2 bins if needed)
MAX_BINS = 5
MIN_BIN_COUNT = 5                 # if any bin has < MIN_BIN_COUNT, degrade bins
MIN_N_FOR_BINS = 200              # if n < this, degrade bins (prevents shape mismatch / NaNs)

# Toy strategy parameters (thresholds are quantiles of feature z-score)
LO_Q = 0.10
HI_Q = 0.90

RNG_SEED = 0
EPS = 1e-12


# ----------------------- Helpers -----------------------
def safe_corr(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    tmp = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(tmp)
    if n < 3:
        return (np.nan, n)
    return (float(tmp.iloc[:, 0].corr(tmp.iloc[:, 1])), n)


def choose_target_column(df: pd.DataFrame, h: int) -> str:
    """Prefer non-overlap if present; else fall back to overlap."""
    non = f"ret_fwd_{h}_nonoverlap"
    raw = f"ret_fwd_{h}"
    if non in df.columns:
        y = non
        print(f"Using NON-OVERLAP target: {y}")
    elif raw in df.columns:
        y = raw
        print(f"Using target: {y} (non-overlap column not found)")
    else:
        raise KeyError(f"Target columns not found: {non} or {raw}")
    return y


def tight_spread_mask(df: pd.DataFrame, q: float) -> pd.Series:
    if "spread" not in df.columns:
        raise KeyError("Column 'spread' missing. Ensure build_features writes 'spread'.")
    spread_q = df["spread"].quantile(q)
    return df["spread"] <= spread_q


def alpha_decay(df: pd.DataFrame, feature: str, ycol: str, tag: str) -> list[tuple[int, float, int]]:
    """Correlation of feature with ret_fwd_h (or non-overlap if available only for H_MAIN)."""
    res = []
    for h in HORIZONS:
        y = f"ret_fwd_{h}"
        if y not in df.columns:
            continue
        ic, n = safe_corr(df[feature], df[y])
        res.append((h, ic, n))

    # Plot
    hs = [r[0] for r in res]
    ics = [r[1] for r in res]
    plt.figure()
    plt.plot(hs, ics, marker="o")
    plt.title(f"Alpha Decay: {feature} ({tag})")
    plt.xlabel("Horizon (samples)")
    plt.ylabel("IC (corr with log-return)")
    out = OUT_DIR / f"{feature}_{tag}_decay.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"saved {out}")
    return res


def _try_qcut_bins(x: pd.Series, k: int) -> pd.Series | None:
    """Try quantile cut into k bins; return codes 0..k-1 or None if fails."""
    try:
        return pd.qcut(x, k, labels=False, duplicates="drop")
    except Exception:
        return None


def adaptive_bins(x: pd.Series, max_bins: int = MAX_BINS) -> tuple[pd.Series, int]:
    """
    Make bin labels 0..(b-1) using qcut, degrading b if necessary.
    Handles duplicate quantiles and small n.
    """
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(x)
    # Start from max_bins, degrade to 3 then 2
    for b in [max_bins, 3, 2]:
        if n < MIN_N_FOR_BINS and b == max_bins:
            continue
        codes = _try_qcut_bins(x, b)
        if codes is None:
            continue
        # ensure we really got b unique bins
        ub = int(pd.Series(codes).nunique(dropna=True))
        if ub < 2:
            continue
        # if ub < b due to duplicates drop, accept ub
        return codes, ub
    # last resort: median split
    med = x.median()
    codes = (x > med).astype(int)
    return codes, 2


def binned_mean_plot(df: pd.DataFrame, feature: str, ycol: str, tag: str) -> pd.DataFrame:
    """
    Plot mean(y) by bins of feature. Uses adaptive qcut + ensures fixed-length arrays.
    """
    cols = [feature, ycol]
    tmp = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    n = len(tmp)
    if n < 20:
        print(f"[warn] too few rows for bins: {feature} {tag} (n={n})")
        return pd.DataFrame()

    # Compute bins on a copy of feature only
    codes, b = adaptive_bins(tmp[feature], MAX_BINS)
    tmp = tmp.loc[codes.index].copy()
    tmp["bin"] = codes.astype(int)

    grp = tmp.groupby("bin")[ycol]
    out_df = pd.DataFrame({"mean": grp.mean(), "count": grp.size()}).sort_index()

    # If bins are missing (e.g., only 0,2 present), reindex to full range
    idx = pd.Index(range(b), name="bin")
    out_df = out_df.reindex(idx)

    # Plot with safe lengths
    means = out_df["mean"].to_numpy()
    counts = out_df["count"].fillna(0).to_numpy()

    plt.figure()
    x = np.arange(b)
    plt.bar(x, np.nan_to_num(means, nan=0.0))
    plt.title(f"Binned mean return: {feature} ({tag})")
    plt.xlabel("bin (low -> high)")
    plt.ylabel(f"mean({ycol})")
    plt.xticks(x, [str(i) for i in x])
    out = OUT_DIR / f"{feature}_{tag}_b{b}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"saved {out}")
    print(out_df)
    return out_df


def toy_backtest_tstat(df: pd.DataFrame, feature: str, ycol: str, tag: str) -> dict:
    """
    Toy direction strategy:
      z = (feature - mean)/std
      pos = +1 if z > hi_q, -1 if z < lo_q, else 0
      pnl = pos * y (log-return)
    Report mean pnl + t-stat (NOT annualized Sharpe).
    """
    tmp = df[[feature, ycol]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    n = len(tmp)
    if n < 50:
        print(f"[warn] too few rows for toy backtest: {feature} {tag} (n={n})")
        return {"n": n}

    z = (tmp[feature] - tmp[feature].mean()) / (tmp[feature].std(ddof=1) + 1e-12)
    lo = z.quantile(LO_Q)
    hi = z.quantile(HI_Q)

    pos = np.zeros(len(tmp), dtype=float)
    pos[z > hi] = 1.0
    pos[z < lo] = -1.0

    strat = pos * tmp[ycol].to_numpy()
    m = float(np.nanmean(strat))
    s = float(np.nanstd(strat, ddof=1))

    # t-stat of mean return
    tstat = float((m / (s + 1e-12)) * np.sqrt(len(strat)))

    counts = pd.Series(pos).value_counts().to_dict()

    # plot equity curve (cumulative sum)
    plt.figure()
    plt.plot(np.nancumsum(strat))
    plt.title(f"Toy PnL cum-sum: {feature} ({tag})")
    plt.xlabel("obs")
    plt.ylabel("cum pnl (log-return units)")
    out = OUT_DIR / f"{feature}_{tag}_pnl.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"saved {out}")
    print(f"[{feature} {tag}] mean={m:.3e} tstat={tstat:.3f} counts={counts}")

    return {
        "n": int(len(tmp)),
        "mean": m,
        "tstat": tstat,
        "counts": {str(k): int(v) for k, v in counts.items()},
        "lo_q": float(LO_Q),
        "hi_q": float(HI_Q),
    }


# ----------------------- Main -----------------------
def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"missing input: {IN_PATH} (run build_features.py first)")

    df = pd.read_parquet(IN_PATH).sort_values("ts_ms").reset_index(drop=True)

    # Choose main target
    ycol = choose_target_column(df, H_MAIN)

    # Print how many effective targets
    n_target = int(df[ycol].replace([np.inf, -np.inf], np.nan).dropna().shape[0])
    print(f"[info] target non-NaN rows n={n_target} for {ycol}")

    # Tight subset
    tight_mask = tight_spread_mask(df, TIGHT_Q)
    df_tight = df.loc[tight_mask].copy()

    # ---- Robustify duplicated quantiles for binning only (EPS jitter) ----
    rng = np.random.default_rng(RNG_SEED)

    if "ofi_norm" in df.columns:
        df["ofi_norm_eps"] = df["ofi_norm"] + EPS * rng.standard_normal(len(df))
        df_tight["ofi_norm_eps"] = df["ofi_norm_eps"].loc[df_tight.index]

    if "ofi_norm_clip" in df.columns:
        df["ofi_norm_clip_eps"] = df["ofi_norm_clip"] + EPS * rng.standard_normal(len(df))
        df_tight["ofi_norm_clip_eps"] = df["ofi_norm_clip_eps"].loc[df_tight.index]

    # Features we care about
    # IC uses raw features; binning/pnl for OFI uses eps versions to avoid qcut collapse.
    FEATURES_IC = ["imb", "ofi_norm", "ofi_norm_clip"]
    FEATURES_BIN_PNL = ["imb", "ofi_norm_eps", "ofi_norm_clip_eps"]

    # ---------------- IC ----------------
    print("\n=== IC raw ===")
    report = {"target": ycol, "H_MAIN": int(H_MAIN), "tight_q": float(TIGHT_Q), "ic": {}, "decay": {}, "bins": {}, "toy": {}}

    for f in FEATURES_IC:
        if f not in df.columns:
            continue
        ic, n = safe_corr(df[f], df[ycol])
        print(f"{f} IC = {ic} n= {n}")
        report["ic"][f] = {"ic": ic, "n": n}

    print("\n=== IC tight ===")
    report["ic_tight"] = {}
    for f in FEATURES_IC:
        if f not in df_tight.columns:
            continue
        ic, n = safe_corr(df_tight[f], df_tight[ycol])
        print(f"{f} IC = {ic} n= {n}")
        report["ic_tight"][f] = {"ic": ic, "n": n}

    # ---------------- Decay ----------------
    # Decay always computed against ret_fwd_h (overlap). That's okay; it's descriptive.
    for f in ["imb", "ofi_norm", "ofi_norm_clip"]:
        if f not in df.columns:
            continue
        report["decay"][f] = {
            "raw": alpha_decay(df, f, ycol, "raw"),
            f"tight{int(TIGHT_Q*100)}": alpha_decay(df_tight, f, ycol, f"tight{int(TIGHT_Q*100)}"),
        }

    # ---------------- Bins + Toy PnL ----------------
    for f in FEATURES_BIN_PNL:
        if f not in df.columns:
            continue

        # raw bin plot
        bdf_raw = binned_mean_plot(df, f, ycol, "raw")
        report["bins"][f] = {"raw": bdf_raw.reset_index().to_dict(orient="records") if len(bdf_raw) else []}

        # tight bin plot
        bdf_t = binned_mean_plot(df_tight, f, ycol, f"tight{int(TIGHT_Q*100)}")
        report["bins"][f][f"tight{int(TIGHT_Q*100)}"] = bdf_t.reset_index().to_dict(orient="records") if len(bdf_t) else []

        # toy pnl (t-stat)
        report["toy"][f] = {
            "raw": toy_backtest_tstat(df, f, ycol, "raw"),
            f"tight{int(TIGHT_Q*100)}": toy_backtest_tstat(df_tight, f, ycol, f"tight{int(TIGHT_Q*100)}"),
        }

    # Save report
    out_report = Path("data/report.json")
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w") as fp:
        json.dump(report, fp, indent=2)
    print(f"\nSaved figures to: {OUT_DIR}")
    print(f"Saved report to: {out_report}")


if __name__ == "__main__":
    main()