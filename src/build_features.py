from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

IN_PATH = Path("data/kraken_xbtusd_book.parquet")
OUT_PATH = Path("data/features.parquet")

DEPTH = 10
RESAMPLE_MS = 200
HORIZONS = [1, 2, 5, 10, 20, 50]  # ✅ include 50 for longer-horizon analysis


def top_n(book: dict[float, float], side: str, n: int):
    if not book:
        return []
    if side == "bid":
        prices = sorted(book.keys(), reverse=True)[:n]
    else:
        prices = sorted(book.keys())[:n]
    return [(p, book[p]) for p in prices]


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"missing input: {IN_PATH} (run collect_kraken_l2.py first)")

    df = pd.read_parquet(IN_PATH).sort_values("ts_ms").reset_index(drop=True)

    bids: dict[float, float] = {}
    asks: dict[float, float] = {}

    start = int(df["ts_ms"].iloc[0])
    end = int(df["ts_ms"].iloc[-1])
    grid = np.arange(start, end + RESAMPLE_MS, RESAMPLE_MS)

    rows = []
    i = 0
    prev_bid_vol = None
    prev_ask_vol = None

    for t in tqdm(grid, desc="rebuilding"):
        # apply all updates up to time t
        while i < len(df) and int(df.at[i, "ts_ms"]) <= t:
            side = df.at[i, "side"]
            price = float(df.at[i, "price"])
            size = float(df.at[i, "size"])

            # size==0 -> remove level
            if side == "bid":
                if size <= 0:
                    bids.pop(price, None)
                else:
                    bids[price] = size
            else:
                if size <= 0:
                    asks.pop(price, None)
                else:
                    asks[price] = size

            i += 1

        b = top_n(bids, "bid", DEPTH)
        a = top_n(asks, "ask", DEPTH)
        if len(b) == 0 or len(a) == 0:
            continue

        best_bid_p, best_bid_q = b[0]
        best_ask_p, best_ask_q = a[0]

        mid = 0.5 * (best_bid_p + best_ask_p)
        spread = best_ask_p - best_bid_p

        bid_vol = sum(q for _, q in b)
        ask_vol = sum(q for _, q in a)

        imb = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-12)

        # OFI proxy via depth volume changes (top-DEPTH aggregated)
        if prev_bid_vol is None:
            ofi = 0.0
        else:
            ofi = (bid_vol - prev_bid_vol) - (ask_vol - prev_ask_vol)

        ofi_norm = ofi / (bid_vol + ask_vol + 1e-12)

        prev_bid_vol = bid_vol
        prev_ask_vol = ask_vol

        rows.append(
            {
                "ts_ms": int(t),
                "best_bid_p": float(best_bid_p),
                "best_ask_p": float(best_ask_p),
                "best_bid_q": float(best_bid_q),
                "best_ask_q": float(best_ask_q),
                "mid": float(mid),
                "spread": float(spread),
                "bid_vol": float(bid_vol),
                "ask_vol": float(ask_vol),
                "imb": float(imb),
                "ofi": float(ofi),
                "ofi_norm": float(ofi_norm),
                "log_mid": float(np.log(mid)),
            }
        )

    feat = pd.DataFrame(rows).sort_values("ts_ms").reset_index(drop=True)

    # ----- clipped + jitter features (to avoid ties in quintile plots) -----
    feat["ofi_norm_clip"] = feat["ofi_norm"].clip(-1.0, 1.0)

    # tiny jitter to break ties for quantile bucketing; negligible magnitude
    rng = np.random.default_rng(7)  # fixed seed for reproducibility
    eps = 1e-12
    feat["ofi_norm_eps"] = feat["ofi_norm"] + eps * rng.standard_normal(len(feat))
    feat["ofi_norm_clip_eps"] = feat["ofi_norm_clip"] + eps * rng.standard_normal(len(feat))

    # ----- forward returns -----
    for h in HORIZONS:
        feat[f"ret_fwd_{h}"] = feat["log_mid"].shift(-h) - feat["log_mid"]

    # ----- non-overlapping sanity-check returns -----
    feat["idx"] = np.arange(len(feat))

    # non-overlap for h=10
    h_no_10 = 10
    feat["ret_fwd_10_nonoverlap"] = np.where(
        feat["idx"] % h_no_10 == 0,
        feat["ret_fwd_10"],
        np.nan,
    )

    # non-overlap for h=50
    h_no_50 = 50
    feat["ret_fwd_50_nonoverlap"] = np.where(
        feat["idx"] % h_no_50 == 0,
        feat["ret_fwd_50"],
        np.nan,
    )

    feat.to_parquet(OUT_PATH, index=False)
    print(f"saved: {OUT_PATH} rows: {len(feat)}  resample_ms={RESAMPLE_MS} depth={DEPTH}")


if __name__ == "__main__":
    main()