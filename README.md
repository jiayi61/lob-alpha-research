# Limit Order Book Alpha Research

A microstructure research project that **collects high‑frequency L2 order book data**, reconstructs evenly sampled snapshots, engineers **liquidity‑pressure features**, and evaluates their **short‑horizon predictive power**.

This repository implements an end‑to‑end pipeline:

1. **Market data collection** from a public exchange WebSocket  
2. **Order book reconstruction** into evenly sampled snapshots  
3. **Feature engineering** (depth imbalance, order‑flow imbalance)  
4. **Alpha analysis** (IC, decay curves, binned/quantile tests, toy PnL diagnostics)

Goal: test whether **order book pressure signals predict short‑term returns**.

---

## Repository Structure

```text
lob-alpha-research
│
├── README.md
├── requirements.txt
├── data/
│   └── (generated datasets – not committed to Git)
│
├── src/
│   ├── collect_kraken_l2.py
│   ├── build_features.py
│   └── analyze_alpha.py
│
└── notebooks/
    └── exploratory_analysis.ipynb
```

---

## Environment Setup (macOS friendly)

### Quick rule
- If you see `error: externally-managed-environment` (PEP 668), **don’t use system pip**. Use a **conda env** or **venv**.

### Option A — Conda (recommended if you already use Anaconda)
```bash
conda create -n lobalpha python=3.12 -y
conda activate lobalpha
python -m pip install -r requirements.txt
```

### Option B — venv (works everywhere)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Sanity check:
```bash
python -c "import pandas, numpy, websockets, pyarrow; print('ok')"
```

---

## Step 1 — Collect Order Book Data

Connects to the exchange WebSocket and records **level‑2 order book updates**.

```bash
python src/collect_kraken_l2.py
```

Output:
```text
data/kraken_xbtusd_book.parquet
```

---

## Step 2 — Build Features

Reconstruct evenly spaced order book snapshots and compute microstructure features.

```bash
python src/build_features.py
```

Generated columns include:
- Mid price (`mid`, `log_mid`)
- Bid‑ask spread (`spread`)
- Depth volumes (`bid_vol`, `ask_vol`)
- Depth imbalance (`imb`)
- OFI proxy (`ofi`, `ofi_norm`, `ofi_norm_clip`)
- Forward returns (`ret_fwd_*`) including **non‑overlapping** targets (`*_nonoverlap`)

Output:
```text
data/features.parquet
```

---

## Step 3 — Alpha Analysis

Run diagnostics / plots:

```bash
python src/analyze_alpha.py
```

The analysis produces:
- **IC** (feature vs forward return correlation)
- **alpha decay** across horizons
- **binned/quantile tests** (mean forward return by feature bins)
- **tight‑spread robustness checks** (e.g., bottom‑20% spread)
- **toy PnL** / t‑stat diagnostics

Figures are saved in:
```text
data/figs/
```

---

## Research Methodology

### Depth Imbalance

Measures liquidity pressure across LOB depth:

```text
imb = (bid_volume − ask_volume) / (bid_volume + ask_volume)
```

### Order Flow Imbalance (OFI) proxy

Captures directional pressure from changes in depth volume. Normalized:

```text
ofi_norm = ofi / (bid_volume + ask_volume)
```

---

## Typical Analysis

This pipeline evaluates:

1. **Signal ↔ future return IC** (raw + filtered)
2. **Alpha decay** across horizons
3. **Bin/quantile monotonicity** tests
4. **Spread‑conditioned robustness** checks
5. **Non‑overlap sanity checks** (avoid overlap inflation)

---

## Results (sample run)

From your current “good” run:

- Target: `ret_fwd_10_nonoverlap`
- Resample: 200 ms
- Depth: top‑10

IC summary:

```text
imb IC ≈ 0.228   (n≈898)
ofi_norm IC ≈ 0.049
ofi_norm_clip IC ≈ 0.044
```

### 1) Imbalance alpha decay (important)
![imb decay](data/figs/imb_raw_decay.png)

### 2) OFI alpha decay (important comparison)
![ofi decay](data/figs/ofi_norm_raw_decay.png)

### 3) Binned forward returns (imbalance) — monotonicity check
![imb bins](data/figs/imb_raw_b5.png)

### 4) Toy PnL / t-stat diagnostic (imbalance)
![imb pnl](data/figs/imb_raw_pnl.png)

> Only the **key plots** are embedded here for clarity. All other outputs stay in `data/figs/`.

---

## Common Pitfalls (from your logs)

- `zsh: command not found: python`  
  Use `python3 ...` or activate your env so `python` points to the correct interpreter.

- `error: externally-managed-environment` (PEP 668)  
  Use **conda** or **venv**. Avoid system-wide `pip3 install ...`.

- Binning crashes / “too many zeros” with OFI variants  
  OFI features often have many exact zeros. Typical fixes:
  - filter `!= 0`, or
  - add tiny jitter (`eps`) only for binning (not for IC), or
  - use fewer bins when `n` is small.

---

## Future Improvements

- More precise OFI (level‑by‑level)
- Transaction cost / spread crossing modeling
- Cross‑asset / multi‑venue validation
- Longer collection windows (more data → tighter confidence intervals)

---

## License

MIT License
