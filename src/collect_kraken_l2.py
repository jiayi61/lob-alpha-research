import asyncio, json, time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import websockets
from tqdm import tqdm

# Kraken WS: https://docs.kraken.com/websockets/
WS_URL = "wss://ws.kraken.com/"

@dataclass
class Config:
    pair: str = "XBT/USD"      # Kraken uses XBT
    depth: int = 10            # top N levels per side
    minutes: int = 30          # collection duration
    out_dir: str = "data"
    out_name: str = "kraken_xbtusd_book.parquet"

def now_ms() -> int:
    return int(time.time() * 1000)

def parse_book_update(msg: dict) -> list[dict]:
    """
    Kraken book update has fields like:
      {"a":[["price","size","timestamp"], ...], "b":[...], "c":"checksum"}
    We’ll emit rows per level update (not full snapshot).
    """
    rows = []
    ts = now_ms()
    for side_key, side in [("a", "ask"), ("b", "bid")]:
        if side_key in msg:
            for lvl in msg[side_key]:
                price = float(lvl[0])
                size = float(lvl[1])
                rows.append({"ts_ms": ts, "side": side, "price": price, "size": size})
    return rows

async def collect(cfg: Config) -> Path:
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(cfg.out_dir) / cfg.out_name

    # Local in-memory book state (top depth) to reconstruct snapshots periodically if needed
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}

    rows_all: list[dict] = []
    end_time = time.time() + cfg.minutes * 60

    sub = {
        "event": "subscribe",
        "pair": [cfg.pair],
        "subscription": {"name": "book", "depth": cfg.depth},
    }

    async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
        await ws.send(json.dumps(sub))

        pbar = tqdm(total=int(cfg.minutes * 60), desc="collecting (sec)")
        last_tick = time.time()

        while time.time() < end_time:
            raw = await ws.recv()
            msg = json.loads(raw)

            # subscriptionStatus / heartbeat etc.
            if isinstance(msg, dict):
                continue

            # Data messages are arrays; last fields: channelName, pair
            # Example snapshot: [channelID, {"as":[...], "bs":[...]}, "book-10", "XBT/USD"]
            # Update:          [channelID, {"a":[...], "b":[...]},  "book-10", "XBT/USD"]
            if isinstance(msg, list) and len(msg) >= 4 and isinstance(msg[1], dict):
                payload = msg[1]

                # Snapshot uses "as"/"bs"
                if "as" in payload or "bs" in payload:
                    ts = now_ms()
                    if "bs" in payload:
                        for p, s, *_ in payload["bs"]:
                            bids[float(p)] = float(s)
                            rows_all.append({"ts_ms": ts, "side": "bid", "price": float(p), "size": float(s), "type": "snapshot"})
                    if "as" in payload:
                        for p, s, *_ in payload["as"]:
                            asks[float(p)] = float(s)
                            rows_all.append({"ts_ms": ts, "side": "ask", "price": float(p), "size": float(s), "type": "snapshot"})
                else:
                    upd_rows = parse_book_update(payload)
                    for r in upd_rows:
                        r["type"] = "update"
                        rows_all.append(r)

            # progress bar tick each second
            if time.time() - last_tick >= 1.0:
                pbar.update(1)
                last_tick = time.time()

        pbar.close()

    df = pd.DataFrame(rows_all)
    df.to_parquet(out_path, index=False)
    return out_path

if __name__ == "__main__":
    cfg = Config()
    out = asyncio.run(collect(cfg))
    print(f"saved: {out}")