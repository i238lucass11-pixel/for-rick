"""
precios_cache.py — Local on-disk price cache, shared by the whole pipeline
(Quant_Miner re-selection, mt5 executor, etc).

Purpose: decouple Yahoo downloads from everything else. Once a ticker is
cached, every later step (re-simulations, re-selections, retries) reads from
disk and no longer depends on the internet staying up.

Usage:
    from precios_cache import obtener_precios
    data = obtener_precios('AAPL', start='2015-01-01')   # OHLCV DataFrame

    To pre-download the whole universe before running anything else:
    py precios_cache.py precargar
"""
import os
import time

import pandas as pd

CACHE_DIR = 'precios_cache'
MAX_RETRIES = 3
DELAYS = [2, 5, 12]


def _safe_name(ticker):
    return ticker.replace('-', '_').replace('=', '_').replace('/', '_').replace('^', '')


def _cache_path(ticker):
    return os.path.join(CACHE_DIR, f"{_safe_name(ticker)}.parquet")


def tiene_cache(ticker):
    return os.path.exists(_cache_path(ticker))


def _download(ticker, start, retries=MAX_RETRIES):
    """Download with retries and backoff. Returns None if hopeless
    (no internet, invalid ticker, persistent rate-limit)."""
    import yfinance as yf
    for i in range(retries):
        try:
            if i > 0:
                time.sleep(DELAYS[min(i - 1, len(DELAYS) - 1)])
            data = yf.Ticker(ticker).history(start=start, interval='1d',
                                              auto_adjust=False)
            if data is not None and len(data) >= 100:
                return data[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            if i == retries - 1:
                print(f"    {ticker}: download failed after {retries} attempts ({e})")
    return None


def obtener_precios(ticker, start, forzar=False):
    """
    Returns a ticker's OHLCV DataFrame. Priority:
      1. On-disk cache (if present and not forcing a refresh) — trusted as-is.
      2. Download with retries; on success, save to cache.
      3. None if no cache and the download fails (e.g. no internet).

    Unlike raw yf.download, this NEVER lets a network failure masquerade as
    "possibly delisted" — it returns an explicit None so the caller decides
    (skip, mark pending, etc).

    NOTE on `start`: once a ticker is cached, `start` is IGNORED on cache
    hits — the cache is trusted outright. There is no reliable way to tell
    "cache doesn't cover the requested start" apart from "this ticker simply
    didn't trade before its cached start date" (IPO/listing date), and
    guessing wrong here re-downloads on every single call. A previous
    version tried to guess (comparing df.index.min() against `start`), which
    both (a) broke on tz-aware vs tz-naive Timestamps — silently caught by a
    broad except, forcing a re-download on EVERY cache hit — and (b) even
    once fixed, still re-downloaded late-listed tickers forever since
    nothing could ever satisfy the check. Found 2026-07 when 5 parallel
    miners re-hammered Yahoo for already-cached tickers, mass-rate-limited,
    and two of five asset groups mined zero surviving strategies. If you
    deliberately need more history than what's cached, pass forzar=True or
    delete the cached parquet.
    """
    path = _cache_path(ticker)
    if not forzar and os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass  # corrupt cache, re-download

    data = _download(ticker, start)
    if data is None:
        return None

    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = path + '.tmp'
    data.to_parquet(tmp)
    os.replace(tmp, path)
    return data


def precargar_universo(tickers, start_por_ticker=None, default_start='2010-01-01'):
    """
    Downloads and caches ALL tickers in a list in one pass, with retries.
    Resumable: if the internet drops mid-run, running it again skips
    whatever is already cached.

    start_por_ticker: optional dict {ticker: start_date}.
    """
    start_por_ticker = start_por_ticker or {}
    missing = [t for t in tickers if not tiene_cache(t)]
    print(f"Universe: {len(tickers)} tickers | already cached: {len(tickers) - len(missing)} "
          f"| to download: {len(missing)}")

    ok, fail = [], []
    for i, t in enumerate(missing, 1):
        start = start_por_ticker.get(t, default_start)
        print(f"[{i}/{len(missing)}] {t} (from {start})", end=' ... ', flush=True)
        data = obtener_precios(t, start)
        if data is None:
            print("FAILED (no internet or persistent rate-limit)")
            fail.append(t)
        else:
            print(f"OK ({len(data)} rows)")
            ok.append(t)

    print(f"\nPreload: {len(ok)} OK, {len(fail)} failed.")
    if fail:
        print("Failed (run again later to retry only these):")
        print(f"  {sorted(fail)}")
    return ok, fail


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'precargar':
        import glob
        pool = set()
        for f in glob.glob(os.path.join('estrategias_por_ticker', 'estrategias_*.csv')):
            try:
                pool.update(pd.read_csv(f, usecols=['ticker'])['ticker'].unique())
            except Exception:
                continue
        precargar_universo(sorted(pool))
    else:
        print(__doc__)
