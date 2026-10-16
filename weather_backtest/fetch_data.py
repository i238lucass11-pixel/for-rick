"""
Descarga y prepara los datos para el backtest clima NY vs indices USA.

Fuentes (accesibles desde este entorno):
  - NOAA GHCN-Daily, estacion Central Park (USW00094728): Tmax, Tmin, precipitacion,
    nieve, nieve en el suelo, viento.                      -> data/ny_daily_ghcn.csv
  - NOAA ISD (Integrated Surface Data) horario, LaGuardia (72503014732): nubosidad,
    precipitacion y tiempo presente hora a hora. Se agrega a:
       * "manana"  (00:00-09:00 ET, conocido ANTES de la apertura)  -> tradeable
       * "dia"     (07:00-17:00 ET, dia completo de mercado)        -> no tradeable
                                                                   -> data/ny_daily_isd.csv
  - S&P 500 diario OHLC 1999-2018 (Yahoo ^GSPC, empaquetado en `arch`) -> data/spx_daily.csv

Uso:  python fetch_data.py
"""
import io
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

GHCN_URL = "https://noaa-ghcn-pds.s3.amazonaws.com/csv/by_station/USW00094728.csv"
ISD_URL = "https://noaa-global-hourly-pds.s3.amazonaws.com/{year}/72503014732.csv"
ISD_YEARS = range(1990, 2019)  # 1990-1998 solo se usa para climatologia (normales)

# Cobertura de capa de nubes ISD GD1 (0=CLR,1=FEW,2=SCT,3=BKN,4=OVC,5=oscurecido)
GD_TO_FRAC = {0: 0.0, 1: 0.19, 2: 0.44, 3: 0.75, 4: 1.0, 5: 1.0}


def _get(url):
    with urllib.request.urlopen(url, timeout=300) as r:
        return r.read()


def fetch_ghcn():
    out = os.path.join(DATA, "ny_daily_ghcn.csv")
    if os.path.exists(out):
        return
    print("GHCN Central Park ...")
    raw = pd.read_csv(io.BytesIO(_get(GHCN_URL)), dtype={"DATE": str})
    raw = raw[(raw.DATE >= "19700101") & raw.Q_FLAG.isna()]
    keep = ["TMAX", "TMIN", "PRCP", "SNOW", "SNWD", "AWND"]
    w = raw[raw.ELEMENT.isin(keep)].pivot_table(index="DATE", columns="ELEMENT",
                                                values="DATA_VALUE", aggfunc="first")
    w.index = pd.to_datetime(w.index)
    w.index.name = "date"
    # unidades: temp y precip en decimas, nieve en mm, viento en decimas de m/s
    for c in ["TMAX", "TMIN", "PRCP", "AWND"]:
        w[c] = w[c] / 10.0
    w = w.rename(columns={"TMAX": "tmax_c", "TMIN": "tmin_c", "PRCP": "prcp_mm",
                          "SNOW": "snow_mm", "SNWD": "snwd_mm", "AWND": "wind_ms"})
    w.to_csv(out)


def _cloud_frac(row_gd):
    """Maxima cobertura entre las capas GD1..GD4 de una observacion."""
    best = np.nan
    for v in row_gd:
        if isinstance(v, str) and v[:1].isdigit():
            code = int(v[0])
            if code in GD_TO_FRAC:
                f = GD_TO_FRAC[code]
                best = f if np.isnan(best) else max(best, f)
    return best


def _isd_year(year):
    df = pd.read_csv(io.BytesIO(_get(ISD_URL.format(year=year))), dtype=str, low_memory=False)
    # observaciones horarias rutinarias: METAR (FM-15) desde ~1996, SAO antes
    df = df[df.REPORT_TYPE.str.strip().isin(["FM-15", "SAO"])]
    ts = pd.to_datetime(df.DATE).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    out = pd.DataFrame({"ts": ts.values})
    out["date"] = ts.dt.tz_localize(None).dt.normalize().values
    out["hour"] = ts.dt.hour.values

    # nubosidad total: GF1 (octas 0-8, 9=cielo oscurecido) y si falta, maxima capa GD
    gf = pd.to_numeric(df["GF1"].str[:2], errors="coerce") if "GF1" in df else pd.Series(np.nan, df.index)
    gf = np.where(gf <= 8, gf / 8.0, np.where(gf == 9, 1.0, np.nan))
    gd_cols = [c for c in ["GD1", "GD2", "GD3", "GD4"] if c in df]
    gd = df[gd_cols].apply(_cloud_frac, axis=1).values
    out["cloud"] = np.where(np.isnan(gf), gd, gf)

    t = df["TMP"].str.split(",", expand=True)
    tv = pd.to_numeric(t[0], errors="coerce")
    out["temp_c"] = np.where((tv.abs() < 9999) & t[1].isin(list("01459ACIMPRU")), tv / 10, np.nan)

    # precip horaria (AA1 periodo 01h, profundidad en decimas de mm)
    if "AA1" in df:
        a = df["AA1"].fillna("").str.split(",", expand=True).reindex(columns=range(4))
        per = pd.to_numeric(a[0], errors="coerce")
        dep = pd.to_numeric(a[1], errors="coerce")
        out["prcp_mm"] = np.where((per == 1) & (dep < 9999), dep / 10, np.nan)
    else:
        out["prcp_mm"] = np.nan

    # tiempo presente (MW1..MW3): lluvia/llovizna/nieve/tormenta
    mw = pd.DataFrame(index=df.index)
    for c in ["MW1", "MW2", "MW3"]:
        mw[c] = pd.to_numeric(df[c].str[:2], errors="coerce") if c in df else np.nan
    is_snow = mw.isin(list(range(70, 80)) + [83, 84, 85, 86, 87, 88]).any(axis=1)
    is_rain = mw.isin(list(range(50, 70)) + list(range(80, 83)) + list(range(91, 100))).any(axis=1)
    out["snowing"] = is_snow.values
    out["raining"] = is_rain.values | (out["prcp_mm"].fillna(0) > 0.2).values
    # una observacion por hora
    return out.drop_duplicates(subset=["date", "hour"])


def _agg(g):
    return pd.Series({
        "cloud": g.cloud.mean(),
        "temp_c": g.temp_c.mean(),
        "tmin_c": g.temp_c.min(),
        "prcp_mm": g.prcp_mm.sum(min_count=1),
        "rain_hours": g.raining.sum(),
        "snow_hours": g.snowing.sum(),
        "n_obs": len(g),
    })


def fetch_isd():
    out = os.path.join(DATA, "ny_daily_isd.csv")
    if os.path.exists(out):
        return
    rows = []
    for y in ISD_YEARS:
        print(f"ISD LaGuardia {y} ...", flush=True)
        h = _isd_year(y)
        m = h[h.hour < 9].groupby("date").apply(_agg, include_groups=False).add_prefix("am_")
        d = h[(h.hour >= 7) & (h.hour < 17)].groupby("date").apply(_agg, include_groups=False).add_prefix("day_")
        rows.append(m.join(d, how="outer"))
    isd = pd.concat(rows).sort_index()
    isd.index.name = "date"
    isd.to_csv(out)


def fetch_prices():
    out = os.path.join(DATA, "spx_daily.csv")
    if os.path.exists(out):
        return
    print("S&P 500 (arch.data.sp500) ...")
    from arch.data import sp500
    px = sp500.load()[["Open", "High", "Low", "Close", "Volume"]]
    px.index.name = "date"
    px.to_csv(out)


if __name__ == "__main__":
    fetch_ghcn()
    fetch_isd()
    fetch_prices()
    print("ok ->", DATA, file=sys.stderr)
