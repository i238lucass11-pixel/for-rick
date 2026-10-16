"""
Backtest: S&P 500 segun el "regimen climatico" del dia en Nueva York.

Idea:  LONG en dias "lindos" (soleado, seco, temperatura agradable PARA LA ESTACION)
       SHORT en dias "feos" (lluvia, nieve, nublado cerrado, frio/calor extremo)
       FLAT el resto.

Tres variantes de timing (de menos a mas realista):
  1. CONTEMPORANEO   clima del dia completo t  -> retorno close(t-1)->close(t), 1999-2018.
                     Requiere "conocer" el clima del dia antes de la apertura (pronostico
                     perfecto). Es la version academica (Saunders 1993, Hirshleifer & Shumway 2003).
  2. MANANA (intrad.) clima observado 00-09h ET del dia t -> retorno open(t)->close(t).
                     100% operable. Solo dias con Open valido (Yahoo ^GSPC repite el close
                     previo como Open en gran parte de 1999-2006), ~2006-2018.
  3. REZAGADO        clima del dia completo t-1 -> retorno close(t-1)->close(t) (orden MOC).
                     100% operable, 1999-2018.

Uso:  python fetch_data.py && python backtest.py
Salida: results/report.md + graficos PNG + CSV de senales.
"""
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

START, END = "1999-01-01", "2018-12-31"
SPLIT = "2009-01-01"            # in-sample 1999-2008 / out-of-sample 2009-2018
COST_BP = 1.0                   # costo por lado (futuros ES / SPY), en puntos basicos
N_PERM = 2000
RNG = np.random.default_rng(42)

SEASON = {12: "Invierno", 1: "Invierno", 2: "Invierno", 3: "Primavera", 4: "Primavera",
          5: "Primavera", 6: "Verano", 7: "Verano", 8: "Verano", 9: "Otono", 10: "Otono",
          11: "Otono"}

# ---------------------------------------------------------------------------
# Parametros de "dia lindo / dia feo" por estacion.
#   t_anom = temperatura vs. normal climatologica de ese dia del anio (1970/1990-1998)
#   cloud  = fraccion de cielo cubierto 0..1 (0 = despejado, 1 = cubierto)
# FEO  si se cumple CUALQUIER condicion fea.
# LINDO si no es feo Y se cumplen TODAS las condiciones lindas.
# ---------------------------------------------------------------------------
PARAMS = {
    #             nublado si >=   soleado si <=   frio feo si anom<=  frio abs feo (Tmax<=)  calor feo (Tmax>=)  lindo: anom en [lo, hi]
    "Invierno":  dict(cloud_ugly=0.95, cloud_nice=0.45, cold_anom=-5.0, cold_abs=-2.0, hot_abs=99.0, nice_lo=-2.0, nice_hi=99.0),
    "Primavera": dict(cloud_ugly=0.95, cloud_nice=0.45, cold_anom=-5.0, cold_abs=-99., hot_abs=33.0, nice_lo=-2.0, nice_hi=8.0),
    "Verano":    dict(cloud_ugly=0.90, cloud_nice=0.40, cold_anom=-99., cold_abs=-99., hot_abs=33.0, nice_lo=-4.0, nice_hi=3.0),
    "Otono":     dict(cloud_ugly=0.95, cloud_nice=0.40, cold_anom=-5.0, cold_abs=-99., hot_abs=33.0, nice_lo=-2.0, nice_hi=8.0),
}
RAIN_MM = 2.5          # dia completo: lluvia "relevante" (>= 2.5 mm entre 07 y 17 h)
RAIN_HOURS_AM = 2      # manana: >= 2 horas con lluvia/llovizna entre 00 y 09 h
SNOW_HOURS_AM = 1      # manana: >= 1 hora nevando


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def _normals(series, base_end="1998-12-31"):
    """Normal climatologica por dia del anio (media movil circular +-15 dias) usando
    SOLO datos anteriores al backtest -> sin look-ahead."""
    s = series[:base_end].dropna()
    doy = s.groupby(s.index.dayofyear.map(lambda d: min(d, 365))).mean().reindex(range(1, 366)).interpolate()
    ext = pd.concat([doy.iloc[-15:], doy, doy.iloc[:15]])
    smooth = ext.rolling(31, center=True).mean().iloc[15:-15]
    smooth.index = range(1, 366)
    return smooth


def load():
    g = pd.read_csv(os.path.join(DATA, "ny_daily_ghcn.csv"), index_col=0, parse_dates=True)
    i = pd.read_csv(os.path.join(DATA, "ny_daily_isd.csv"), index_col=0, parse_dates=True)
    p = pd.read_csv(os.path.join(DATA, "spx_daily.csv"), index_col=0, parse_dates=True)

    w = g.join(i, how="outer")
    n_tmax = _normals(w.tmax_c)
    n_am = _normals(w.am_temp_c)
    doy = w.index.dayofyear.map(lambda d: min(d, 365))
    w["tmax_anom"] = w.tmax_c - n_tmax.reindex(doy).values
    w["am_temp_anom"] = w.am_temp_c - n_am.reindex(doy).values
    w["am_tmax_proxy"] = w.am_temp_c + (n_tmax - n_am).reindex(doy).values  # Tmax esperada desde la manana
    w["season"] = w.index.month.map(SEASON)

    px = p[["Open", "Close"]].copy()
    px["r_cc"] = px.Close.pct_change()
    px["r_oc"] = px.Close / px.Open - 1
    px["open_ok"] = px.Open.round(2) != px.Close.shift().round(2)
    return w, px


# ---------------------------------------------------------------------------
# Clasificacion
# ---------------------------------------------------------------------------
def classify(w, mode, params=PARAMS, rain_mm=RAIN_MM, rain_h=RAIN_HOURS_AM, snow_h=SNOW_HOURS_AM):
    """Devuelve Serie con +1 (lindo), -1 (feo), 0 (neutral)."""
    if mode == "day":
        cloud, tmax, anom = w.day_cloud, w.tmax_c, w.tmax_anom
        wet = (w.day_prcp_mm >= rain_mm) | (w.day_rain_hours >= 4)
        snow = (w.snow_mm > 0) | (w.day_snow_hours >= 1)
    else:  # "am": solo informacion disponible antes de las 9:00 ET
        cloud, tmax, anom = w.am_cloud, w.am_tmax_proxy, w.am_temp_anom
        wet = w.am_rain_hours >= rain_h
        snow = w.am_snow_hours >= snow_h

    P = pd.DataFrame([params[s] for s in w.season], index=w.index)
    ugly = (wet | snow | (cloud >= P.cloud_ugly) | (anom <= P.cold_anom)
            | (tmax <= P.cold_abs) | (tmax >= P.hot_abs))
    nice = (~ugly & ~wet & (cloud <= P.cloud_nice) & (anom >= P.nice_lo) & (anom <= P.nice_hi)
            & (tmax < P.hot_abs))
    sig = pd.Series(0, index=w.index, dtype=int)
    sig[nice] = 1
    sig[ugly] = -1
    sig[cloud.isna() | anom.isna()] = 0
    return sig


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def build_variants(w, px, **kw):
    sig_day = classify(w, "day", **kw)
    sig_am = classify(w, "am", **kw)
    px = px[START:END]

    v = {}
    d = px[["r_cc"]].join(sig_day.rename("sig"), how="left").dropna()
    v["1_contemporaneo"] = dict(df=d.rename(columns={"r_cc": "r"}), intraday=False)

    d = px.loc[px.open_ok, ["r_oc"]].join(sig_am.rename("sig"), how="left").dropna()
    v["2_manana_intradia"] = dict(df=d.rename(columns={"r_oc": "r"}), intraday=True)

    # senal del ultimo dia calendario previo (incluye fin de semana -> viernes usa jueves, lunes usa domingo)
    prev = sig_day.shift(1)
    d = px[["r_cc"]].join(prev.rename("sig"), how="left").dropna()
    v["3_rezagado"] = dict(df=d.rename(columns={"r_cc": "r"}), intraday=False)
    for k in v:
        v[k]["df"]["sig"] = v[k]["df"]["sig"].astype(int)
        v[k]["df"]["season"] = v[k]["df"].index.month.map(SEASON)
    return v, sig_day, sig_am


def strat_returns(df, intraday, kind="ls", cost_bp=COST_BP):
    s = df.sig
    pos = {"ls": s, "long_nice": (s == 1).astype(int), "short_ugly": -(s == -1).astype(int),
           "long_not_ugly": (s != -1).astype(int), "bh": pd.Series(1, index=df.index)}[kind]
    c = cost_bp / 1e4
    if kind == "bh":
        cost = 0.0
    elif intraday:
        cost = 2 * c * pos.abs()                      # entra en open, sale en close
    else:
        cost = c * pos.diff().abs().fillna(pos.abs())
    out = pos * df.r - cost
    out.attrs["pos"] = pos
    return out


def stats(r, periods=252):
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / periods
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = r.std() * np.sqrt(periods)
    sharpe = r.mean() / r.std() * np.sqrt(periods) if r.std() > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    pos = r.attrs.get("pos")
    active = r[(pos.reindex(r.index) != 0)] if pos is not None else r[r != 0]
    return dict(CAGR=cagr, Vol=vol, Sharpe=sharpe, MaxDD=mdd,
                HitRate=(active > 0).mean() if len(active) else np.nan,
                Exposicion=len(active) / len(r) if len(r) else np.nan, Dias=len(r))


def class_table(df):
    rows = []
    for name, g in [("Todos", df)] + list(df.groupby("season")):
        for lab, k in [("Lindo", 1), ("Neutral", 0), ("Feo", -1)]:
            x = g.r[g.sig == k]
            rows.append(dict(Estacion=name, Clase=lab, N=len(x), Media_bp=x.mean() * 1e4,
                             t=x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 else np.nan,
                             PctPositivo=(x > 0).mean()))
        a, b = g.r[g.sig == 1], g.r[g.sig == -1]
        diff = a.mean() - b.mean()
        se = np.sqrt(a.var() / len(a) + b.var() / len(b))
        rows.append(dict(Estacion=name, Clase="Lindo-Feo", N=len(a) + len(b),
                         Media_bp=diff * 1e4, t=diff / se, PctPositivo=np.nan))
    return pd.DataFrame(rows)


def permutation_pvalue(df, n=N_PERM):
    """p-valor de la media de la estrategia L/S barajando etiquetas DENTRO de cada estacion
    (mantiene cuantos dias lindos/feos hay en cada estacion)."""
    obs = (df.sig * df.r).mean()
    sig = df.sig.values
    groups = [np.where(df.season.values == s)[0] for s in df.season.unique()]
    r = df.r.values
    cnt = 0
    for _ in range(n):
        perm = sig.copy()
        for idx in groups:
            perm[idx] = RNG.permutation(sig[idx])
        cnt += (perm * r).mean() >= obs
    return (cnt + 1) / (n + 1)


def cloud_regression(w, px):
    """Estilo Hirshleifer-Shumway: r_t = a + b * nubosidad desestacionalizada + e."""
    d = px[START:END][["r_cc"]].join(w.day_cloud, how="left").dropna()
    m = d.day_cloud.groupby(d.index.month).transform("mean")
    x = d.day_cloud - m
    X = np.column_stack([np.ones(len(x)), x.values])
    beta, *_ = np.linalg.lstsq(X, d.r_cc.values, rcond=None)
    resid = d.r_cc.values - X @ beta
    s2 = resid @ resid / (len(x) - 2)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return dict(beta_bp_por_100pct_nubes=beta[1] * 1e4, t=beta[1] / se[1], n=len(x))


# ---------------------------------------------------------------------------
# Robustez: grilla de umbrales
# ---------------------------------------------------------------------------
def grid(w, px):
    rows = []
    for cn in [0.25, 0.35, 0.45, 0.55]:
        for cu in [0.85, 0.90, 0.95, 1.0]:
            for rmm in [1.0, 2.5, 5.0]:
                prm = {s: {**p, "cloud_nice": cn, "cloud_ugly": cu} for s, p in PARAMS.items()}
                v, _, _ = build_variants(w, px, params=prm, rain_mm=rmm)
                for name, vv in v.items():
                    r = strat_returns(vv["df"], vv["intraday"])
                    rows.append(dict(variante=name, cloud_nice=cn, cloud_ugly=cu, rain_mm=rmm,
                                     Sharpe_IS=stats(r[:SPLIT])["Sharpe"],
                                     Sharpe_OOS=stats(r[SPLIT:])["Sharpe"],
                                     Sharpe_total=stats(r)["Sharpe"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Graficos (paleta categorica validada: azul, naranja, aqua; gris = benchmark)
# ---------------------------------------------------------------------------
C_BLUE, C_ORANGE, C_AQUA, C_GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_equity(variants, path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), facecolor="#fcfcfb")
    titles = {"1_contemporaneo": "1. Contemporaneo (clima del dia, close->close) — requiere pronostico perfecto",
              "2_manana_intradia": "2. Manana -> intradia (clima 00-09h, open->close) — operable",
              "3_rezagado": "3. Rezagado (clima de ayer, close->close) — operable"}
    for ax, (name, vv) in zip(axes, variants.items()):
        ax.set_facecolor("#fcfcfb")
        series = [("Long/Short clima", strat_returns(vv["df"], vv["intraday"], "ls"), C_BLUE),
                  ("Solo long dias lindos", strat_returns(vv["df"], vv["intraday"], "long_nice"), C_AQUA),
                  ("Long salvo dias feos", strat_returns(vv["df"], vv["intraday"], "long_not_ugly"), C_ORANGE),
                  ("Buy & hold", strat_returns(vv["df"], vv["intraday"], "bh"), C_GRAY)]
        for lab, r, col in series:
            eq = (1 + r).cumprod()
            ax.plot(eq.index, eq.values, color=col, lw=2 if col != C_GRAY else 1.5, label=lab)
            ax.annotate(f"{eq.iloc[-1]:.2f}x", (eq.index[-1], eq.iloc[-1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=8, color=INK2, va="center")
        ax.set_yscale("log")
        ax.set_title(titles[name], loc="left", fontsize=10.5, color=INK)
        ax.set_ylabel("Capital (log, base 1)", fontsize=9, color=INK2)
        _style(ax)
    axes[0].legend(loc="upper left", fontsize=8.5, frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_classes(tables, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor="#fcfcfb", sharey=True)
    seasons = ["Todos", "Invierno", "Primavera", "Verano", "Otono"]
    for ax, (name, t) in zip(axes, tables.items()):
        ax.set_facecolor("#fcfcfb")
        x = np.arange(len(seasons))
        for j, (lab, col) in enumerate([("Lindo", C_AQUA), ("Neutral", C_GRAY), ("Feo", C_ORANGE)]):
            vals = [t[(t.Estacion == s) & (t.Clase == lab)].Media_bp.iloc[0] for s in seasons]
            ax.bar(x + (j - 1) * 0.26, vals, width=0.24, color=col, label=lab, edgecolor="#fcfcfb", lw=1)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(x, seasons, fontsize=8.5)
        ax.set_title(name.split("_", 1)[1].replace("_", " "), fontsize=10, color=INK, loc="left")
        _style(ax)
    axes[0].set_ylabel("Retorno medio diario (pb)", fontsize=9, color=INK2)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_grid(g, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), facecolor="#fcfcfb")
    for ax, name in zip(axes, sorted(g.variante.unique())):
        x = g[g.variante == name]
        ax.set_facecolor("#fcfcfb")
        ax.scatter(x.Sharpe_IS, x.Sharpe_OOS, s=28, color=C_BLUE, edgecolor="#fcfcfb", lw=1)
        lim = max(0.6, np.nanmax(np.abs(x[["Sharpe_IS", "Sharpe_OOS"]].values)) * 1.1)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Sharpe in-sample 1999-2008", fontsize=9, color=INK2)
        ax.set_title(name.split("_", 1)[1].replace("_", " ") + f"  ({len(x)} combinaciones)",
                     fontsize=10, color=INK, loc="left")
        _style(ax)
    axes[0].set_ylabel("Sharpe out-of-sample 2009-2018", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
def fmt(df, floatfmt=".2f"):
    return df.to_markdown(index=False, floatfmt=floatfmt)


def main():
    w, px = load()
    variants, sig_day, sig_am = build_variants(w, px)

    pd.DataFrame({"senal_dia": sig_day, "senal_manana": sig_am}).join(
        w[["season", "tmax_c", "tmax_anom", "prcp_mm", "snow_mm", "day_cloud",
            "am_cloud", "am_temp_c", "am_temp_anom", "am_rain_hours", "am_snow_hours"]]
    )[START:END].round(3).to_csv(os.path.join(OUT, "senales_diarias.csv"))

    perf, tables, perms, sub = [], {}, {}, []
    for name, vv in variants.items():
        df = vv["df"]
        for kind, lab in [("ls", "Long/Short clima"), ("long_nice", "Solo long dias lindos"),
                          ("short_ugly", "Solo short dias feos"),
                          ("long_not_ugly", "Long salvo dias feos"), ("bh", "Buy & hold")]:
            r = strat_returns(df, vv["intraday"], kind)
            perf.append(dict(Variante=name, Estrategia=lab, **stats(r)))
            if kind in ("ls", "bh"):
                for a, b, per in [(START, SPLIT, "1999-2008"), (SPLIT, END, "2009-2018")]:
                    sub.append(dict(Variante=name, Estrategia=lab, Periodo=per, **stats(r[a:b])))
        tables[name] = class_table(df)
        perms[name] = permutation_pvalue(df)
        print(name, "p-perm =", perms[name], flush=True)

    perf = pd.DataFrame(perf)
    sub = pd.DataFrame(sub)
    reg = cloud_regression(w, px)
    g = grid(w, px)
    g.round(3).to_csv(os.path.join(OUT, "grilla_parametros.csv"), index=False)

    plot_equity(variants, os.path.join(OUT, "equity_curves.png"))
    plot_classes(tables, os.path.join(OUT, "retorno_por_clase.png"))
    plot_grid(g, os.path.join(OUT, "robustez_grilla.png"))

    dist = pd.DataFrame({k: v["df"].sig.value_counts(normalize=True) for k, v in variants.items()}).T
    dist = dist.rename(columns={1: "Lindo", 0: "Neutral", -1: "Feo"})[["Lindo", "Neutral", "Feo"]]
    dist.insert(0, "Variante", dist.index)

    gs = g.groupby("variante").agg(Sharpe_IS_mediana=("Sharpe_IS", "median"),
                                   Sharpe_OOS_mediana=("Sharpe_OOS", "median"),
                                   pct_OOS_positivo=("Sharpe_OOS", lambda s: (s > 0).mean()),
                                   corr_IS_OOS=("Sharpe_IS", lambda s: np.nan)).reset_index()
    gs["corr_IS_OOS"] = [g[g.variante == v][["Sharpe_IS", "Sharpe_OOS"]].corr().iloc[0, 1] for v in gs.variante]

    pfmt = perf.copy()
    for c in ["CAGR", "Vol", "MaxDD", "HitRate", "Exposicion"]:
        pfmt[c] = (pfmt[c] * 100).round(1).astype(str) + "%"
    sfmt = sub.copy()
    for c in ["CAGR", "Vol", "MaxDD", "HitRate", "Exposicion"]:
        sfmt[c] = (sfmt[c] * 100).round(1).astype(str) + "%"

    with open(os.path.join(OUT, "report.md"), "w") as f:
        f.write("# Resultados del backtest clima NY -> S&P 500\n\n")
        f.write(f"Costo: {COST_BP:.0f} pb por lado. Permutaciones: {N_PERM}.\n\n")
        f.write("## Distribucion de dias\n\n" + fmt(dist) + "\n\n")
        f.write("## Performance (periodo completo)\n\n" + fmt(pfmt) + "\n\n")
        f.write("## Sub-periodos (L/S vs buy & hold)\n\n" + fmt(sfmt) + "\n\n")
        f.write("## p-valor por permutacion (L/S, etiquetas barajadas dentro de cada estacion)\n\n")
        for k, p in perms.items():
            f.write(f"- {k}: p = {p:.3f}\n")
        f.write("\n## Regresion retorno ~ nubosidad desestacionalizada (close->close, dia completo)\n\n")
        f.write(f"beta = {reg['beta_bp_por_100pct_nubes']:.2f} pb por pasar de despejado a cubierto, "
                f"t = {reg['t']:.2f}, n = {reg['n']}\n\n")
        for k, t in tables.items():
            f.write(f"## Retorno medio por clase — {k}\n\n" + fmt(t) + "\n\n")
        f.write("## Robustez: grilla de " + str(len(g) // 3) + " combinaciones de umbrales por variante\n\n")
        f.write(fmt(gs) + "\n")
    print(open(os.path.join(OUT, "report.md")).read())


if __name__ == "__main__":
    main()
