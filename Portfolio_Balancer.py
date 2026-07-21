{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Quant Trend-Following Pipeline — Review Notebook\n",
    "\n",
    "This notebook orchestrates the full pipeline so it is easy to follow step by\n",
    "step. Each stage remains an independent `.py` script (easier to maintain and\n",
    "version); this notebook just runs them in order and shows intermediate results.\n",
    "\n",
    "**Professor-requested changes (all applied):**\n",
    "\n",
    "| Request | Change applied |\n",
    "|---|---|\n",
    "| 60/40 split | `SPLIT = 0.60` in `Quant_Miner.py` (was 0.70) |\n",
    "| Leave canaries out for now | `CANARIOS_ON = False` — code intact, not used in the score |\n",
    "| No beta rotations, keep it simple with momentum | Regime tilt removed from `Portfolio_Balancer.py` |\n",
    "| Select top-3 by IS, not OOS (\"that is fitting\") | Selection filters and `selection_score` are 100% IS. OOS is kept for reporting and `validation.py` only |\n",
    "| Table of top results by IS | Section 2 of this notebook generates it |\n",
    "| Minimum 2 trades/year | `MIN_TRADES_PER_YEAR = 2.0` filter (IS and OOS) |\n",
    "\n",
    "**Pipeline (4 stages):**\n",
    "```\n",
    "Quant_Miner.py            (mining + full metrics + IS-only top-3 selection,\n",
    "                           per ASSET_GROUP; subcommand `reseleccionar`\n",
    "                           re-selects from the raw cache without re-mining)\n",
    "  -> Strategy_Assembler.py   (per-ticker ensembles + IS-based pruning)\n",
    "  -> Portfolio_Balancer.py   (official config: inverse-vol across sleeves with\n",
    "                              caps + VIX de-risk gate + real costs; no\n",
    "                              DEFENSIVE_QUALITY, uniform 126d rebalance,\n",
    "                              no leverage)\n",
    "  -> validation.py           (IS/OOS, walk-forward, Monte Carlo)\n",
    "```\n",
    "\n",
    "**Headline results (official portfolio, with costs, 2016-2026 OOS):**\n",
    "CAGR 7.03% | Sharpe 1.03 | MaxDD -12.09% | Calmar 0.58.\n",
    "Walk-forward mean Sharpe 1.30 (11/12 windows positive); Monte Carlo places the\n",
    "realized result at the 51st percentile; plan for the p5 MaxDD of -15.2%.\n",
    "\n",
    "Note: `Quant_Miner.py` is documented in Spanish (mining-stage working\n",
    "language); everything else — code, reports, outputs — is in English.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "## Setup"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": "import subprocess, sys, os\nimport pandas as pd\n\n# --- Anchor the working directory to this notebook's folder -----------------\n# Jupyter/VS Code often start the kernel in a DIFFERENT working directory\n# (e.g. the workspace root or the user's home), which breaks every relative\n# path below. This block finds the folder that actually contains THIS\n# package's files. The sentinel set is strict on purpose: other folders may\n# contain OLD copies of the pipeline, and anchoring there would run against\n# stale data.\n_SENTINELS = ('top3_global.csv', 'Portfolio_Balancer.py', 'precios_cache.py')\n\ndef _is_package_dir(d):\n    try:\n        return d and all(os.path.exists(os.path.join(d, s)) for s in _SENTINELS)\n    except Exception:\n        return False\n\ndef _candidates():\n    # 1. Editor-provided notebook path (VS Code / JupyterLab >= 3.4)\n    nb = globals().get('__vsc_ipynb_file__')\n    if nb:\n        yield os.path.dirname(nb)\n    nb = os.environ.get('JPY_SESSION_NAME')\n    if nb:\n        yield os.path.dirname(os.path.abspath(nb))\n    # 2. Current working directory and its parents (kernel may start above)\n    d = os.getcwd()\n    for _ in range(4):\n        yield d\n        parent = os.path.dirname(d)\n        if parent == d:\n            break\n        d = parent\n    # 3. Subfolders up to 3 levels below the cwd\n    base_depth = os.getcwd().count(os.sep)\n    for root, dirs, files in os.walk(os.getcwd()):\n        dirs[:] = [x for x in dirs if not x.startswith(('.', '_', '$'))]\n        yield root\n        if root.count(os.sep) - base_depth >= 3:\n            dirs.clear()\n    # 4. Typical locations relative to the user's home\n    home = os.path.expanduser('~')\n    for sub in ('Downloads', 'Desktop', 'Documents', 'Escritorio', 'Documentos'):\n        base = os.path.join(home, sub)\n        if not os.path.isdir(base):\n            continue\n        yield base\n        try:\n            for name in os.listdir(base):\n                d1 = os.path.join(base, name)\n                if os.path.isdir(d1):\n                    yield d1\n                    try:\n                        for name2 in os.listdir(d1):\n                            d2 = os.path.join(d1, name2)\n                            if os.path.isdir(d2):\n                                yield d2\n                    except OSError:\n                        pass\n        except OSError:\n            pass\n\n_pkg = next((d for d in _candidates() if _is_package_dir(d)), None)\nif _pkg:\n    os.chdir(_pkg)\n    print(f\"Working directory: {os.getcwd()}\")\nelse:\n    raise FileNotFoundError(\n        \"Could not locate the package folder (looked for \"\n        + \", \".join(_SENTINELS) + \").\\n\"\n        \"DIAGNOSTICS - copy this to whoever is debugging:\\n\"\n        f\"  cwd                = {os.getcwd()}\\n\"\n        f\"  __vsc_ipynb_file__ = {globals().get('__vsc_ipynb_file__')}\\n\"\n        f\"  JPY_SESSION_NAME   = {os.environ.get('JPY_SESSION_NAME')}\\n\"\n        \"Manual fix: run in a new cell ->  import os; \"\n        \"os.chdir(r'<folder containing top3_global.csv>')\")\n\nPY = sys.executable  # equivalent to PowerShell's 'py' in this environment\n# NOTE: ALWAYS run this cell first (defines ejecutar()). It is named\n# 'ejecutar' rather than 'run' on purpose: if the kernel restarts and you\n# call run() without this cell, IPython confuses it with its %run magic and\n# throws a cryptic error like: File '(\"Strategy_Assembler.py\")' not found.\n\ndef ejecutar(script, *args):\n    \"\"\"Runs a pipeline script and streams its output LIVE, line by line,\n    as it is produced (does not buffer everything until the end).\"\"\"\n    print(\"\\n\" + \"=\" * 70)\n    print(\">\", script, *args)\n    print(\"=\" * 70)\n    proc = subprocess.Popen([PY, script, *args], stdout=subprocess.PIPE,\n                             stderr=subprocess.STDOUT, text=True,\n                             encoding='utf-8', errors='replace', bufsize=1,\n                             cwd=os.getcwd())\n    for line in proc.stdout:\n        print(line, end='')\n    proc.wait()\n    if proc.returncode != 0:\n        print(\"WARNING:\", script, \"exited with code\", proc.returncode)\n    return proc.returncode\n"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "## 1. Mining + selection (Quant_Miner)\n\nOne pass per asset group: Optuna mines, the full IS/OOS metrics are computed\nwith the portfolio object in hand, and the top-3 selection is **100%\nin-sample** (professor's request): sleeve-adapted IS-Sharpe thresholds,\n`is_dd <= 0.55`, `is_profit_factor >= 1.20`, `sensitivity_drop <= 20`,\n`is/oos_trades_per_year >= 2`, IS-only `selection_score`, dedup by indicator\nfamily + IS correlation. The OOS is stored in the CSV for reporting and for\n`validation.py` only.\n\nTo re-run the selection over the existing mined cache WITHOUT re-mining\n(fast, uses the local price cache):\n\n    py Quant_Miner.py reseleccionar          # full re-selection\n    py Quant_Miner.py reseleccionar --dry    # filters only, writes nothing\n"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": "# One run per group — uncomment as needed (slow):\n# ejecutar(\"Quant_Miner.py\", \"CRYPTO\")   # CRYPTO | STOCKS | STOCKS_ETF |\n#                                        # COMMODITIES | FOREX | DEFENSIVO\n# ejecutar(\"Quant_Miner.py\", \"reseleccionar\")   # re-select without re-mining\nprint(\"Uncomment to mine (slow; needs internet for yfinance).\")\n"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "### Table for the professor — Top results by IS (from top3_global.csv)"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": "df = pd.read_csv(\"top3_global.csv\")\n\ncols_is = ['ticker', 'sleeve', 'indicator', 'side',\n           'is_sharpe', 'is_cagr', 'is_profit_factor', 'is_dd',\n           'is_trades_per_year', 'is_monthly_consistency', 'is_tail_ratio',\n           'stability_score', 'sensitivity_drop', 'selection_score']\ncols_is = [c for c in cols_is if c in df.columns]\n\ntop_table = (df.sort_values('selection_score', ascending=False)\n                .head(30)[cols_is])\ntop_table\n"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": "# Plain-text version:\nprint(top_table.to_markdown(index=False))\n"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "## 2. Strategy Assembler\n\nPer-ticker rolling inverse-vol ensembles over the selected top-3,\ncross-indicator dedup, and dead-weight pruning by **IS** Sharpe (pruning by\nOOS would be covert selection)."
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": "ejecutar(\"Strategy_Assembler.py\")"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "## 3. Portfolio Balancer — official config\n\nInverse-vol across sleeves (Ledoit-Wolf shrunk covariance) with hard caps\n(GROWTH_BETA 80%, DIVERSIFIERS/CYCLICAL 30%, CRYPTO 25%), diversification\nscore within sleeves, VIX de-risk gate, broker-differentiated costs and\nhistorical forex swaps. No leverage (swept 1.15-1.50x: monotonically worse),\nno DEFENSIVE_QUALITY (48-config sweep), uniform 126d rebalance (phase test,\n9 offsets). Every finding is documented in the file's comments."
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": "ejecutar(\"Portfolio_Balancer.py\")"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": "# Equity + drawdown chart\neq = None\nfor ext in ('parquet', 'csv'):\n    p = f\"portfolio_equity.{ext}\"\n    if os.path.exists(p):\n        eq = pd.read_parquet(p) if ext == 'parquet' else pd.read_csv(p)\n        break\nif eq is not None:\n    eq['fecha'] = pd.to_datetime(eq['fecha'])\n    ax = eq.plot(x='fecha', y=['equity', 'baseline_equity'], figsize=(10, 4),\n                 title='Portfolio equity (official vs no-cost reference)')\n    eq.plot(x='fecha', y='drawdown', figsize=(10, 2), color='firebrick',\n            title='Drawdown (%)')\nelse:\n    print(\"portfolio_equity not found — run the balancer first.\")\n"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": "## 4. Validation (IS/OOS, walk-forward, Monte Carlo)"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": "ejecutar(\"validation.py\")"
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}