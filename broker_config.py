"""
asset_group_config.py
=====================

Shared module defining each asset universe's parameters (start_date, fees,
freq_anual) so the whole pipeline uses a single source of truth.

WHY THIS EXISTS:
  Previously each script had its own hardcoded START_DATE and FEES. Result:
  downstream stages recomputed metrics with parameters different from the
  ones the miner used to approve the strategy, so the enriched IS/OOS
  columns described a different reality than the original CSV.

  Centralizing here guarantees that when any downstream script sees a
  strategy from asset_group X, it processes it with exactly the same
  parameters it was originally mined with.

HOW TO USE:
  from asset_group_config import get_config_for_group, infer_group_from_ticker

  cfg = get_config_for_group('STOCKS')
  # cfg.start_date  → '2015-01-01'
  # cfg.fees        → 0.0005
  # cfg.freq_anual  → 252
"""

from dataclasses import dataclass


# ============================================================================
# CONFIGURATION PER ASSET GROUP
# ============================================================================
#
# IMPORTANT: these values MUST match the ones the miner uses. If you change
# a value in the miner, update it here too.
#
# Current values:
#   - CRYPTO:     start 2019, fees 0.001,  freq 365 (crypto trades 24/7)
#   - STOCKS:     start 2015, fees 0.0005, freq 252 (US market business days)
#   - STOCKS_ETF: start 2010, fees 0.0005, freq 252
#   - FOREX:      start 2010, fees 0.0002, freq 252 (low forex spreads)
#   - COMMODITIES:start 2010, fees 0.0005, freq 252
#   - DEFENSIVO:  start 2010, fees 0.0005, freq 252
# ============================================================================

@dataclass(frozen=True)
class AssetGroupConfig:
    """Simulation parameters for one asset universe."""
    start_date: str        # data download start date
    fees: float            # fee per trade (decimal, e.g. 0.001 = 0.1%)
    freq_anual: int        # days per year to annualize Sharpe/vol (365 or 252)
    dd_base: float         # maximum allowed drawdown (quality-gate reference)


ASSET_GROUP_CONFIG = {
    'CRYPTO': AssetGroupConfig(
        start_date='2019-01-01',
        fees=0.001,
        freq_anual=365,
        dd_base=0.65,
    ),
    'STOCKS': AssetGroupConfig(
        start_date='2015-01-01',
        fees=0.0005,
        freq_anual=252,
        dd_base=0.65,
    ),
    'STOCKS_ETF': AssetGroupConfig(
        start_date='2010-01-01',
        fees=0.0005,
        freq_anual=252,
        dd_base=0.50,
    ),
    'FOREX': AssetGroupConfig(
        start_date='2010-01-01',
        fees=0.0002,
        freq_anual=252,
        dd_base=0.40,
    ),
    'COMMODITIES': AssetGroupConfig(
        start_date='2010-01-01',
        fees=0.0005,
        freq_anual=252,
        dd_base=0.50,
    ),
    'DEFENSIVO': AssetGroupConfig(
        start_date='2010-01-01',
        fees=0.0005,
        freq_anual=252,
        dd_base=0.35,  # defensives require low DD by design
    ),
}


def get_config_for_group(asset_group: str) -> AssetGroupConfig:
    """
    Returns the config for an asset_group.

    Raises:
        ValueError if the group is not registered (better to fail loudly
        than to use silent defaults that can contaminate metrics).
    """
    if asset_group not in ASSET_GROUP_CONFIG:
        raise ValueError(
            f"Unknown asset_group: '{asset_group}'. "
            f"Available: {list(ASSET_GROUP_CONFIG.keys())}. "
            f"If it is a new group, add it to ASSET_GROUP_CONFIG in "
            f"asset_group_config.py."
        )
    return ASSET_GROUP_CONFIG[asset_group]


# ============================================================================
# GROUP INFERENCE FROM TICKER (useful fallback)
# ============================================================================
#
# Ideally the asset_group comes in the CSV. But if it is missing for any
# reason (old CSVs, errors), this function infers the group from the ticker
# format. It is a fallback — always prefer reading the group from the CSV.
# ============================================================================

def infer_group_from_ticker(ticker: str, carpeta_origen: str = '') -> str:
    """
    Infers the asset_group from the ticker and/or the source folder.

    carpeta_origen (source folder) is the first criterion (if it comes from
    'estrategias_defensivas' it is DEFENSIVO regardless of the ticker).
    Otherwise, ticker-format rules apply.

    Args:
        ticker: asset symbol (e.g. 'BTC-USD', 'NVDA', 'EURUSD=X')
        carpeta_origen: name of the folder containing the CSV (optional)

    Returns:
        One of: 'CRYPTO', 'FOREX', 'COMMODITIES', 'STOCKS_ETF', 'STOCKS', 'DEFENSIVO'
    """
    # Priority 1: the folder knows more than the ticker
    if 'defensiv' in carpeta_origen.lower():
        return 'DEFENSIVO'

    # Priority 2: ticker format
    if ticker.endswith('-USD'):
        return 'CRYPTO'
    if ticker.endswith('=X'):
        return 'FOREX'
    if ticker.endswith('=F'):
        return 'COMMODITIES'

    # Priority 3: stock vs ETF heuristic
    # US ETFs are usually 3-4 pure uppercase letters. Stocks can be 1-5
    # letters. Not perfect (e.g. 'V' is Visa stock but looks like an ETF).
    # In practice, if the CSV carries a correct asset_group this function
    # is almost never called.
    KNOWN_ETFS = {
        'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'XLK', 'XLF', 'XLE',
        'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XAR', 'XBI', 'SMH', 'SOXX',
        'TLT', 'BND', 'GLD', 'SLV', 'GDX', 'GDXJ', 'EEM', 'EWZ', 'MCHI',
        'EWJ', 'VGK', 'INDA', 'KWEB', 'DBMF', 'KMLM', 'BTAL', 'WTMF',
        'MOAT', 'QUAL', 'MTUM', 'USMV', 'VLUE', 'DGRW', 'DGRS', 'DON',
        'DLN', 'DES', 'ARKK', 'ARKG', 'ARKQ', 'ARKW', 'ARKF', 'BOTZ',
        'AIQ', 'CLOU', 'FINX', 'WCLD', 'HACK', 'BUG', 'CIBR', 'IBUY',
        'BUZZ', 'CHAT', 'QTUM', 'JETS', 'HERO', 'ESPO', 'LIT', 'URA',
        'URNM', 'NLR', 'SETM', 'BBH', 'PPH', 'HEAL', 'AGNG', 'GNOM',
        'TIP', 'SCHP', 'EMB', 'LQD', 'HYG', 'HYD', 'AGGY', 'ANGL',
        'SCHD', 'NOBL', 'VYM', 'VIG', 'DGRO', 'CAOS', 'TAIL', 'SWAN',
        'VNQ', 'XLRE', 'IYR', 'KRE', 'IAT', 'COPX', 'CPER', 'DBA',
        'MOO', 'PPLT', 'PALL', 'AIPO', 'SRVR', 'TRFK', 'VPN', 'CHPS',
        'UFO', 'PHO', 'FIW', 'CGW', 'PRNT', 'MSOS', 'MJ', 'WEED',
        # add more as the universe grows
    }
    if ticker in KNOWN_ETFS:
        return 'STOCKS_ETF'

    return 'STOCKS'


# ============================================================================
# UTILITY: load config from a strategy CSV row
# ============================================================================

def get_config_from_strategy_row(row, carpeta_origen: str = '') -> AssetGroupConfig:
    """
    Given a strategy row (pandas Series), returns the applicable config.

    Reads the 'asset_group' column if present; otherwise infers from ticker.
    """
    asset_group = row.get('asset_group')

    # Case 1: the CSV carries a valid asset_group
    if asset_group and asset_group in ASSET_GROUP_CONFIG:
        return get_config_for_group(asset_group)

    # Case 2: the CSV carries a legacy group → infer from ticker
    ticker = row.get('ticker', '')
    inferred = infer_group_from_ticker(ticker, carpeta_origen)
    return get_config_for_group(inferred)


# ============================================================================
# SELF-TEST — run this file directly to verify consistency
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("ASSET GROUP CONFIG — Consistency check")
    print("=" * 70)
    for group, cfg in ASSET_GROUP_CONFIG.items():
        print(f"\n  {group}")
        print(f"    start_date: {cfg.start_date}")
        print(f"    fees:       {cfg.fees}")
        print(f"    freq_anual: {cfg.freq_anual}")
        print(f"    dd_base:    {cfg.dd_base}")

    print("\n" + "=" * 70)
    print("Ticker inference test:")
    print("=" * 70)
    tests = [
        ('BTC-USD', '', 'CRYPTO'),
        ('NVDA', '', 'STOCKS'),
        ('SPY', '', 'STOCKS_ETF'),
        ('EURUSD=X', '', 'FOREX'),
        ('GC=F', '', 'COMMODITIES'),
        ('XLU', 'estrategias_defensivas', 'DEFENSIVO'),
        ('NVDA', 'estrategias_defensivas', 'DEFENSIVO'),  # folder wins
    ]
    for ticker, folder, expected in tests:
        result = infer_group_from_ticker(ticker, folder)
        ok = "✅" if result == expected else "❌"
        print(f"  {ok} {ticker:12s} (folder='{folder}') → {result} "
              f"(expected: {expected})")

    print("\n✅ Module loaded correctly.\n")
