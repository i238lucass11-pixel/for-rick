"""
forex_swap.py
=============

Forex swap model based on the HISTORICAL policy-rate differential (per
date), instead of the flat estimated 2% that broker_config.py used.

WHY:
  A forex pair's swap IS the rate differential between its two currencies
  (plus the broker markup). That differential changes with every central-
  bank decision, so a flat number for the whole backtest is a poor
  approximation — especially for exotic pairs (USDMXN, USDBRL) whose carry
  swung hard between 2015 and 2026.

  The sign matters too: if you are long the high-rate currency you RECEIVE
  swap (positive carry), you don't pay it. With long-only strategies:
    - long USDMXN = long USD / short MXN → you pay (MXN pays the high rate): negative swap
    - long USDCHF = long USD / short CHF → you receive (CHF pays the low rate): positive swap

WHAT THIS MODEL IS AND IS NOT:
  - IT IS: the policy-rate differential (Fed, ECB, BoJ, etc.) per date,
    which gives the correct SIGN and ORDER OF MAGNITUDE of the swap.
  - IT IS NOT: the exact swap Vantage charges. The broker adds its markup
    (Pepperstone adds a fixed ~2.5%, for example) and rounds. For the exact
    number, read Swap Long/Short from MT4/MT5. This module is the agreed
    directional approximation until that data is available.

USAGE:
  from forex_swap import carry_series, BROKER_MARKUP
  # annual carry (%) of going long 'USDMXN=X' at each date of the index:
  c = carry_series('USDMXN=X', dates)           # pure differential
  swap = -(c) + BROKER_MARKUP                    # effective long cost (see below)

SIGN CONVENTION:
  carry_series returns (base_rate - quote_rate). Positive = going long PAYS
  less / receives. The swap COST of a long position is -carry + markup:
  if carry is +4% (you receive), the cost is -4% (credited); if carry is
  -8% (you pay), the cost is +8% + markup.
"""

import pandas as pd


# =============================================================================
# MONETARY POLICY RATES (effective end of month, % annual)
# =============================================================================
# Breakpoints per currency: each rate applies until the next point (step
# function). Source: documented values from each central bank (Fed funds,
# ECB deposit facility, BoJ, BoE bank rate, RBA, RBNZ, BoC, SNB, BCB Selic,
# Banxico). Directional approximation, not day-exact.
# TODO: for precision, replace with the BIS series
# (bis.org/statistics/cbpol) which has all of these in a single dataset.
POLICY_RATES = {
    'USD': [('2015-01-01', 0.25), ('2015-12-01', 0.50), ('2016-12-01', 0.75),
            ('2017-03-01', 1.00), ('2017-06-01', 1.25), ('2017-12-01', 1.50),
            ('2018-03-01', 1.75), ('2018-06-01', 2.00), ('2018-09-01', 2.25),
            ('2018-12-01', 2.50), ('2019-08-01', 2.25), ('2019-09-01', 2.00),
            ('2019-10-01', 1.75), ('2020-03-01', 0.25), ('2022-03-01', 0.50),
            ('2022-05-01', 1.00), ('2022-06-01', 1.75), ('2022-07-01', 2.50),
            ('2022-09-01', 3.25), ('2022-11-01', 4.00), ('2022-12-01', 4.50),
            ('2023-02-01', 4.75), ('2023-03-01', 5.00), ('2023-05-01', 5.25),
            ('2023-07-01', 5.50), ('2024-09-01', 5.00), ('2024-11-01', 4.75),
            ('2024-12-01', 4.50)],
    'EUR': [('2015-01-01', -0.20), ('2015-12-01', -0.30), ('2016-03-01', -0.40),
            ('2019-09-01', -0.50), ('2022-07-01', 0.00), ('2022-09-01', 0.75),
            ('2022-11-01', 1.50), ('2022-12-01', 2.00), ('2023-02-01', 2.50),
            ('2023-03-01', 3.00), ('2023-05-01', 3.25), ('2023-06-01', 3.50),
            ('2023-08-01', 3.75), ('2023-09-01', 4.00), ('2024-06-01', 3.75),
            ('2024-09-01', 3.50), ('2024-10-01', 3.25), ('2024-12-01', 3.00)],
    'JPY': [('2015-01-01', 0.10), ('2016-02-01', -0.10), ('2024-03-01', 0.10),
            ('2024-08-01', 0.25), ('2025-01-01', 0.50)],
    'GBP': [('2015-01-01', 0.50), ('2016-08-01', 0.25), ('2017-11-01', 0.50),
            ('2018-08-01', 0.75), ('2020-03-01', 0.10), ('2021-12-01', 0.25),
            ('2022-02-01', 0.50), ('2022-03-01', 0.75), ('2022-05-01', 1.00),
            ('2022-06-01', 1.25), ('2022-08-01', 1.75), ('2022-09-01', 2.25),
            ('2022-11-01', 3.00), ('2022-12-01', 3.50), ('2023-02-01', 4.00),
            ('2023-03-01', 4.25), ('2023-05-01', 4.50), ('2023-06-01', 5.00),
            ('2023-08-01', 5.25), ('2024-08-01', 5.00), ('2024-11-01', 4.75)],
    'AUD': [('2015-01-01', 2.50), ('2015-02-01', 2.25), ('2015-05-01', 2.00),
            ('2016-05-01', 1.75), ('2016-08-01', 1.50), ('2019-06-01', 1.25),
            ('2019-07-01', 1.00), ('2019-10-01', 0.75), ('2020-03-01', 0.25),
            ('2020-11-01', 0.10), ('2022-05-01', 0.35), ('2022-06-01', 0.85),
            ('2022-07-01', 1.35), ('2022-08-01', 1.85), ('2022-09-01', 2.35),
            ('2022-10-01', 2.60), ('2022-11-01', 2.85), ('2022-12-01', 3.10),
            ('2023-02-01', 3.35), ('2023-03-01', 3.60), ('2023-05-01', 3.85),
            ('2023-06-01', 4.10), ('2023-11-01', 4.35), ('2025-02-01', 4.10)],
    'NZD': [('2015-01-01', 3.50), ('2015-06-01', 3.25), ('2015-07-01', 3.00),
            ('2015-09-01', 2.75), ('2015-12-01', 2.50), ('2016-03-01', 2.25),
            ('2016-08-01', 2.00), ('2016-11-01', 1.75), ('2019-05-01', 1.50),
            ('2019-08-01', 1.00), ('2020-03-01', 0.25), ('2021-10-01', 0.50),
            ('2021-11-01', 0.75), ('2022-02-01', 1.00), ('2022-04-01', 1.50),
            ('2022-05-01', 2.00), ('2022-07-01', 2.50), ('2022-08-01', 3.00),
            ('2022-10-01', 3.50), ('2022-11-01', 4.25), ('2023-02-01', 4.75),
            ('2023-04-01', 5.25), ('2023-05-01', 5.50), ('2024-08-01', 5.25),
            ('2024-10-01', 4.75), ('2024-11-01', 4.25), ('2025-02-01', 3.75)],
    'CAD': [('2015-01-01', 1.00), ('2015-01-15', 0.75), ('2015-07-01', 0.50),
            ('2017-07-01', 0.75), ('2017-09-01', 1.00), ('2018-01-01', 1.25),
            ('2018-07-01', 1.50), ('2018-10-01', 1.75), ('2020-03-01', 0.25),
            ('2022-03-01', 0.50), ('2022-04-01', 1.00), ('2022-06-01', 1.50),
            ('2022-07-01', 2.50), ('2022-09-01', 3.25), ('2022-10-01', 3.75),
            ('2022-12-01', 4.25), ('2023-01-01', 4.50), ('2023-06-01', 4.75),
            ('2023-07-01', 5.00), ('2024-06-01', 4.75), ('2024-07-01', 4.50),
            ('2024-09-01', 4.25), ('2024-10-01', 3.75), ('2024-12-01', 3.25)],
    'CHF': [('2015-01-01', -0.75), ('2022-06-01', -0.25), ('2022-09-01', 0.50),
            ('2022-12-01', 1.00), ('2023-03-01', 1.50), ('2023-06-01', 1.75),
            ('2024-03-01', 1.50), ('2024-06-01', 1.25), ('2024-09-01', 1.00),
            ('2024-12-01', 0.50)],
    'BRL': [('2015-01-01', 12.25), ('2015-07-01', 14.25), ('2016-10-01', 14.00),
            ('2017-01-01', 13.00), ('2017-04-01', 11.25), ('2017-07-01', 9.25),
            ('2017-12-01', 7.00), ('2018-03-01', 6.50), ('2019-08-01', 6.00),
            ('2019-10-01', 5.00), ('2019-12-01', 4.50), ('2020-03-01', 3.75),
            ('2020-08-01', 2.00), ('2021-03-01', 2.75), ('2021-05-01', 3.50),
            ('2021-06-01', 4.25), ('2021-08-01', 5.25), ('2021-09-01', 6.25),
            ('2021-10-01', 7.75), ('2021-12-01', 9.25), ('2022-02-01', 10.75),
            ('2022-03-01', 11.75), ('2022-05-01', 12.75), ('2022-06-01', 13.25),
            ('2022-08-01', 13.75), ('2023-08-01', 13.25), ('2023-09-01', 12.75),
            ('2023-11-01', 12.25), ('2023-12-01', 11.75), ('2024-01-01', 11.25),
            ('2024-03-01', 10.75), ('2024-05-01', 10.50), ('2024-09-01', 10.75),
            ('2024-11-01', 11.25), ('2024-12-01', 12.25), ('2025-01-01', 13.25)],
    'MXN': [('2015-01-01', 3.00), ('2015-12-01', 3.25), ('2016-02-01', 3.75),
            ('2016-06-01', 4.25), ('2016-09-01', 4.75), ('2016-11-01', 5.25),
            ('2016-12-01', 5.75), ('2017-02-01', 6.25), ('2017-03-01', 6.50),
            ('2017-05-01', 6.75), ('2017-06-01', 7.00), ('2018-06-01', 7.75),
            ('2018-11-01', 8.00), ('2018-12-01', 8.25), ('2019-08-01', 8.00),
            ('2019-09-01', 7.75), ('2019-11-01', 7.50), ('2019-12-01', 7.25),
            ('2020-02-01', 7.00), ('2020-03-01', 6.50), ('2020-04-01', 6.00),
            ('2020-05-01', 5.50), ('2020-06-01', 5.00), ('2020-08-01', 4.50),
            ('2021-06-01', 4.25), ('2021-08-01', 4.50), ('2021-09-01', 4.75),
            ('2021-11-01', 5.00), ('2021-12-01', 5.50), ('2022-02-01', 6.00),
            ('2022-03-01', 6.50), ('2022-05-01', 7.00), ('2022-06-01', 7.75),
            ('2022-08-01', 8.50), ('2022-09-01', 9.25), ('2022-11-01', 10.00),
            ('2022-12-01', 10.50), ('2023-02-01', 11.00), ('2023-03-01', 11.25),
            ('2024-03-01', 11.00), ('2024-08-01', 10.75), ('2024-09-01', 10.50),
            ('2024-11-01', 10.25), ('2024-12-01', 10.00), ('2025-01-01', 10.00)],
}

# Broker markup over the theoretical differential. Vantage does not publish
# it; we use Pepperstone's (2.5% annual, official) as a proxy — this is what
# the broker adds on top of the rate differential, and it is ALWAYS paid
# (long or short). Adjust once real Vantage MT4/MT5 data is available.
BROKER_MARKUP = 0.025   # 2.5% annual


def _rate_series(currency: str, dates: pd.DatetimeIndex) -> pd.Series:
    """Daily policy-rate series (step function, forward-fill)."""
    pts = POLICY_RATES[currency]
    s = pd.Series({pd.Timestamp(f): r / 100.0 for f, r in pts})
    full = dates.union(s.index)
    return s.reindex(full).sort_index().ffill().reindex(dates)


def carry_series(pair: str, dates: pd.DatetimeIndex) -> pd.Series:
    """
    Annual carry (fraction, not %) of going LONG the pair, per date.
    pair: 'USDMXN=X' or 'USDMXN'. Returns base_rate - quote_rate.
    Positive = long receives; negative = long pays.
    """
    p = pair.replace('=X', '')
    base, quote = p[:3], p[3:]
    if base not in POLICY_RATES or quote not in POLICY_RATES:
        return pd.Series(0.0, index=dates)
    return _rate_series(base, dates) - _rate_series(quote, dates)


def long_swap_cost_series(pair: str, dates: pd.DatetimeIndex,
                          markup: float = BROKER_MARKUP) -> pd.Series:
    """
    Annual swap COST (fraction) of HOLDING a LONG position in the pair, per
    date. Cost convention: positive = costs (drag), negative = credits
    (favorable carry).

      cost = -carry + markup
      (if carry=+0.04 you receive → cost=-0.04+0.025=-0.015 net in your favor;
       if carry=-0.08 you pay → cost=+0.08+0.025=+0.105 against you)
    """
    return -carry_series(pair, dates) + markup


def is_forex(ticker: str) -> bool:
    return ticker.endswith('=X')
