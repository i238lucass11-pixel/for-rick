"""
Quant_Miner.py
==============

Minero + Enriquecedor + Seleccion en UN solo archivo (fusion de Optuna_5.py y
Enriquecedor_v3.py, sesion 2026-07). Por que existe:

  El Enriquecedor nacio como parche: el minero original no calculaba todas
  las metricas, asi que un segundo script RECONSTRUIA las senales desde los
  params para calcularlas. Esa reconstruccion duplicaba descarga de datos,
  requeria un cache fragil, y fue la fuente del bug de los indicadores MR
  (senales que el minero conocia pero el reconstructor no). Aca las metricas
  se calculan EN EL MOMENTO del minado, cuando el portfolio ya esta armado:
  cero reconstruccion, cero cache, un solo SPLIT.

FLUJO (por ticker):
  descarga -> Optuna por indicador -> aduana (umbrales por grupo) ->
  sensibilidad -> metricas completas IS/OOS -> seleccion top3 (100% IS) ->
  dedup por correlacion -> outputs globales.

OUTPUTS (contrato identico al del Enriquecedor — el resto del pipeline no
cambia): top3_global.csv, oos_returns_long, equity_curves_long, y el CSV
crudo por ticker (archivo historico, permite re-seleccionar sin re-minar).

SELECCION IS-ONLY (pedido del profesor): filtros y score usan solo metricas
in-sample; el OOS queda en el CSV para reporte y para validation.py.

GRUPOS: 'CRYPTO'|'STOCKS'|'STOCKS_ETF'|'COMMODITIES'|'FOREX'|'DEFENSIVO'|'ALL'
(DEFENSIVO absorbido de Optuna_Defensivo.py: mismo flujo, universo e
indicadores lentos propios, umbrales de aduana mas laxos en Sharpe y mas
estrictos en DD.)

Correr:  py Quant_Miner.py FOREX     (grupo por argumento — recomendado)
         py Quant_Miner.py           (usa el ASSET_GROUP editado abajo)

PARALELO: abrir varias terminales, una por grupo:
         py Quant_Miner.py FOREX
         py Quant_Miner.py CRYPTO
         py Quant_Miner.py DEFENSIVO   ... etc.
Los outputs globales están protegidos con lock file — las corridas pueden
terminar en cualquier orden sin pisarse. Sugerencia: arrancarlas con ~1 min
de diferencia para no saturar a Yahoo Finance con descargas simultáneas.
"""
import hashlib
import sys
import time
import yfinance as yf
import pandas as pd
import numpy as np
import vectorbt as vbt
import optuna
from optuna.trial import FixedTrial
import json
import talib
import warnings
import os
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================================
# ██████████████████████████████████████████████████████████████████████████████
# 1. CONFIGURACIÓN PRINCIPAL — EDITÁ AQUÍ
# ██████████████████████████████████████████████████████████████████████████████
# ==============================================================================

# --- Elegí el grupo de activos a minar ---
# Opciones: 'CRYPTO' | 'STOCKS' | 'STOCKS_ETF' | 'COMMODITIES' | 'FOREX' | 'ALL'
ASSET_GROUP = 'STOCKS_ETF'
# Override por línea de comandos: py Quant_Miner.py FOREX
# Solo nombres de grupo válidos — cualquier otro argv (el subcomando
# 'reseleccionar', flags de scripts que importan este módulo, etc.) se ignora
# acá. Fix deuda técnica 2026-07: antes CUALQUIER argv[1] pisaba ASSET_GROUP
# y rompía el import con ValueError en la selección de tickers.
_GRUPOS_CLI = {'CRYPTO', 'STOCKS', 'STOCKS_ETF', 'COMMODITIES', 'FOREX',
               'DEFENSIVO', 'ALL'}
if len(sys.argv) > 1 and sys.argv[1].upper() in _GRUPOS_CLI:
    ASSET_GROUP = sys.argv[1].upper()

# --- Fecha de inicio por grupo (más historia = más confianza) ---
START_DATES = {
    'CRYPTO':      '2019-01-01',
    'STOCKS':      '2015-01-01',
    'STOCKS_ETF':  '2000-01-01',   # extendido: ETFs growth con historia desde dot-com
    'COMMODITIES': '2010-01-01',
    'FOREX':       '2010-01-01',
}

# --- Parámetros generales ---
N_TRIALS   = 700     # producción (200 era config de prueba; 2500->1250->700
                     # 2026-07 para acelerar el re-minado en paralelo: con la
                     # bateria de sensibilidad por candidato, 1250 trials
                     # proyectaba ~8 dias para STOCKS entero)
SENSIBILIDAD_MAX_CANDIDATOS = 60   # top-K por Sharpe IS que entra a la bateria
                                   # de sensibilidad (ver comentario en el loop)
FEES       = 0.001      # 0.1% crypto / ajustar a 0.0002 para forex
OUTPUT_DIR = 'estrategias_por_ticker'

# ==============================================================================
# 2. LISTAS DE ACTIVOS
# ==============================================================================

TICKERS_CRYPTO = [
    # Reconstruido desde sleeve_map.json (2026-07) — universo completo
    'ADA-USD', 'APT-USD', 'ATOM-USD', 'AVAX-USD', 'BCH-USD', 'BNB-USD',
    'BTC-USD', 'DOGE-USD', 'DOT-USD', 'ETC-USD', 'ETH-USD', 'FIL-USD',
    'HBAR-USD', 'HYPE-USD', 'ICP-USD', 'LINK-USD', 'LTC-USD', 'MATIC-USD',
    'NEAR-USD', 'SHIB-USD', 'SOL-USD',  # SUI-USD removido 2026-07: 23 retornos corruptos ene-may 2024 (listing sucio en Yahoo) 'TON-USD', 'TRX-USD',
    'UNI-USD', 'VET-USD', 'XLM-USD', 'XMR-USD', 'XRP-USD',
    # Ampliación 2026-07
    'AAVE-USD', 'ALGO-USD', 'GRT-USD', 'SAND-USD', 'MANA-USD',
    'ARB-USD', 'OP-USD', 'INJ-USD', 'IMX-USD',
    # STX-USD removido 2026-07: datos Yahoo corruptos (+2309%/-99% jul-2023,
    # glitch de identidad del ticker). Detectado por diagnostico_retornos.py.
]

# Nota: SHIB tiene precio ~0.000012 — puede haber errores numéricos en TA-Lib.
# TON y MATIC pueden fallar en Yahoo Finance; el script los saltea automáticamente.
# XMR fue delistado en varios exchanges en 2023-24; datos pueden ser incompletos.
TICKERS_STOCKS = [
    # Reconstruido desde sleeve_map.json (2026-07); sin DEFENSIVE_QUALITY
    'AAPL', 'ABNB', 'ALAB', 'ALB', 'AMD', 'AMZN',
    'ANET', 'APLD', 'ARM', 'ASML', 'ASTS', 'AVAV',
    'AVGO', 'BABA', 'BAL', 'BE', 'BIDU', 'BIZD',
    'BKCH', 'BKNG', 'BLDP', 'BWX', 'CCL', 'CEG',
    'COHR', 'COIN', 'CORN', 'CORZ', 'COST', 'CRDO',
    'CRM', 'CRWD', 'CVX', 'CXSE', 'DAL', 'DAPP',
    'DHI', 'DLR', 'DRIV', 'DTCR', 'DXJ', 'EA',
    'EBIZ', 'EPI', 'EQIX', 'FANUY', 'FTNT', 'GE',
    'GEV', 'GFS', 'GOOG', 'GS', 'HEDJ', 'HLT',
    'HON', 'IBM', 'IDRV', 'INTC', 'IONQ', 'IPGP',
    'IRDM', 'IREN', 'IRM', 'ISRG', 'JD', 'JO',
    'JPM', 'KTOS', 'LEN', 'LITE', 'LMT', 'LUNR',
    'MA', 'MAGS', 'MAR', 'MDB', 'META', 'MILN',
    'MRVL', 'MSFT', 'MSTR', 'NBIS', 'NCLH', 'NET',
    'NFLX', 'NIB', 'NIO', 'NKE', 'NOC', 'NRG',
    'NVDA', 'NVR', 'OKTA', 'PANW', 'PATH', 'PDD',
    'PHM', 'PL', 'PLTR', 'PLUG', 'PNC', 'QBTS',
    'QCOM', 'QUBT', 'RACE', 'RBLX', 'RCL', 'RDW',
    'RGTI', 'RKLB', 'ROK', 'RTX', 'SBLK', 'SGG',
    'SHLD', 'SNOW', 'SOCL', 'SOYB', 'SQM', 'TFC',
    'TOL', 'TSLA', 'TSM', 'TTWO', 'UAL', 'UNG',
    'USB', 'USO', 'V', 'VNM', 'VRT', 'VST',
    'WCBR', 'WEAT', 'WMT', 'XOM', 'XSOE', 'ZIM',
    'ZS',
]

TICKERS_STOCKS_ETF = [
    # Reconstruido desde sleeve_map.json (2026-07); el archivo original era la version demo con solo SPY
    'AIQ', 'AIRR', 'ARKF', 'ARKG', 'ARKK', 'ARKQ',
    'ARKW', 'AVUV', 'BATT', 'BLOK', 'BOAT', 'BOTZ',
    'BUG', 'BUZZ', 'CHAT', 'CHPS', 'CIBR', 'CLOU',
    'CQQQ', 'EBAY', 'EEM', 'ESPO', 'EWJ', 'EWZ',
    'FINX', 'HACK', 'HD', 'HERO', 'IAT', 'IBUY',
    'IHAK', 'IJK', 'IJR', 'INDA', 'IRBO', 'IVW',
    'IWF', 'IWM', 'IWN', 'JETS', 'KARS', 'KRE',
    'KWEB', 'LIT', 'MCHI', 'NLR', 'ORCL', 'PRNT',
    'QQQ', 'QTUM', 'SBUX', 'SEA', 'SETM', 'SMH',
    'SNSR', 'SRVR', 'TRFK', 'UFO', 'URA', 'URNM',
    'VBK', 'VBR', 'VGK', 'VPN', 'WCLD', 'WEED',
    'XLE', 'XLK', 'XSMO',
    # Ampliación 2026-07: subsectores no explorados (sleeves vivos)
    'COPX', 'SIL', 'TAN', 'ICLN', 'PAVE', 'XME', 'SLX', 'MOO',
    'DBA', 'WOOD', 'ITB', 'XOP', 'IGV', 'FDN', 'SKYY', 'ROBO',
]

TICKERS_COMMODITIES = [
    # Reconstruido desde sleeve_map.json (2026-07)
    'CC=F', 'CL=F', 'CT=F', 'GC=F', 'HG=F', 'HO=F',
    'KC=F', 'KE=F', 'NG=F', 'OJ=F', 'PL=F', 'RB=F',
    'SB=F', 'SI=F', 'ZC=F', 'ZL=F', 'ZM=F', 'ZO=F',
    'ZR=F', 'ZS=F', 'ZW=F',
    # Ampliación 2026-07: ganadería (descorrelación real) + paladio
    'LE=F', 'HE=F', 'PA=F',
]

# Forex: los mejores para trend-following con spreads razonables
# Formato Yahoo Finance para pares de divisas
TICKERS_FOREX = [
    # Reconstruido (2026-07); sin exoticas de alto swap (USDBRL/MXN/TRY/ZAR, CADCHF, CHFJPY)
    'AUDJPY=X', 'AUDNZD=X', 'AUDUSD=X', 'CADJPY=X', 'EURGBP=X', 'EURJPY=X',
    'EURUSD=X', 'GBPJPY=X', 'GBPUSD=X', 'NZDJPY=X', 'NZDUSD=X', 'USDCAD=X',
    'USDCHF=X', 'USDNOK=X', 'USDSEK=X',
    # Ampliación 2026-07: crosses G10 líquidos, swap moderado (re-aplicada:
    # un rewrite posterior la había revertido — detectado en auditoría MT5)
    'EURCHF=X', 'GBPCHF=X', 'EURAUD=X', 'EURCAD=X', 'GBPAUD=X', 'GBPCAD=X',
    'AUDCAD=X', 'NZDCAD=X', 'EURNZD=X', 'GBPNZD=X',
]

# --- Selección automática según ASSET_GROUP ---
if ASSET_GROUP == 'CRYPTO':
    TICKERS   = TICKERS_CRYPTO
    START_DATE = START_DATES['CRYPTO']
    FEES       = 0.001
elif ASSET_GROUP == 'STOCKS_ETF':
    TICKERS   = TICKERS_STOCKS_ETF
    START_DATE = START_DATES['STOCKS_ETF']
    FEES       = 0.0005
elif ASSET_GROUP == 'COMMODITIES':
    TICKERS   = TICKERS_COMMODITIES
    START_DATE = START_DATES['COMMODITIES']
    FEES       = 0.0005
elif ASSET_GROUP == 'FOREX':
    TICKERS   = TICKERS_FOREX
    START_DATE = START_DATES['FOREX']
    FEES       = 0.0002   # Spread típico forex retail
elif ASSET_GROUP == 'STOCKS':
    TICKERS   = TICKERS_STOCKS
    START_DATE = START_DATES['STOCKS']
    FEES       = 0.0005
elif ASSET_GROUP == 'DEFENSIVO':
    TICKERS    = []          # se asigna en la sección 8b (TICKERS_DEFENSIVOS)
    START_DATE = '2010-01-01'
    FEES       = 0.0005
    VOL_REFERENCIA_DEFENSIVO = 0.15   # bonds/utilities: vol baja de referencia
elif ASSET_GROUP == 'ALL':
    # Modo ALL (2026-07): TODOS los grupos en una corrida, con parámetros
    # POR TICKER (start date, fees, perfil de aduana e indicadores según el
    # grupo real de cada uno — no un promedio incorrecto para todos).
    # El universo DEFENSIVO se suma en la sección 8b.
    TICKERS    = (TICKERS_CRYPTO + TICKERS_STOCKS + TICKERS_STOCKS_ETF +
                  TICKERS_COMMODITIES + TICKERS_FOREX)
    START_DATE = None   # per-ticker: ver GROUP_START / grupo_de_ticker()
    FEES       = None   # per-ticker: ver GROUP_FEES
else:
    raise ValueError(f"ASSET_GROUP '{ASSET_GROUP}' no reconocido. Usá: CRYPTO | STOCKS | STOCKS_ETF | COMMODITIES | FOREX | DEFENSIVO | ALL")

# --- Sharding opcional: particionar un grupo grande en N procesos paralelos ---
#   py Quant_Miner.py STOCKS shard=1/2    (proceso 1: tickers 1,3,5,...)
#   py Quant_Miner.py STOCKS shard=2/2    (proceso 2: tickers 2,4,6,...)
# Interleaved (no por bloques) para que cada shard reciba una mezcla pareja.
# Cada proceso escribe sus CSVs crudos por ticker y consolida via el lock —
# no hay solapamiento porque los tickers de cada shard son disjuntos.
for _a in sys.argv[1:]:
    if _a.startswith('shard='):
        try:
            _k, _n = (int(x) for x in _a.split('=', 1)[1].split('/'))
            if not (1 <= _k <= _n):
                raise ValueError
            TICKERS = TICKERS[_k - 1::_n]
            print(f"🔀 Shard {_k}/{_n}: {len(TICKERS)} tickers en este proceso")
        except ValueError:
            raise ValueError(f"shard invalido: '{_a}' (formato: shard=K/N, 1<=K<=N)")


# Parámetros por grupo (usados per-ticker en modo ALL; en modo single-group
# los globals START_DATE/FEES ya vienen resueltos arriba)
GROUP_FEES = {'CRYPTO': 0.001, 'STOCKS': 0.0005, 'STOCKS_ETF': 0.0005,
              'COMMODITIES': 0.0005, 'FOREX': 0.0002, 'DEFENSIVO': 0.0005}
GROUP_START = dict(START_DATES); GROUP_START['DEFENSIVO'] = '2010-01-01'
GRUPO_TICKER_ACTUAL = ASSET_GROUP   # se actualiza per-ticker en modo ALL


def grupo_de_ticker(t):
    """Grupo real de un ticker (para modo ALL). TICKERS_DEFENSIVOS se define
    en la sección 8b, pero esta función solo se invoca en runtime (main)."""
    if 'TICKERS_DEFENSIVOS' in globals() and t in TICKERS_DEFENSIVOS:
        return 'DEFENSIVO'
    if t.endswith('-USD'): return 'CRYPTO'
    if t.endswith('=X'):   return 'FOREX'
    if t.endswith('=F'):   return 'COMMODITIES'
    if t in TICKERS_STOCKS_ETF: return 'STOCKS_ETF'
    return 'STOCKS'

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# 1b. SPLIT + SELECCION (unificados — antes vivian en el Enriquecedor)
# ==============================================================================
SPLIT = 0.60            # IS/OOS 60/40 (pedido del profesor, sesion 2026-07)

# Seleccion top-N por ticker — 100% sobre metricas IS
TOP_N               = 3
CORR_MAX            = 0.85   # dedup: correlacion maxima entre finalistas
MIN_IS_SHARPE       = 0.60
MAX_IS_DD           = 0.55
MIN_IS_PF           = 1.20
MIN_STABILITY       = 0.50   # NO usada en seleccion desde el leakage fix 2026-07 (stability=oos/is es informativa)
MAX_SENS_DROP       = 20.0
MIN_TRADES_PER_YEAR = 2.0    # IS y OOS
DEBUG_FUNNEL        = True   # embudo: cuantos pasan cada filtro por separado

SKIP_EXISTING = True    # saltear tickers cuyo CSV crudo ya existe (resume)

OUT_TOP3    = 'top3_global.csv'
OUT_RETURNS = 'oos_returns_long'      # .parquet o .csv
OUT_EQUITY  = 'equity_curves_long'

# Perfiles de aduana por grupo (umbral de entrada al pool de candidatos).
# DEFENSIVO: menos trades y Sharpe (activos lentos), DD mucho mas estricto.
# LEAKAGE FIX (2026-07): la aduana ya NO filtra por performance OOS
# (min_oos_sharpe / min_oos_totret / min_stability eliminados). Cualquier
# filtro OOS previo a la seleccion es seleccion-por-OOS encubierta. Se
# CONSERVA min_oos_trades porque es el requisito de ACTIVIDAD del profesor
# (min 2 trades/anio en ambos periodos), no un filtro de performance.
# El dd_base aplica solo sobre el DD IS por la misma razon.
# FIX #5 (2026-07): el umbral de Sharpe antes era uniforme (0.40 para todos).
# Ahora se ADAPTA por grupo: activos de vol baja generan Sharpes altos con
# menor ruido, activos de vol alta con muchos trades tienen distribucion
# ruidosa; usar la misma vara sesga la aceptacion (favorece los primeros y
# rechaza los segundos por razones estructurales, no de calidad).
# Los valores por grupo son plantados a dedo pero razonables:
#   - FOREX / DEFENSIVO (bonos, ETFs anchos): vol baja, Sharpe historico
#     tipico > 1 para trend real -> pedir 0.50.
#   - STOCKS_ETF: mezcla, 0.40.
#   - STOCKS / COMMODITIES / CRYPTO: vol alta, Sharpe historico > 0 con ruido
#     grande -> pedir 0.30 (calidad de la estrategia, no del activo).
MINING_PROFILES = {
    'DEFAULT':   dict(min_is_trades=15, min_oos_trades=5, dd_base=0.65,
                      min_is_pf=1.10, min_is_sharpe=0.40,
                      dd_clip=(0.30, 0.85)),
    'DEFENSIVO': dict(min_is_trades=8,  min_oos_trades=3, dd_base=0.35,
                      min_is_pf=1.05, min_is_sharpe=0.50,
                      dd_clip=(0.15, 0.45)),
}

# FIX #6 (2026-07): agrupacion de indicadores por familia. Usada por
# seleccionar_top_ticker para forzar diversidad estructural en el ensemble
# intra-ticker (no basta con corr < 0.85; MACD y EMA-cross son primos).
FAMILIA_INDICADOR = {
    # cruce/tendencia de medias
    'MACD':          'MEDIAS_MOVILES',
    'EMA_LARGO':     'MEDIAS_MOVILES',
    '3EMA':          'MEDIAS_MOVILES',
    'DEMA':          'MEDIAS_MOVILES',
    'TEMA':          'MEDIAS_MOVILES',
    'KAMA':          'MEDIAS_MOVILES',
    'KAMA_LARGO':    'MEDIAS_MOVILES',
    'GOLDEN_CROSS':  'MEDIAS_MOVILES',
    # breakout / channels
    'DONCHIAN':      'BREAKOUT',
    'DONCHIAN_LARGO':'BREAKOUT',
    'BBANDS':        'BREAKOUT',
    'CHANDELIER':    'BREAKOUT',
    'SUPERTREND':    'BREAKOUT',
    # osciladores de momentum
    'RSI':           'OSCILADOR',
    'STOCH':         'OSCILADOR',
    'CCI':           'OSCILADOR',
    'WILLR':         'OSCILADOR',
    # medidas de fuerza direccional
    'ADX':           'FUERZA_DIRECCION',
    'ADX_LARGO':     'FUERZA_DIRECCION',
    'AROON':         'FUERZA_DIRECCION',
    'SAR':           'FUERZA_DIRECCION',
}

MIN_IS_SHARPE_BY_GROUP = {
    'FOREX':       0.50,
    'DEFENSIVO':   0.50,
    'STOCKS_ETF':  0.40,
    'STOCKS':      0.30,
    'COMMODITIES': 0.30,
    'CRYPTO':      0.30,
}

# Mapas para las columnas sleeve/sector del top3 (mismos JSON del pipeline)
def _load_maps():
    sleeve, sector = {}, {}
    try:
        sm = json.load(open('sleeve_map.json', encoding='utf-8'))
        sleeve = dict(sm.get('ticker_sleeve', {}))
        sleeve.update(sm.get('ticker_override', {}))
        sec_map = json.load(open('sector_map.json', encoding='utf-8'))
        sector = dict(sec_map.get('ticker_sector', {}))
        s2s = sm.get('sector_to_sleeve', {})
        for t, sec in sector.items():
            sleeve.setdefault(t, s2s.get(sec, 'UNASSIGNED'))
    except Exception as e:
        print(f"  (aviso: no pude cargar sleeve/sector maps: {e})")
    return sleeve, sector

TICKER_SLEEVE, TICKER_SECTOR = _load_maps()
# ==============================================================================
# 3. VARIABLES GLOBALES (se reinician por ticker)
# ==============================================================================
close_full = high_full = low_full = None
split_idx  = None
estrategias_aprobadas_temp = {}
MODO_SENSIBILIDAD = False
TICKER_ACTUAL     = ''

# ==============================================================================
# 3b. ESCALADO POR VOLATILIDAD DEL TICKER
# Cada ticker tiene su propia volatilidad anualizada. Los rangos de lookback
# de los indicadores y el umbral de drawdown de la aduana se ajustan según
# qué tan volátil es ESTE activo respecto al promedio de su grupo — en vez
# de usar el mismo rango fijo para BTC que para DOGE, o para XLU que para XLE.
# ==============================================================================
VOL_REFERENCIA_POR_GRUPO = {
    # Volatilidad anualizada "típica" de referencia por grupo — calibrada a ojo,
    # sirve de punto 1.0 para el factor de escala (no es un umbral de filtro).
    'CRYPTO':      0.65,
    'STOCKS':      0.35,
    'STOCKS_ETF':  0.20,
    'COMMODITIES': 0.25,
    'FOREX':       0.10,
}

def vol_referencia_para_ticker(ticker):
    """
    Determina el vol_ref correcto según el TICKER, no según ASSET_GROUP.
    Necesario porque cuando ASSET_GROUP='ALL', todos los grupos se minan
    juntos y un único vol_ref fijo distorsiona el factor de escala de cada uno
    (bug encontrado: 'ALL' no estaba en el diccionario, caía a un default de
    0.30 que inflaba el factor de TODAS las cryptos al techo del clip).
    """
    if ASSET_GROUP == 'DEFENSIVO' or \
       ('TICKERS_DEFENSIVOS' in globals() and ticker in TICKERS_DEFENSIVOS):
        return 0.15   # bonds/oro/managed futures: referencia de vol baja
    if ASSET_GROUP != 'ALL':
        return VOL_REFERENCIA_POR_GRUPO.get(ASSET_GROUP, 0.30)
    if ticker in TICKERS_CRYPTO:
        return VOL_REFERENCIA_POR_GRUPO['CRYPTO']
    if ticker in TICKERS_STOCKS:
        return VOL_REFERENCIA_POR_GRUPO['STOCKS']
    if ticker in TICKERS_STOCKS_ETF:
        return VOL_REFERENCIA_POR_GRUPO['STOCKS_ETF']
    if ticker in TICKERS_COMMODITIES:
        return VOL_REFERENCIA_POR_GRUPO['COMMODITIES']
    if ticker in TICKERS_FOREX:
        return VOL_REFERENCIA_POR_GRUPO['FOREX']
    return 0.30  # fallback solo si el ticker no está en ninguna lista conocida

VOL_TICKER_ACTUAL  = None   # volatilidad anualizada del ticker que se está minando
VOL_SCALE_FACTOR   = 1.0    # factor de escala resultante (vol_ticker / vol_referencia_grupo), clippeado

# Feedback del profesor 2026-07: "you should not volatility scale before a
# gridsearch". OFF = escalar_param devuelve los rangos base sin tocar (factor
# 1.0) y la busqueda de Optuna encuentra sola los lookbacks. El codigo del
# escalado se conserva por referencia; si alguna vez se reactiva, notar que
# el calculo de vol ahora usa SOLO el segmento IS (antes usaba la historia
# completa -> el espacio de busqueda quedaba informado por vol del OOS).
VOL_SCALING = False

def calcular_vol_anualizada(close_series):
    """Volatilidad anualizada simple a partir de retornos diarios."""
    rets = close_series.pct_change().dropna()
    if len(rets) < 30 or rets.std() == 0:
        return None
    return float(rets.std() * np.sqrt(365 if ASSET_GROUP == 'CRYPTO' else 252))

def escalar_param(base_min, base_max, factor=None, min_floor=2, max_ceiling=400):
    """
    Escala un rango de lookback [base_min, base_max] por el factor de volatilidad.
    Activos MÁS volátiles que la referencia → lookbacks más CORTOS (factor > 1 → escala < 1)
    Activos MENOS volátiles → lookbacks más LARGOS (factor < 1 → escala > 1)
    Es lo inverso de la volatilidad: más ruido por vela = ventanas más cortas
    para no diluir la señal; menos ruido = se puede mirar más atrás sin perder nada.
    """
    # FIX: Use the global ticker scale factor if none is passed in the function call
    if factor is None:
        factor = VOL_SCALE_FACTOR if VOL_SCALING else 1.0

    escala = 1.0 / max(factor, 0.3)   # invertido, con piso para no explotar el rango
    escala = float(np.clip(escala, 0.5, 2.0))  # no más de 2x ni menos de 0.5x el rango original
    if not np.isfinite(escala):
        escala = 1.0   # FIX 2026-07: defensa ante VOL_SCALE_FACTOR NaN
    nuevo_min = int(np.clip(base_min * escala, min_floor, max_ceiling))
    nuevo_max = int(np.clip(base_max * escala, nuevo_min + 1, max_ceiling))
    return nuevo_min, nuevo_max
# ==============================================================================
# 4. ADUANA + EVALUACIÓN
# ==============================================================================
# ==============================================================================
# 4. ADUANA + METRICAS COMPLETAS (fusion: antes el Enriquecedor recalculaba
#    esto reconstruyendo senales; ahora se calcula con el portfolio en mano)
# ==============================================================================

def _freq_anual_ticker():
    return 365 if TICKER_ACTUAL.endswith('-USD') else 252


def calc_cagr_pct(total_return_frac, years):
    try:
        if years <= 0:
            return np.nan
        gf = 1.0 + total_return_frac
        if gf <= 0:
            return -100.0
        return round((gf ** (1.0 / years) - 1) * 100, 2)
    except Exception:
        return np.nan


def calc_monthly_consistency(returns_series):
    """% de meses con retorno positivo (0 a 1). Consistencia temporal."""
    try:
        if len(returns_series) < 22:
            return np.nan
        monthly = returns_series.resample('ME').apply(
            lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0.0).dropna()
        if len(monthly) < 3:
            return np.nan
        return round(float((monthly > 0).mean()), 4)
    except Exception:
        return np.nan


def calc_tail_ratio(returns_series):
    """Promedio top 5% dias / |promedio bottom 5%|. >1 = asimetria positiva."""
    try:
        if len(returns_series) < 40:
            return np.nan
        p95 = returns_series.quantile(0.95)
        p05 = returns_series.quantile(0.05)
        top = returns_series[returns_series >= p95].mean()
        bot = returns_series[returns_series <= p05].mean()
        if bot == 0 or np.isnan(bot):
            return np.nan
        return round(min(float(abs(top / bot)), 10.0), 4)
    except Exception:
        return np.nan


def build_strategy_id(ticker, indicator, params):
    params_str = params if isinstance(params, str) else json.dumps(params, sort_keys=True)
    h = hashlib.md5(params_str.encode()).hexdigest()[:8]
    return f"{ticker}__{indicator}__{h}"


def evaluar_y_guardar(indicator_name, params, pf_is, pf_oos):
    global MODO_SENSIBILIDAD, estrategias_aprobadas_temp

    if MODO_SENSIBILIDAD:
        # LEAKAGE FIX (2026-07): la sensibilidad/clustering se mide sobre el
        # Sharpe IS. Antes devolvia OOS -> sensitivity_drop y la seleccion de
        # region estable usaban informacion fuera de muestra ("fitting" que el
        # profesor pidio evitar). El OOS no participa de NINGUNA decision.
        is_s = pf_is.sharpe_ratio()
        return is_s if not pd.isna(is_s) else -1.0

    _grupo = GRUPO_TICKER_ACTUAL if ASSET_GROUP == 'ALL' else ASSET_GROUP
    prof = MINING_PROFILES['DEFENSIVO'] if _grupo == 'DEFENSIVO' \
        else MINING_PROFILES['DEFAULT']

    if pf_is.trades.count() < prof['min_is_trades'] or \
       pf_oos.trades.count() < prof['min_oos_trades']:
        return -1.0

    try:
        lo, hi = prof['dd_clip']
        dd_umbral = float(np.clip(prof['dd_base'] * VOL_SCALE_FACTOR, lo, hi))
        is_dd  = abs(pf_is.max_drawdown())
        oos_dd = abs(pf_oos.max_drawdown())   # solo para reporte
        if is_dd > dd_umbral:                  # LEAKAGE FIX: gate solo IS
            return -1.0

        is_pf = pf_is.trades.profit_factor()
        if pd.isna(is_pf) or is_pf < prof['min_is_pf']:
            return -1.0

        is_sharpe  = pf_is.sharpe_ratio()
        oos_sharpe = pf_oos.sharpe_ratio()   # solo para reporte
        _min_sharpe = MIN_IS_SHARPE_BY_GROUP.get(_grupo, prof['min_is_sharpe'])
        if pd.isna(is_sharpe) or is_sharpe < _min_sharpe:
            return -1.0
        if pd.isna(oos_sharpe):
            oos_sharpe = 0.0

        # ratio oos/is: queda como columna INFORMATIVA (reporte/profesor).
        # LEAKAGE FIX: ya no filtra ni participa del selection_score.
        stability = round(oos_sharpe / max(is_sharpe, 0.01), 4)

        # ---- metricas completas (lo que antes agregaba el Enriquecedor) ----
        freq = _freq_anual_ticker()
        is_rets  = pf_is.returns().fillna(0)
        oos_rets = pf_oos.returns().fillna(0)
        years_is  = max(len(is_rets)  / freq, 0.1)
        years_oos = max(len(oos_rets) / freq, 0.1)

        oos_pf = pf_oos.trades.profit_factor()
        try:
            is_wr  = float(pf_is.trades.win_rate())
            oos_wr = float(pf_oos.trades.win_rate())
        except Exception:
            is_wr = oos_wr = np.nan

        param_key = json.dumps(params, sort_keys=True)
        estrategias_aprobadas_temp[param_key] = {
            'ticker':             TICKER_ACTUAL,
            'asset_group':        _grupo,
            'indicator':          indicator_name,
            'params':             params,
            # IS
            'is_sharpe':          round(float(is_sharpe), 4),
            'is_dd':              round(float(is_dd), 4),
            'is_trades':          int(pf_is.trades.count()),
            'is_trades_per_year': round(pf_is.trades.count() / years_is, 1),
            'is_total_return':    round(float(pf_is.total_return()) * 100, 2),
            'is_cagr':            calc_cagr_pct(float(pf_is.total_return()), years_is),
            'is_profit_factor':   round(float(min(is_pf, 20.0)), 4),
            'is_win_rate':        round(is_wr, 4) if not pd.isna(is_wr) else np.nan,
            'is_monthly_consistency': calc_monthly_consistency(is_rets),
            'is_tail_ratio':          calc_tail_ratio(is_rets),
            # OOS (solo reporte / validation — NO participa de la seleccion)
            'oos_sharpe':         round(float(oos_sharpe), 4),
            'oos_dd':             round(float(oos_dd), 4),
            'oos_trades':         int(pf_oos.trades.count()),
            'oos_trades_per_year': round(pf_oos.trades.count() / years_oos, 1),
            'oos_total_return':   round(float(pf_oos.total_return()) * 100, 2),
            'oos_cagr':           calc_cagr_pct(float(pf_oos.total_return()), years_oos),
            'oos_profit_factor':  round(float(min(oos_pf, 20.0)), 4) if not pd.isna(oos_pf) else 0.0,
            'oos_win_rate':       round(oos_wr, 4) if not pd.isna(oos_wr) else np.nan,
            'monthly_consistency': calc_monthly_consistency(oos_rets),
            'tail_ratio':          calc_tail_ratio(oos_rets),
            'calmar_ratio':       round(float(oos_sharpe) / max(float(oos_dd), 0.01), 4),
            'is_oos_sharpe_diff': round(float(is_sharpe) - float(oos_sharpe), 4),
            'stability_score':    stability,
            # retornos crudos (para dedup por correlacion + exports largos;
            # las equity curves se derivan de estos, no se re-simula nada)
            'is_returns':         json.dumps([round(float(r), 6) for r in is_rets.tolist()]),
            'oos_returns':        json.dumps([round(float(r), 6) for r in oos_rets.tolist()]),
        }
        return is_sharpe

    except Exception:
        return -1.0

# ==============================================================================
# 5. PORTAFOLIOS
# ==============================================================================
def procesar_portafolios(entries, exits):
    """FIX #4 (2026-07): antes se construían DOS backtests independientes (uno
    para IS, otro para OOS), cada uno arrancando desde cash en el día 1 de su
    segmento. Eso perdía trades cross-split (posición abierta al final del IS
    que naturalmente cerraría meses después en OOS quedaba huérfana) y hacía
    que el OOS arrancara sin exposición real. Ahora corremos UN backtest
    continuo sobre close_full y particionamos los retornos por fecha después.
    El wrapper _PFView emula la interfaz de vectorbt.Portfolio que consume
    evaluar_y_guardar (sharpe_ratio, max_drawdown, trades.count, etc.).
    """
    pf = vbt.Portfolio.from_signals(close_full, entries, exits,
                                    init_cash=10000, fees=FEES, freq='1d')
    return _PFView(pf, 0, split_idx), _PFView(pf, split_idx, len(close_full))


class _PFView:
    """Vista de un vectorbt.Portfolio restringida a un tramo temporal.
    Emula lo justo del API de vbt.Portfolio que usa evaluar_y_guardar.
    """
    _FREQ = np.sqrt(252)

    def __init__(self, pf, i0, i1):
        self._pf = pf
        self._i0, self._i1 = i0, i1
        r = pf.returns().fillna(0.0)
        self._r = r.iloc[i0:i1]
        eq0 = 10000.0 if i0 == 0 else float((1.0 + r.iloc[:i0]).prod()) * 10000.0
        self._eq = eq0 * (1.0 + self._r).cumprod()

    # ---- retornos y equity ---------------------------------------------
    def returns(self):
        return self._r

    def total_return(self):
        # retorno del TRAMO (no del portafolio entero) — coherente con la
        # antigua semantica de pf_is.total_return() / pf_oos.total_return()
        if len(self._r) == 0:
            return 0.0
        return float((1.0 + self._r).prod() - 1.0)

    def sharpe_ratio(self):
        if len(self._r) < 2:
            return np.nan
        sd = self._r.std()
        if sd == 0 or np.isnan(sd):
            return np.nan
        return float(self._r.mean() / sd * self._FREQ)

    def max_drawdown(self):
        if len(self._eq) == 0:
            return 0.0
        peak = self._eq.cummax()
        return float(((self._eq - peak) / peak).min())

    # ---- trades del tramo ----------------------------------------------
    @property
    def trades(self):
        return _TradesView(self._pf.trades, self._pf.wrapper.index, self._i0, self._i1)


def _records_col(records, *nombres):
    """Extrae una columna de trades.records sea DataFrame (vectorbt >=1.x)
    o numpy structured array (versiones viejas). Devuelve np.ndarray o None.

    FIX 2026-07: con vectorbt 1.0 trades.records es un DataFrame — el codigo
    anterior hacia records.dtype.names (atributo de structured array), que
    lanzaba AttributeError, el except lo tragaba y profit_factor()/win_rate()
    devolvian NaN SIEMPRE -> la aduana rechazaba el 100% de las estrategias
    por pd.isna(is_pf) y el minado terminaba con cero candidatos.
    """
    if hasattr(records, 'columns'):          # pandas DataFrame
        for c in nombres:
            if c in records.columns:
                return records[c].values
        return None
    names = getattr(getattr(records, 'dtype', None), 'names', None) or ()
    for c in nombres:
        if c in names:
            return records[c]
    return None


class _TradesView:
    """Vista de vbt.Trades filtrada por índice de entrada dentro del tramo."""
    def __init__(self, trades, idx, i0, i1):
        self._trades = trades
        entry_idx = _records_col(trades.records, 'entry_idx')
        if entry_idx is None:
            try:
                entry_idx = trades.records_readable['Entry Index'].values
            except Exception:
                entry_idx = None
        if entry_idx is not None:
            mask = (entry_idx >= i0) & (entry_idx < i1)
            self._records = trades.records[mask]
            self._n = int(mask.sum())
        else:
            self._records = trades.records
            self._n = len(trades.records)

    def count(self):
        return self._n

    def profit_factor(self):
        if self._n == 0:
            return np.nan
        pnl = _records_col(self._records, 'pnl', 'PnL')
        if pnl is None:
            return np.nan
        wins = pnl[pnl > 0].sum()
        losses = -pnl[pnl < 0].sum()
        if losses == 0:
            return np.inf if wins > 0 else np.nan
        return float(wins / losses)

    def win_rate(self):
        if self._n == 0:
            return np.nan
        pnl = _records_col(self._records, 'pnl', 'PnL')
        if pnl is None:
            return np.nan
        return float((pnl > 0).sum() / len(pnl))

# ==============================================================================
# 6. FUNCIONES OBJETIVO (18 indicadores)
# ==============================================================================
def optimize_macd(trial):
    p = {
        'macd_fast':   trial.suggest_int('macd_fast', *escalar_param(5, 40)),
        'macd_slow':   trial.suggest_int('macd_slow', *escalar_param(20, 100)),
        'macd_signal': trial.suggest_int('macd_signal', *escalar_param(5, 30))
    }
    try:
        macd, sig, _ = talib.MACD(close_full, fastperiod=p['macd_fast'], slowperiod=p['macd_slow'], signalperiod=p['macd_signal'])
        m_s   = pd.Series(macd, index=close_full.index)
        sig_s = pd.Series(sig,  index=close_full.index)
        entries = (m_s.shift(1) <= sig_s.shift(1)) & (m_s > sig_s)
        exits   = (m_s.shift(1) >= sig_s.shift(1)) & (m_s < sig_s)
        return evaluar_y_guardar('MACD', p, *procesar_portafolios(entries, exits))
    except Exception as e:
        # print(f"[ERROR MACD] {e}")
        return -1.0

def optimize_3ema(trial):
    p = {
        'ema1': trial.suggest_int('ema1', *escalar_param(5, 30)),
        'ema2': trial.suggest_int('ema2', *escalar_param(20, 60)),
        'ema3': trial.suggest_int('ema3', *escalar_param(50, 200))
    }
    try:
        e1 = pd.Series(talib.EMA(close_full, timeperiod=p['ema1']), index=close_full.index)
        e2 = pd.Series(talib.EMA(close_full, timeperiod=p['ema2']), index=close_full.index)
        e3 = pd.Series(talib.EMA(close_full, timeperiod=p['ema3']), index=close_full.index)
        entries = (e1.shift(1) <= e2.shift(1)) & (e1 > e2) & (e1 > e3)
        exits   = (e1.shift(1) >= e2.shift(1)) & (e1 < e2)
        return evaluar_y_guardar('3EMA', p, *procesar_portafolios(entries, exits))
    except Exception as e:
        # print(f"[ERROR 3EMA] {e}")
        return -1.0

def optimize_dema(trial):
    p = {
        'dema_fast': trial.suggest_int('dema_fast', *escalar_param(5, 40)),
        'dema_slow': trial.suggest_int('dema_slow', *escalar_param(20, 120))
    }
    try:
        d_fast = pd.Series(talib.DEMA(close_full, timeperiod=p['dema_fast']), index=close_full.index)
        d_slow = pd.Series(talib.DEMA(close_full, timeperiod=p['dema_slow']), index=close_full.index)
        entries = (d_fast.shift(1) <= d_slow.shift(1)) & (d_fast > d_slow)
        exits   = (d_fast.shift(1) >= d_slow.shift(1)) & (d_fast < d_slow)
        return evaluar_y_guardar('DEMA', p, *procesar_portafolios(entries, exits))
    except Exception as e:
        # print(f"[ERROR DEMA] {e}")
        return -1.0

def _calc_supertrend(high, low, close, length, multiplier):
    """
    SuperTrend vectorizado con numpy puro.
    Evita el loop con .iloc que era lento y producía errores silenciosos.
    Retorna array de dirección: +1 alcista, -1 bajista.
    """
    atr_vals = talib.ATR(high, low, close, timeperiod=length)
    hl2      = (high + low) / 2.0

    ub = hl2 + multiplier * atr_vals   # upper band básica
    lb = hl2 - multiplier * atr_vals   # lower band básica

    n = len(close)
    final_ub  = np.full(n, np.nan)
    final_lb  = np.full(n, np.nan)
    direction = np.ones(n, dtype=np.int8)  # +1 por defecto

    # Primer índice válido (ATR necesita 'length' velas de calentamiento)
    start = length
    final_ub[start] = ub[start]
    final_lb[start] = lb[start]

    for i in range(start + 1, n):
        # Upper band: solo baja, nunca sube si el precio anterior estaba por debajo
        if not np.isnan(final_ub[i-1]):
            final_ub[i] = ub[i] if close[i-1] > final_ub[i-1] else min(ub[i], final_ub[i-1])
        else:
            final_ub[i] = ub[i]

        # Lower band: solo sube, nunca baja si el precio anterior estaba por encima
        if not np.isnan(final_lb[i-1]):
            final_lb[i] = lb[i] if close[i-1] < final_lb[i-1] else max(lb[i], final_lb[i-1])
        else:
            final_lb[i] = lb[i]

        # Dirección: cruces del precio vs la banda activa
        if close[i] > final_ub[i]:
            direction[i] = 1
        elif close[i] < final_lb[i]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]

    return direction


def optimize_supertrend(trial):
    p = {
        'st_len':    trial.suggest_int('st_len', *escalar_param(5, 50)),
        'st_mult':   trial.suggest_float('st_mult', 1.5, 5.0, step=0.5),  # step más grueso = vecinos más similares
        'confirm_p': trial.suggest_int('confirm_p', 1, 5),   # velas de confirmación antes de entrar
    }
    try:
        high  = high_full.values.astype(np.float64)
        low   = low_full.values.astype(np.float64)
        close = close_full.values.astype(np.float64)

        direction = _calc_supertrend(high, low, close, p['st_len'], p['st_mult'])
        dir_s     = pd.Series(direction, index=close_full.index)

        # Confirmación: la dirección debe mantenerse confirm_p velas consecutivas
        # antes de generar la señal. Reduce falsas entradas en bordes inestables.
        confirmed_long  = True
        confirmed_short = True
        for lag in range(1, p['confirm_p'] + 1):
            confirmed_long  = confirmed_long  & (dir_s.shift(lag) == 1)
            confirmed_short = confirmed_short & (dir_s.shift(lag) == -1)

        # Entrada: primera vela donde hay confirm_p velas alcistas previas
        # y la vela anterior era bajista (es decir, acaba de girar)
        entries = (dir_s.shift(p['confirm_p'] + 1) == -1) & confirmed_long
        exits   = (dir_s.shift(p['confirm_p'] + 1) ==  1) & confirmed_short

        return evaluar_y_guardar('SUPERTREND', p, *procesar_portafolios(entries, exits))
    except Exception as e:
        # print(f"[ERROR SUPERTREND] {e}")
        return -1.0

def optimize_rsi(trial):
    p = {
        'rsi_len':    trial.suggest_int('rsi_len', *escalar_param(5, 30)),
        'oversold':   trial.suggest_int('oversold', 15, 45),
        'overbought': trial.suggest_int('overbought', 55, 85)
    }
    try:
        rsi = pd.Series(talib.RSI(close_full, timeperiod=p['rsi_len']), index=close_full.index)
        entries = (rsi.shift(1) <= p['oversold'])   & (rsi > p['oversold'])
        exits   = (rsi.shift(1) >= p['overbought']) & (rsi < p['overbought'])
        return evaluar_y_guardar('RSI', p, *procesar_portafolios(entries, exits))
    except Exception as e:
        # print(f"[ERROR RSI] {e}")
        return -1.0

def optimize_stc(trial):
    p = {
        'stc_fast':   trial.suggest_int('stc_fast', *escalar_param(10, 40)),
        'stc_slow':   trial.suggest_int('stc_slow', *escalar_param(30, 80)),
        'stc_cycle':  trial.suggest_int('stc_cycle', *escalar_param(5, 30)),
        'stc_smooth': trial.suggest_int('stc_smooth', *escalar_param(2, 7, min_floor=2))
    }
    try:
        macd, _, _ = talib.MACD(close_full, fastperiod=p['stc_fast'], slowperiod=p['stc_slow'])
        macd_smooth = talib.EMA(talib.EMA(macd, timeperiod=p['stc_cycle']), timeperiod=p['stc_smooth'])
        highest = talib.MAX(macd_smooth, timeperiod=p['stc_cycle'])
        lowest  = talib.MIN(macd_smooth, timeperiod=p['stc_cycle'])
        stc_k   = 100 * ((macd_smooth - lowest) / (highest - lowest))
        s_line  = pd.Series(talib.SMA(stc_k, timeperiod=3), index=close_full.index)
        entries = (s_line.shift(1) <= 50) & (s_line > 50)
        exits   = (s_line.shift(1) >= 50) & (s_line < 50)
        return evaluar_y_guardar('STC', p, *procesar_portafolios(entries, exits))
    except Exception as e:
        # print(f"[ERROR STC] {e}")
        return -1.0

def optimize_kama(trial):
    p = {
        'kama_fast': trial.suggest_int('kama_fast', *escalar_param(5, 30)),
        'kama_slow': trial.suggest_int('kama_slow', *escalar_param(20, 100))
    }
    try:
        k_f = pd.Series(talib.KAMA(close_full, timeperiod=p['kama_fast']), index=close_full.index)
        k_s = pd.Series(talib.KAMA(close_full, timeperiod=p['kama_slow']), index=close_full.index)
        entries = (k_f.shift(1) <= k_s.shift(1)) & (k_f > k_s)
        exits   = (k_f.shift(1) >= k_s.shift(1)) & (k_f < k_s)
        return evaluar_y_guardar('KAMA', p, *procesar_portafolios(entries, exits))
    except Exception as e:
        # print(f"[ERROR KAMA] {e}")
        return -1.0

def optimize_chandelier(trial):
    p = {
        'chand_len':  trial.suggest_int('chand_len', *escalar_param(10, 60)),
        'chand_mult': trial.suggest_float('chand_mult', 1.5, 5.0, step=0.5),  # step grueso = más estable
        'trend_ema':  trial.suggest_int('trend_ema', *escalar_param(50, 200)),  # filtro de tendencia: solo entrar si precio > EMA
    }
    try:
        atr      = pd.Series(talib.ATR(high_full, low_full, close_full, timeperiod=p['chand_len']), index=close_full.index)
        ema_trend = pd.Series(talib.EMA(close_full, timeperiod=p['trend_ema']), index=close_full.index)
        l_stop   = high_full.rolling(p['chand_len']).max() - (atr * p['chand_mult'])
        s_stop   = low_full.rolling(p['chand_len']).min()  + (atr * p['chand_mult'])

        # Filtro de régimen: solo operamos long cuando el precio está por encima de la EMA de tendencia
        # Esto evita entrar en mercados bajistas donde Chandelier genera mucho ruido
        in_uptrend = close_full > ema_trend

        entries = (close_full.shift(1) <= l_stop.shift(1)) & (close_full > l_stop) & in_uptrend
        exits   = (close_full.shift(1) >= s_stop.shift(1)) & (close_full < s_stop)
        return evaluar_y_guardar('CHANDELIER', p, *procesar_portafolios(entries, exits))
    except Exception as e:
        # print(f"[ERROR CHANDELIER] {e}")
        return -1.0

def optimize_bbands(trial):
    p = {
        'bb_len': trial.suggest_int('bb_len', *escalar_param(10, 60)),
        'bb_std': trial.suggest_float('bb_std', 1.5, 3.5, step=0.1)
    }
    try:
        u, m, _ = talib.BBANDS(close_full, timeperiod=p['bb_len'], nbdevup=p['bb_std'], nbdevdn=p['bb_std'], matype=0)
        u_s = pd.Series(u, index=close_full.index)
        m_s = pd.Series(m, index=close_full.index)
        entries = (close_full.shift(1) <= u_s.shift(1)) & (close_full > u_s)
        exits   = (close_full.shift(1) >= m_s.shift(1)) & (close_full < m_s)
        return evaluar_y_guardar('BBANDS', p, *procesar_portafolios(entries, exits))
    except Exception as e:
        # print(f"[ERROR BBANDS] {e}")
        return -1.0

def optimize_keltner(trial):
    p = {
        'kc_len':  trial.suggest_int('kc_len', *escalar_param(10, 60)),
        'kc_mult': trial.suggest_float('kc_mult', 1.0, 4.0, step=0.1)
    }
    try:
        ema   = pd.Series(talib.EMA(close_full, timeperiod=p['kc_len']), index=close_full.index)
        atr   = pd.Series(talib.ATR(high_full, low_full, close_full, timeperiod=p['kc_len']), index=close_full.index)
        upper = ema + (p['kc_mult'] * atr)
        entries = (close_full.shift(1) <= upper.shift(1)) & (close_full > upper)
        exits   = (close_full.shift(1) >= ema.shift(1))   & (close_full < ema)
        return evaluar_y_guardar('KELTNER', p, *procesar_portafolios(entries, exits))
    except Exception as e:
        # print(f"[ERROR KELTNER] {e}")
        return -1.0

def optimize_stoch(trial):
    p = {
        'stoch_k':  trial.suggest_int('stoch_k', *escalar_param(3, 40)),
        'stoch_d':  trial.suggest_int('stoch_d', *escalar_param(2, 15)),
        'stoch_os': trial.suggest_int('stoch_os', 10, 35),
        'stoch_ob': trial.suggest_int('stoch_ob', 65, 92)
    }
    try:
        slowk, slowd = talib.STOCH(high_full, low_full, close_full,
                                    fastk_period=p['stoch_k'],
                                    slowk_period=p['stoch_d'],
                                    slowd_period=p['stoch_d'])
        k_s = pd.Series(slowk, index=close_full.index)
        d_s = pd.Series(slowd, index=close_full.index)
        entries = (k_s.shift(1) <= d_s.shift(1)) & (k_s > d_s) & (k_s < p['stoch_os'])
        exits   = (k_s.shift(1) >= d_s.shift(1)) & (k_s < d_s) & (k_s > p['stoch_ob'])
        pf_is, pf_oos = procesar_portafolios(entries, exits)
        if pf_is.trades.count() < 10 or pf_oos.trades.count() < 3:
            return -1.0
        return evaluar_y_guardar('STOCH', p, pf_is, pf_oos)
    except Exception as e:
        # print(f"[ERROR STOCH] {e}")
        return -1.0

def optimize_adx(trial):
    p = {
        'adx_len':    trial.suggest_int('adx_len', *escalar_param(7, 30)),
        'adx_thresh': trial.suggest_int('adx_thresh', 8, 25),   # piso bajado de 10 a 8 — daba 0 en activos volátiles
        'adx_exit':   trial.suggest_int('adx_exit', *escalar_param(5, 20))
    }
    try:
        adx    = pd.Series(talib.ADX(high_full, low_full, close_full, timeperiod=p['adx_len']), index=close_full.index)
        p_di   = pd.Series(talib.PLUS_DI(high_full, low_full, close_full, timeperiod=p['adx_len']), index=close_full.index)
        m_di   = pd.Series(talib.MINUS_DI(high_full, low_full, close_full, timeperiod=p['adx_len']), index=close_full.index)
        ema_ex = pd.Series(talib.EMA(close_full, timeperiod=p['adx_exit']), index=close_full.index)
        entries = (p_di > m_di) & (adx > p['adx_thresh'])
        exits   = (m_di > p_di) | (close_full < ema_ex)
        pf_is, pf_oos = procesar_portafolios(entries, exits)
        # Mínimo relajado: ADX genera menos señales por naturaleza que cruces de medias
        if pf_is.trades.count() < 6 or pf_oos.trades.count() < 2:
            return -1.0
        return evaluar_y_guardar('ADX', p, pf_is, pf_oos)
    except Exception as e:
        # print(f"[ERROR ADX] {e}")
        return -1.0

def optimize_sar(trial):
    p = {
        'sar_step': trial.suggest_float('sar_step', 0.01, 0.05, step=0.005),
        'sar_max':  trial.suggest_float('sar_max',  0.1,  0.4,  step=0.025)
    }
    try:
        sar = pd.Series(talib.SAR(high_full, low_full, acceleration=p['sar_step'], maximum=p['sar_max']), index=close_full.index)
        entries = (close_full.shift(1) <= sar.shift(1)) & (close_full > sar)
        exits   = (close_full.shift(1) >= sar.shift(1)) & (close_full < sar)
        return evaluar_y_guardar('SAR', p, *procesar_portafolios(entries, exits))
    except Exception as e:
        # print(f"[ERROR SAR] {e}")
        return -1.0

def optimize_cci(trial):
    p = {
        'cci_len':   trial.suggest_int('cci_len', *escalar_param(10, 60)),
        'cci_level': trial.suggest_int('cci_level', 80, 150)
    }
    try:
        cci = pd.Series(talib.CCI(high_full, low_full, close_full, timeperiod=p['cci_len']), index=close_full.index)
        entries = (cci.shift(1) <= -p['cci_level']) & (cci > -p['cci_level'])
        exits   = (cci.shift(1) >= p['cci_level'])  & (cci < p['cci_level'])
        return evaluar_y_guardar('CCI', p, *procesar_portafolios(entries, exits))
    except Exception as e:
        # print(f"[ERROR CCI] {e}")
        return -1.0

def optimize_willr(trial):
    p = {
        'will_len': trial.suggest_int('will_len', *escalar_param(10, 40)),
        'will_os':  trial.suggest_int('will_os', -90, -70),
        'will_ob':  trial.suggest_int('will_ob', -30, -10)
    }
    try:
        willr = pd.Series(talib.WILLR(high_full, low_full, close_full, timeperiod=p['will_len']), index=close_full.index)
        entries = (willr.shift(1) <= p['will_os']) & (willr > p['will_os'])
        exits   = (willr.shift(1) >= p['will_ob']) & (willr < p['will_ob'])
        return evaluar_y_guardar('WILLR', p, *procesar_portafolios(entries, exits))
    except Exception as e:
        # print(f"[ERROR WILLR] {e}")
        return -1.0

def optimize_aroon(trial):
    p = {'aroon_len': trial.suggest_int('aroon_len', *escalar_param(10, 50))}
    try:
        d, u = talib.AROON(high_full, low_full, timeperiod=p['aroon_len'])
        u_s  = pd.Series(u, index=close_full.index)
        d_s  = pd.Series(d, index=close_full.index)
        entries = (u_s.shift(1) <= d_s.shift(1)) & (u_s > d_s)
        exits   = (u_s.shift(1) >= d_s.shift(1)) & (u_s < d_s)
        return evaluar_y_guardar('AROON', p, *procesar_portafolios(entries, exits))
    except Exception as e:
        # print(f"[ERROR AROON] {e}")
        return -1.0

def optimize_donchian(trial):
    p = {
        'don_len':  trial.suggest_int('don_len', *escalar_param(20, 120)),
        'exit_len': trial.suggest_int('exit_len', *escalar_param(5, 40))
    }
    try:
        highest  = high_full.rolling(p['don_len']).max()
        exit_low = low_full.rolling(p['exit_len']).min()
        entries  = close_full > highest.shift(1)
        exits    = close_full < exit_low.shift(1)
        return evaluar_y_guardar('DONCHIAN', p, *procesar_portafolios(entries, exits))
    except Exception as e:
        # print(f"[ERROR DONCHIAN] {e}")
        return -1.0

def optimize_tema(trial):
    p = {
        'tema_fast': trial.suggest_int('tema_fast', *escalar_param(5, 30)),
        'tema_slow': trial.suggest_int('tema_slow', *escalar_param(20, 100))
    }
    try:
        t_fast = pd.Series(talib.TEMA(close_full, timeperiod=p['tema_fast']), index=close_full.index)
        t_slow = pd.Series(talib.TEMA(close_full, timeperiod=p['tema_slow']), index=close_full.index)
        entries = (t_fast.shift(1) <= t_slow.shift(1)) & (t_fast > t_slow)
        exits   = (t_fast.shift(1) >= t_slow.shift(1)) & (t_fast < t_slow)
        return evaluar_y_guardar('TEMA', p, *procesar_portafolios(entries, exits))
    except Exception as e:
        # print(f"[ERROR TEMA] {e}")
        return -1.0


# ==============================================================================
# CLUSTERING DE PARÁMETROS (solo para indicadores adaptativos)
# Para KAMA, STC y SUPERTREND: en vez del pico óptimo, busca la REGIÓN
# más estable del espacio de parámetros (menor desviación estándar entre vecinos).
# Esto evita islas de overfitting que pasan la sensibilidad ±2 pero colapsan
# con variaciones mayores.
# ==============================================================================
ADAPTIVE_INDICATORS = {'KAMA', 'STC', 'SUPERTREND'}
N_CLUSTER_NEIGHBORS = 20   # cuántos vecinos cercanos evaluar por candidato

def clustering_params(objective_func, indicator_name, study, top_n=50):
    """
    Toma los top_n trials de Optuna, evalúa N_CLUSTER_NEIGHBORS vecinos para cada uno,
    calcula la std del Sharpe IS en ese vecindario (LEAKAGE FIX 2026-07), y retorna los parámetros del
    centro de la región más estable (menor std).
    """
    global MODO_SENSIBILIDAD
    
    # Tomamos los mejores trials del estudio
    completed = [t for t in study.trials if t.state.name == 'COMPLETE' and t.value is not None and t.value > 0]
    if not completed:
        return None
    
    completed.sort(key=lambda t: t.value, reverse=True)
    top_trials = completed[:top_n]
    
    best_stability = None
    best_params    = None
    
    with _modo_sensibilidad():
        for trial in top_trials:
            params = trial.params
            neighbor_sharpes = []
            
            for param_name, param_value in params.items():
                # Evaluamos vecinos más amplios que en sensibilidad normal
                for variation in range(-N_CLUSTER_NEIGHBORS // 2, N_CLUSTER_NEIGHBORS // 2 + 1):
                    if variation == 0:
                        continue
                    neighbor_params = params.copy()
                    if isinstance(param_value, int):
                        neighbor_params[param_name] += variation
                    else:
                        neighbor_params[param_name] *= (1 + (0.02 * variation))
                    
                    try:
                        t = FixedTrial(neighbor_params)
                        oos_s = objective_func(t)
                        if not pd.isna(oos_s) and oos_s > -0.5:
                            neighbor_sharpes.append(oos_s)
                    except Exception:
                        continue
            
            if len(neighbor_sharpes) >= 5:
                std_sharpe = np.std(neighbor_sharpes)
                mean_sharpe = np.mean(neighbor_sharpes)
                # Queremos: alta media + baja std (estabilidad)
                stability_score = mean_sharpe - std_sharpe
                
                if best_stability is None or stability_score > best_stability:
                    best_stability = stability_score
                    best_params    = params
    
    return best_params

# ==============================================================================
# 7. MOTOR DE SENSIBILIDAD
# ==============================================================================
def sensibilidad_masiva(objective_func, best_params, base_is_sharpe):
    """Perturba cada parametro +-1/+-2 y mide la peor caida del Sharpe IS.
    LEAKAGE FIX 2026-07: antes media la caida del Sharpe OOS -> el filtro de
    robustez usaba informacion fuera de muestra."""
    global MODO_SENSIBILIDAD
    MODO_SENSIBILIDAD = True
    worst_is = base_is_sharpe

    for param_name, param_value in best_params.items():
        for variation in [-2, -1, 1, 2]:
            neighbor_params = best_params.copy()
            if isinstance(param_value, int):
                neighbor_params[param_name] += variation
            else:
                neighbor_params[param_name] *= (1 + (0.03 * variation))
            try:
                trial = FixedTrial(neighbor_params)
                is_s = objective_func(trial)   # MODO_SENSIBILIDAD -> Sharpe IS
                if not pd.isna(is_s) and is_s < worst_is:
                    worst_is = is_s
            except:
                continue

    MODO_SENSIBILIDAD = False
    drop_pct = ((base_is_sharpe - worst_is) / base_is_sharpe) * 100 if base_is_sharpe > 0 else 100.0
    return worst_is, drop_pct

# ==============================================================================
# 8. SETS DE INDICADORES POR TIPO DE ACTIVO
# Cada grupo tiene su propio mix de indicadores según el régimen dominante.
# Agregar/quitar indicadores aquí sin tocar las funciones def de arriba.
# ==============================================================================

# --- TREND FOLLOWING puro (crypto y stocks en tendencia) ---
# Elimina osciladores de reversión que no funcionan en activos tendenciales.
INDICATORS_TREND = {
    'MACD':       optimize_macd,
    '3EMA':       optimize_3ema,
    'DEMA':       optimize_dema,
    'SUPERTREND': optimize_supertrend,
    'STC':        optimize_stc,
    'KAMA':       optimize_kama,
    'CHANDELIER': optimize_chandelier,
    'BBANDS':     optimize_bbands,    # versión breakout
    'KELTNER':    optimize_keltner,   # versión breakout
    'ADX':        optimize_adx,
    'DONCHIAN':   optimize_donchian,
    'TEMA':       optimize_tema,
}

# --- MEAN REVERSION + TREND (forex y commodities) ---
# Incluye osciladores que aprovechan rangos y reversiones.
# También mantiene los de tendencia porque forex tiene tendencias largas ocasionalmente.
INDICATORS_FOREX_COMMODITIES = {
    'MACD':     optimize_macd,
    'DONCHIAN': optimize_donchian,
    'RSI':      optimize_rsi,
    'STOCH':    optimize_stoch,
    'WILLR':    optimize_willr,
    'CCI':      optimize_cci,
    'BBANDS':   optimize_bbands,
    'SAR':      optimize_sar,
    'AROON':    optimize_aroon,
    'ADX':      optimize_adx,
    'KELTNER':  optimize_keltner,
}

# --- ETFs e índices (mix tendencia + filtros de volatilidad) ---
INDICATORS_ETF = {
    'MACD':       optimize_macd,
    '3EMA':       optimize_3ema,
    'TEMA':       optimize_tema,
    'DEMA':       optimize_dema,
    'DONCHIAN':   optimize_donchian,
    'SUPERTREND': optimize_supertrend,
    'CHANDELIER': optimize_chandelier,
    'RSI':        optimize_rsi,       # en ETFs de baja vol RSI sí funciona
    'CCI':        optimize_cci,
    'ADX':        optimize_adx,
    'KAMA':       optimize_kama,
}

# Selector automático según ASSET_GROUP
if ASSET_GROUP in ('CRYPTO', 'STOCKS'):
    strategies_to_mine = INDICATORS_TREND
elif ASSET_GROUP in ('FOREX', 'COMMODITIES'):
    strategies_to_mine = INDICATORS_FOREX_COMMODITIES
elif ASSET_GROUP == 'STOCKS_ETF':
    strategies_to_mine = INDICATORS_ETF
else:  # ALL: usamos el set completo
    strategies_to_mine = {
        'MACD':       optimize_macd,
        '3EMA':       optimize_3ema,
        'DEMA':       optimize_dema,
        'SUPERTREND': optimize_supertrend,
        'RSI':        optimize_rsi,
        'STC':        optimize_stc,
        'KAMA':       optimize_kama,
        'CHANDELIER': optimize_chandelier,
        'BBANDS':     optimize_bbands,
        'KELTNER':    optimize_keltner,
        'STOCH':      optimize_stoch,
        'ADX':        optimize_adx,
        'SAR':        optimize_sar,
        'CCI':        optimize_cci,
        'WILLR':      optimize_willr,
        'AROON':      optimize_aroon,
        'DONCHIAN':   optimize_donchian,
        'TEMA':       optimize_tema,
    }

def print_progress(study, trial):
    if trial.number % 500 == 0 and trial.number > 0:
        print(f"   [Progreso] {trial.number} / {N_TRIALS} iteraciones completadas...")


# ==============================================================================
# 8b. UNIVERSO + INDICADORES DEFENSIVOS (absorbidos de Optuna_Defensivo.py)
# ==============================================================================
TICKERS_DEFENSIVOS = [
    # ============ DIVERSIFIERS (el sleeve esencial — sweep maestro 2026-07) ============
    # Bonos / renta fija
    'TLT', 'BND', 'AGGY', 'ANGL', 'HYD', 'NTSX',
    # Metales preciosos (spot ETFs; los futuros =F los mina COMMODITIES)
    'GLD', 'SLV', 'GDX', 'GDXJ', 'REMX', 'PPLT', 'PALL', 'CPER',
    # Managed futures / hedge
    'DBMF', 'KMLM', 'BTAL', 'USMF', 'WTMF', 'KRBN',
    # FX ETFs (exposición a monedas SIN swap — vía Hapi; los pares =X
    # los mina FOREX y ejecutan por Vantage CFD)
    'FXE', 'FXY', 'FXB', 'UUP',
    # ============ Sector ETFs de sleeves vivos (GROWTH/CYCLICAL) ============
    'XLF', 'XLI', 'XLY', 'XAR',
    # NOTA (2026-07): XLU/XLP/XLV removidos — mapean a DEFENSIVE_QUALITY,
    # sleeve excluido del portafolio; minarlos era cómputo tirado.
    # El nombre del grupo sigue siendo 'DEFENSIVO' por compatibilidad con el
    # pipeline downstream (asset_group en CSVs), pero su rol real ahora es
    # "activos lentos": DIVERSIFIERS + sector ETFs. Los 5 indicadores lentos
    # (GOLDEN_CROSS, EMA/DONCHIAN/ADX/KAMA largos) son ideales para esta clase.
]

# Nota: muchos de estos (BUZZ, FRDM, QTUM, OKLO-like) tienen historia corta.
# El script los saltea automáticamente si hay menos de 300 velas — igual que
# el minero principal. No hace falta editarlos a mano.

print(f"📋 Universo defensivo: {len(TICKERS_DEFENSIVOS)} tickers")

def optimize_golden_cross(trial):
    """Cruce de medias largas — el clásico 50/200 generalizado."""
    p = {
        'ma_fast': trial.suggest_int('ma_fast', *escalar_param(30, 80)),
        'ma_slow': trial.suggest_int('ma_slow', *escalar_param(100, 250)),
    }
    try:
        f = pd.Series(talib.SMA(close_full, p['ma_fast']), index=close_full.index)
        s = pd.Series(talib.SMA(close_full, p['ma_slow']), index=close_full.index)
        entries = (f.shift(1) <= s.shift(1)) & (f > s)
        exits   = (f.shift(1) >= s.shift(1)) & (f < s)
        return evaluar_y_guardar('GOLDEN_CROSS', p, *procesar_portafolios(entries, exits))
    except Exception:
        return -1.0

def optimize_ema_largo(trial):
    """EMA largo simple — para activos sideways que necesitan paciencia."""
    p = {
        'ema_fast': trial.suggest_int('ema_fast', *escalar_param(40, 100)),
        'ema_slow': trial.suggest_int('ema_slow', *escalar_param(120, 300)),
    }
    try:
        f = pd.Series(talib.EMA(close_full, p['ema_fast']), index=close_full.index)
        s = pd.Series(talib.EMA(close_full, p['ema_slow']), index=close_full.index)
        entries = (f.shift(1) <= s.shift(1)) & (f > s)
        exits   = (f.shift(1) >= s.shift(1)) & (f < s)
        return evaluar_y_guardar('EMA_LARGO', p, *procesar_portafolios(entries, exits))
    except Exception:
        return -1.0

def optimize_donchian_largo(trial):
    """Donchian con lookbacks largos — breakout conservador para sectores lentos."""
    p = {
        'don_len':  trial.suggest_int('don_len', *escalar_param(60, 250)),
        'exit_len': trial.suggest_int('exit_len', *escalar_param(20, 100)),
    }
    try:
        highest  = high_full.rolling(p['don_len']).max()
        exit_low = low_full.rolling(p['exit_len']).min()
        entries  = close_full > highest.shift(1)
        exits    = close_full < exit_low.shift(1)
        return evaluar_y_guardar('DONCHIAN_LARGO', p, *procesar_portafolios(entries, exits))
    except Exception:
        return -1.0

def optimize_adx_largo(trial):
    """ADX con período largo — confirma tendencia antes de comprometerse."""
    p = {
        'adx_len':    trial.suggest_int('adx_len', *escalar_param(20, 50)),
        'adx_thresh': trial.suggest_int('adx_thresh', 18, 35),
        'adx_exit':   trial.suggest_int('adx_exit', *escalar_param(30, 100)),
    }
    try:
        adx    = pd.Series(talib.ADX(high_full, low_full, close_full, p['adx_len']), index=close_full.index)
        p_di   = pd.Series(talib.PLUS_DI(high_full, low_full, close_full, p['adx_len']), index=close_full.index)
        m_di   = pd.Series(talib.MINUS_DI(high_full, low_full, close_full, p['adx_len']), index=close_full.index)
        ema_ex = pd.Series(talib.EMA(close_full, p['adx_exit']), index=close_full.index)
        entries = (p_di > m_di) & (adx > p['adx_thresh'])
        exits   = (m_di > p_di) | (close_full < ema_ex)
        pf_is, pf_oos = procesar_portafolios(entries, exits)
        if pf_is.trades.count() < 8 or pf_oos.trades.count() < 3:
            return -1.0
        return evaluar_y_guardar('ADX_LARGO', p, pf_is, pf_oos)
    except Exception:
        return -1.0

def optimize_kama_largo(trial):
    """KAMA con períodos largos — se adapta sin sobre-reaccionar al ruido."""
    p = {
        'kama_fast': trial.suggest_int('kama_fast', *escalar_param(20, 60)),
        'kama_slow': trial.suggest_int('kama_slow', *escalar_param(80, 200)),
    }
    try:
        kf = pd.Series(talib.KAMA(close_full, p['kama_fast']), index=close_full.index)
        ks = pd.Series(talib.KAMA(close_full, p['kama_slow']), index=close_full.index)
        entries = (kf.shift(1) <= ks.shift(1)) & (kf > ks)
        exits   = (kf.shift(1) >= ks.shift(1)) & (kf < ks)
        return evaluar_y_guardar('KAMA_LARGO', p, *procesar_portafolios(entries, exits))
    except Exception:
        return -1.0


INDICATORS_DEFENSIVO = {
    'GOLDEN_CROSS':   optimize_golden_cross,
    'EMA_LARGO':      optimize_ema_largo,
    'DONCHIAN_LARGO': optimize_donchian_largo,
    'ADX_LARGO':      optimize_adx_largo,
    'KAMA_LARGO':     optimize_kama_largo,
}
if ASSET_GROUP == 'DEFENSIVO':
    strategies_to_mine = INDICATORS_DEFENSIVO
    TICKERS = TICKERS_DEFENSIVOS
elif ASSET_GROUP == 'ALL':
    TICKERS = TICKERS + TICKERS_DEFENSIVOS
    # (los tickers DEFENSIVO usan INDICATORS_DEFENSIVO per-ticker en main)

# ==============================================================================
# 10. SELECCION TOP-N POR TICKER — 100% IS (antes: Enriquecedor)
# ==============================================================================

def seleccionar_top_ticker(df):
    """Filtra y ordena candidatos de UN ticker usando solo metricas IS.
    El OOS queda en las columnas para reporte, jamas filtra ni ordena."""
    antes = len(df)
    # LEAKAGE FIX 2026-07: stability_score (= oos/is) fuera del filtro y del
    # score — era OOS encubierto con 20% de peso. Queda como columna
    # informativa. Su rol de "robustez" lo cumple sensitivity_drop (ahora IS).
    checks = {
        f'is_sharpe >= {MIN_IS_SHARPE}':        df['is_sharpe']         >= MIN_IS_SHARPE,
        f'sensitivity_drop <= {MAX_SENS_DROP}': df['sensitivity_drop']  <= MAX_SENS_DROP,
        f'is_dd <= {MAX_IS_DD}':                df['is_dd']             <= MAX_IS_DD,
        f'is_profit_factor >= {MIN_IS_PF}':     df['is_profit_factor']  >= MIN_IS_PF,
        f'is_trades/yr >= {MIN_TRADES_PER_YEAR}':  df['is_trades_per_year']  >= MIN_TRADES_PER_YEAR,
        f'oos_trades/yr >= {MIN_TRADES_PER_YEAR}': df['oos_trades_per_year'] >= MIN_TRADES_PER_YEAR,
    }
    if DEBUG_FUNNEL and antes > 0:
        print(f"     --- embudo ({antes} candidatos) ---")
        for nombre, mask in checks.items():
            print(f"       {nombre:40s} pasan {int(mask.fillna(False).sum()):5d}/{antes}")

    mask_total = pd.Series(True, index=df.index)
    for mask in checks.values():
        mask_total &= mask.fillna(False)
    df = df[mask_total]
    print(f"     Filtros IS + actividad minima: {antes} -> {len(df)}")
    if len(df) == 0:
        return pd.DataFrame()

    mc = df['is_monthly_consistency'].fillna(0.5)
    tr = df['is_tail_ratio'].fillna(1.0).clip(upper=3.0)
    df = df.copy()
    # FIX #8 (2026-07): pesos del selection_score con justificacion documentada.
    # Los rangos tipicos IS-only son: sharpe 0.3-2.0, PF (clipeado) 1.0-3.0,
    # mc 0.3-0.8, tr/3 0.3-1.0, sens_drop 0-25, is_dd 0.05-0.55. Los pesos
    # normalizan magnitudes y priorizan Sharpe (calidad de retorno ajustado a
    # riesgo) sobre metricas complementarias. NO es una funcion optimizada;
    # es una combinacion RAZONABLE, y su rol es ordenar dentro del pool que
    # YA paso todos los gates duros. Todas las metricas son IS.
    #
    # PESOS Y RACIONAL:
    #   +0.45  is_sharpe            metrica primaria (retorno/riesgo)
    #   +0.15  monthly_consistency  premia distribucion pareja (no un solo mes)
    #   +0.12  profit_factor        gana lo suficiente relativo a lo que pierde
    #   +0.08  tail_ratio/3         asimetria positiva (colas)
    #   -0.10  sensitivity_drop     penaliza picos afilados en la superficie IS
    #                               (mismo caveat que ya discutimos: mide
    #                               curvatura del maximo, no generalizacion)
    #   -0.10  is_dd                penaliza recorrido feo aunque el Sharpe
    #                               termine bien (nadie tolera el DD en vivo)
    df['selection_score'] = (
        df['is_sharpe']                          * 0.45 +
        df['is_profit_factor'].clip(upper=3.0)   * 0.12 +
        mc                                        * 0.15 +
        (tr / 3.0)                                * 0.08 -
        df['sensitivity_drop']                    * 0.10 -
        df['is_dd']                               * 0.10
    )

    # dedup por correlacion de retornos IS entre finalistas
    # (LEAKAGE FIX 2026-07: antes usaba retornos OOS — no seleccionaba por
    # performance, pero espiaba datos fuera de muestra; IS cumple igual)
    # FIX #6 (2026-07): ademas del gate de correlacion, exigir DIVERSIDAD
    # POR FAMILIA — MACD/EMA/KAMA son todos "cruce de medias" y aunque su
    # correlacion sea 0.4-0.7 (bajo el CORR_MAX de 0.85), aportan poco al
    # ensemble intra-ticker. Si ya hay 2 estrategias de la misma familia
    # entre las seleccionadas, la tercera slot se reserva para otra familia.
    df = df.sort_values('selection_score', ascending=False)
    selected = []
    for _, row in df.iterrows():
        if len(selected) >= TOP_N:
            break
        try:
            ret_curr = pd.Series(json.loads(row['is_returns']))
        except Exception:
            continue
        fam_curr = FAMILIA_INDICADOR.get(row['indicator'], 'OTRA')
        n_misma_fam = sum(FAMILIA_INDICADOR.get(s['indicator'], 'OTRA') == fam_curr
                          for s in selected)
        # regla: hasta 2 de la misma familia; la 3ra slot obliga a familia nueva
        if len(selected) >= 2 and n_misma_fam >= 2:
            continue
        too_similar = False
        for s in selected:
            try:
                ret_prev = pd.Series(json.loads(s['is_returns']))
                n = min(len(ret_curr), len(ret_prev))
                if n < 20:
                    continue
                corr = ret_curr.iloc[-n:].corr(ret_prev.iloc[-n:])
                if not np.isnan(corr) and corr > CORR_MAX:
                    too_similar = True
                    break
            except Exception:
                pass
        if not too_similar:
            selected.append(row)
    return pd.DataFrame(selected)


def exportar_finalistas(df_top, fechas_is, fechas_oos,
                        buf_top3, buf_returns, buf_equity):
    """Vuelca finalistas a los buffers globales. Equity derivada de los
    retornos guardados — NO se re-simula nada."""
    for _, row in df_top.iterrows():
        sid = build_strategy_id(row['ticker'], row['indicator'], row['params'])
        rec = row.to_dict()
        rec['strategy_id'] = sid
        rec['sleeve']      = TICKER_SLEEVE.get(row['ticker'], 'UNASSIGNED')
        rec['sector']      = TICKER_SECTOR.get(row['ticker'], 'UNKNOWN')
        rec['side']        = rec.get('side', 'LONG')
        rec['params']      = rec['params'] if isinstance(rec['params'], str) \
            else json.dumps(rec['params'], sort_keys=True)
        is_rets  = json.loads(rec.pop('is_returns'))
        oos_rets = json.loads(rec.pop('oos_returns'))
        buf_top3.append(rec)

        k = min(len(oos_rets), len(fechas_oos))
        for i in range(k):
            buf_returns.append({'strategy_id': sid, 'fecha': fechas_oos[i],
                                'retorno_oos': float(oos_rets[i])})
        eq = 10000.0
        for i in range(min(len(is_rets), len(fechas_is))):
            eq *= (1.0 + is_rets[i])
            buf_equity.append({'strategy_id': sid, 'fecha': fechas_is[i],
                               'equity': round(eq, 2), 'segmento': 'IS'})
        eq = 10000.0
        for i in range(k):
            eq *= (1.0 + oos_rets[i])
            buf_equity.append({'strategy_id': sid, 'fecha': fechas_oos[i],
                               'equity': round(eq, 2), 'segmento': 'OOS'})


def _leer_existente(base):
    for ext in ('parquet', 'csv'):
        p = f"{base}.{ext}"
        if os.path.exists(p):
            try:
                return (pd.read_parquet(p) if ext == 'parquet'
                        else pd.read_csv(p)), ext
            except Exception:
                pass
    return None, 'csv'


_LOCK_PATH = 'quant_miner_outputs.lock'

def _adquirir_lock(timeout_s=1800, stale_s=1800):
    """Lock file simple multiplataforma. Espera hasta timeout si otra
    instancia está consolidando; roba locks viejos (proceso muerto)."""
    t0 = time.time()
    while True:
        try:
            fd = os.open(_LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(_LOCK_PATH) > stale_s:
                    os.remove(_LOCK_PATH)   # lock viejo: instancia muerta
                    continue
            except OSError:
                pass
            if time.time() - t0 > timeout_s:
                return False
            print("  ⏳ otra instancia está consolidando outputs; esperando...")
            time.sleep(10)


def _liberar_lock():
    try:
        os.remove(_LOCK_PATH)
    except OSError:
        pass


def consolidar_outputs(buf_top3, buf_returns, buf_equity, tickers_re_minados):
    """Fusiona lo nuevo con lo existente: las filas de tickers re-minados se
    REEMPLAZAN; el resto de grupos/corridas anteriores se conserva.
    Protegido con lock file: seguro para corridas paralelas por grupo."""
    if not _adquirir_lock():
        print("  ⚠️  No pude adquirir el lock en 30 min. Guardando en archivos")
        print("     _pendiente_{PID} — fusionalos a mano o re-corré consolidación.")
        pd.DataFrame(buf_top3).to_csv(f"top3_pendiente_{os.getpid()}.csv", index=False)
        pd.DataFrame(buf_returns).to_csv(f"returns_pendiente_{os.getpid()}.csv", index=False)
        pd.DataFrame(buf_equity).to_csv(f"equity_pendiente_{os.getpid()}.csv", index=False)
        return
    try:
        _consolidar_outputs_inner(buf_top3, buf_returns, buf_equity,
                                  tickers_re_minados)
    finally:
        _liberar_lock()


def _consolidar_outputs_inner(buf_top3, buf_returns, buf_equity, tickers_re_minados):
    if not buf_top3:
        print("\n(No hubo finalistas nuevos; no se tocan los outputs globales.)")
        return

    df_new = pd.DataFrame(buf_top3)
    viejo = pd.read_csv(OUT_TOP3) if os.path.exists(OUT_TOP3) else None
    if viejo is not None:
        viejo = viejo[~viejo['ticker'].isin(tickers_re_minados)]
        df_all = pd.concat([viejo, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(OUT_TOP3, index=False)
    print(f"\n✅ {OUT_TOP3}: {len(df_all)} estrategias "
          f"({len(df_new)} nuevas/actualizadas)")

    for base, buf, keycol in ((OUT_RETURNS, buf_returns, 'retorno_oos'),
                              (OUT_EQUITY,  buf_equity,  'equity')):
        df_b = pd.DataFrame(buf)
        old, ext = _leer_existente(base)
        if old is not None and len(old):
            pref = old['strategy_id'].str.split('__').str[0]
            old = old[~pref.isin(tickers_re_minados)]
            df_b = pd.concat([old, df_b], ignore_index=True)
        df_b = df_b.drop_duplicates(['strategy_id', 'fecha'])
        try:
            df_b.to_parquet(f"{base}.parquet", index=False)
            print(f"✅ {base}.parquet: {len(df_b):,} filas")
        except Exception:
            df_b.to_csv(f"{base}.csv", index=False)
            print(f"✅ {base}.csv: {len(df_b):,} filas")


# ==============================================================================
# 11. BUCLE MAESTRO — minado + seleccion + outputs, todo en una pasada
# ==============================================================================

def _validar_datos(data, ticker, grupo):
    """FIX #7 (2026-07): saneamiento previo al minado. Devuelve (data_limpia, motivo)
    o (None, motivo) si el ticker debe saltearse. Cubre los patrones de basura
    de Yahoo que llegaron hasta el Balancer en corridas anteriores:
      - retornos absurdos por prints malos (>150% crypto, >60% resto)
      - gaps de precio > 30% (probable split no ajustado)
      - precios <= 0 (dato invalido)
      - historia mas corta que MIN_VELAS despues de sanear
    """
    if data is None or 'Close' not in data.columns:
        return None, "sin datos"
    close = data['Close'].squeeze()
    if (close <= 0).any() or close.isna().all():
        return None, "precios invalidos (<=0 o todos NaN)"
    r = close.pct_change().dropna()
    if len(r) < 300:
        return None, f"historia corta ({len(r)} velas)"
    umbral = 1.50 if grupo == 'CRYPTO' else 0.60
    outliers = r.abs() > umbral
    n_out = int(outliers.sum())
    if n_out > 0:
        # winsorize: NO borrar dias (rompe alineacion), reemplazar por el
        # signo x umbral. Un outlier real se convierte en un dia grande pero
        # no destructivo; un print malo pierde su capacidad de contaminar.
        r_clip = r.clip(-umbral, umbral)
        # reconstruir close desde el retorno clippeado
        c0 = float(close.iloc[0])
        close_fix = pd.Series(c0 * (1.0 + r_clip).cumprod().values,
                              index=r_clip.index).reindex(close.index).ffill().bfill()
        data = data.copy()
        # ajustar OHL proporcionalmente al Close saneado
        ratio = (close_fix / close).replace([np.inf, -np.inf], 1.0).fillna(1.0)
        data['Close'] = close_fix
        for c in ('High', 'Low', 'Open'):
            if c in data.columns:
                data[c] = data[c].squeeze() * ratio
    return data, (f"winsorized {n_out} outliers > {umbral:.0%}" if n_out else "ok")


def descargar_datos(tickers):
    """Pre-carga TODO antes de minar via el cache local de precios
    (precios_cache.py): lo cacheado se lee de disco, lo que falta se
    descarga con retries y queda cacheado. Si se corta internet después de
    esta etapa, el minado continúa igual (los datos ya están en memoria).
    FIX #7: ahora ademas VALIDA cada serie (outliers, precios <=0, historia)."""
    from precios_cache import obtener_precios
    print(f"\n📥 PRE-CARGANDO datos para {len(tickers)} tickers (cache local)...")
    out = {}
    for ticker in tickers:
        try:
            start = START_DATE or GROUP_START[grupo_de_ticker(ticker)]
            data = obtener_precios(ticker, start)
            data, motivo = _validar_datos(data, ticker, grupo_de_ticker(ticker))
            if data is None:
                print(f"  ⏭  {ticker}: {motivo} - salteado")
                continue
            if motivo != "ok":
                print(f"  ⚠  {ticker}: {motivo}")
            if data is not None and len(data) >= 300:
                out[ticker] = data
                print(f"  ✓ {ticker}: {len(data)} velas")
            else:
                print(f"  ⚠️  {ticker}: datos insuficientes — se saltea")
        except Exception as e:
            print(f"  ❌ {ticker}: {e} — se saltea")
    return out


def main():
    global TICKER_ACTUAL, close_full, high_full, low_full, split_idx
    global VOL_TICKER_ACTUAL, VOL_SCALE_FACTOR
    global GRUPO_TICKER_ACTUAL, FEES

    print(f"\n{'='*70}")
    print(f"🚀 QUANT MINER — Grupo: {ASSET_GROUP}  |  SPLIT {SPLIT:.0%} IS")
    print(f"   Tickers: {len(TICKERS)} | Indicadores: {len(strategies_to_mine)}"
          f" | Trials: {N_TRIALS}")
    if ASSET_GROUP == 'ALL':
        print(f"   Historia y fees POR TICKER según grupo (ver GROUP_START/GROUP_FEES)")
    else:
        print(f"   Historia desde: {START_DATE} | Fees: {FEES*100:.3f}%")
    print(f"{'='*70}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pendientes = []
    for t in TICKERS:
        safe = t.replace('-', '_').replace('=', '_').replace('/', '_')
        raw_path = os.path.join(OUTPUT_DIR,
                                f"estrategias_{grupo_de_ticker(t)}_{safe}.csv")
        if SKIP_EXISTING and os.path.exists(raw_path):
            print(f"  ⏭  {t}: ya minado ({raw_path}); se saltea (SKIP_EXISTING).")
            continue
        pendientes.append(t)

    datos = descargar_datos(pendientes)
    if not datos:
        print("❌ Nada para minar (todo salteado o sin datos).")
        return

    buf_top3, buf_returns, buf_equity = [], [], []
    tickers_ok = []

    for TICKER_ACTUAL in list(datos.keys()):
        GRUPO_TICKER_ACTUAL = grupo_de_ticker(TICKER_ACTUAL)
        if ASSET_GROUP == 'ALL':
            FEES = GROUP_FEES[GRUPO_TICKER_ACTUAL]
        inds_ticker = (INDICATORS_DEFENSIVO
                       if GRUPO_TICKER_ACTUAL == 'DEFENSIVO'
                       else strategies_to_mine)
        print(f"\n{'─'*70}\n🪙  {TICKER_ACTUAL}  "
              f"[{GRUPO_TICKER_ACTUAL} | fees {FEES*100:.3f}% | "
              f"{len(inds_ticker)} indicadores]\n{'─'*70}")
        try:
            data = datos[TICKER_ACTUAL]
            close_full = data['Close'].squeeze()
            high_full  = data['High'].squeeze()
            low_full   = data['Low'].squeeze()
            split_idx  = int(len(close_full) * SPLIT)

            # Vol SOLO sobre el segmento IS (fix leakage 2026-07: antes se
            # calculaba sobre close_full completo -> el factor de escala de
            # los rangos de busqueda quedaba informado por vol del OOS).
            VOL_TICKER_ACTUAL = calcular_vol_anualizada(close_full.iloc[:split_idx])
            if VOL_SCALING:
                vol_ref = vol_referencia_para_ticker(TICKER_ACTUAL)
                # NaN es truthy — 'if VOL_TICKER_ACTUAL' no atrapaba vol NaN
                # (p.ej. SHIB-USD: precios microscopicos/tramos planos).
                try:
                    _ratio = float(VOL_TICKER_ACTUAL) / float(vol_ref)
                    VOL_SCALE_FACTOR = (float(np.clip(_ratio, 0.4, 3.5))
                                        if np.isfinite(_ratio) else 1.0)
                except (TypeError, ZeroDivisionError, ValueError):
                    VOL_SCALE_FACTOR = 1.0
                _escala_txt = f"escala {VOL_SCALE_FACTOR:.2f}x"
            else:
                VOL_SCALE_FACTOR = 1.0
                _escala_txt = "escala OFF (rangos base)"
            print(f"✅ Velas: {len(close_full)} | IS: {split_idx} | "
                  f"OOS: {len(close_full)-split_idx} | {_escala_txt}")

            # (Guard de calidad se movio a _validar_datos en descargar_datos —
            # los datos que llegan aca ya estan saneados. Fix #7 2026-07.)
        except Exception as e:
            print(f"❌ Error preparando {TICKER_ACTUAL}: {e}. Saltando.")
            continue

        fechas_is  = [d.strftime('%Y-%m-%d') for d in close_full.index[:split_idx]]
        fechas_oos = [d.strftime('%Y-%m-%d') for d in close_full.index[split_idx:]]

        resultados_ticker = []
        for ind_name, obj_func in inds_ticker.items():
            estrategias_aprobadas_temp.clear()
            print(f"\n  🔍 {ind_name}...", end='', flush=True)
            study = optuna.create_study(direction='maximize')
            study.optimize(obj_func, n_trials=N_TRIALS, callbacks=[print_progress])

            candidatos = list(estrategias_aprobadas_temp.values())
            print(f" Aduana: {len(candidatos)}", end='', flush=True)

            sobrevivientes = 0
            # Cap 2026-07: solo el top-K por Sharpe IS entra a la bateria de
            # sensibilidad (4 backtests x parametro x candidato — con cientos
            # de candidatos dominaba el costo total: tickers de 25min pasaban
            # a 5h). Los candidatos fuera del top-60 por indicador casi nunca
            # llegan al top-3 del ticker (la re-seleccion pre-corta top-50
            # por ticker sobre selection_score, 45% ponderado por is_sharpe).
            candidatos = sorted(candidatos, key=lambda s: s.get('is_sharpe', 0),
                                reverse=True)[:SENSIBILIDAD_MAX_CANDIDATOS]
            for strat in candidatos:
                # LEAKAGE FIX: sensibilidad y gate sobre Sharpe IS
                worst_is, drop_pct = sensibilidad_masiva(
                    obj_func, strat['params'], strat['is_sharpe'])
                if drop_pct <= 25.0 and worst_is >= 0.35:
                    strat['sensitivity_drop'] = round(drop_pct, 1)
                    # robust_score IS-only (antes mezclaba 70% OOS)
                    strat['robust_score']     = round(
                        strat['is_sharpe'] * (1 - drop_pct / 100.0), 4)
                    strat['params']           = json.dumps(strat['params'])
                    strat['side']             = 'LONG'
                    resultados_ticker.append(strat)
                    sobrevivientes += 1
            print(f" -> 🏆 {sobrevivientes}")

        if not resultados_ticker:
            print(f"\n  ⚠️  Ninguna estrategia sobrevivio para {TICKER_ACTUAL}.")
            continue

        df_ticker = pd.DataFrame(resultados_ticker).sort_values(
            'robust_score', ascending=False)
        safe = TICKER_ACTUAL.replace('-', '_').replace('=', '_').replace('/', '_')
        raw_path = os.path.join(OUTPUT_DIR,
                                f"estrategias_{GRUPO_TICKER_ACTUAL}_{safe}.csv")
        df_ticker.drop(columns=['is_returns', 'oos_returns']).to_csv(raw_path, index=False)
        print(f"\n  💾 {len(df_ticker)} candidatas -> '{raw_path}'")

        df_top = seleccionar_top_ticker(df_ticker)
        if len(df_top):
            exportar_finalistas(df_top, fechas_is, fechas_oos,
                                buf_top3, buf_returns, buf_equity)
            tickers_ok.append(TICKER_ACTUAL)
            print(f"     🏆 Top final: {len(df_top)} estrategias")
        else:
            print(f"     ⚠️  Ninguna paso la seleccion IS.")

    consolidar_outputs(buf_top3, buf_returns, buf_equity, set(datos.keys()))
    print(f"\n\n✅ MINADO + SELECCION COMPLETOS. "
          f"{len(tickers_ok)} tickers con finalistas.")


# ==============================================================================
# 12. RECONSTRUCCIÓN DE SEÑALES  (fusión 2026-07 — antes en Enriquecedor_v3.py)
# ==============================================================================
# Reconstruye las señales entry/exit de una estrategia ya minada a partir de
# (indicator, params). Usada por la re-selección (abajo) y por mt5_ejecutor.py.
# Cubre los 18 indicadores del minero principal + 5 defensivos + los MR.

def reconstruir_senales(indicator, params, close, high, low):
    p = params
    try:
        if indicator == 'MACD':
            macd, sig, _ = talib.MACD(close, fastperiod=p['macd_fast'],
                                       slowperiod=p['macd_slow'], signalperiod=p['macd_signal'])
            m, s = pd.Series(macd, index=close.index), pd.Series(sig, index=close.index)
            return (m.shift(1) <= s.shift(1)) & (m > s), (m.shift(1) >= s.shift(1)) & (m < s)

        elif indicator == '3EMA':
            e1 = pd.Series(talib.EMA(close, p['ema1']), index=close.index)
            e2 = pd.Series(talib.EMA(close, p['ema2']), index=close.index)
            e3 = pd.Series(talib.EMA(close, p['ema3']), index=close.index)
            return (e1.shift(1) <= e2.shift(1)) & (e1 > e2) & (e1 > e3), \
                   (e1.shift(1) >= e2.shift(1)) & (e1 < e2)

        elif indicator == 'DEMA':
            df_ = pd.Series(talib.DEMA(close, p['dema_fast']), index=close.index)
            ds  = pd.Series(talib.DEMA(close, p['dema_slow']), index=close.index)
            return (df_.shift(1) <= ds.shift(1)) & (df_ > ds), \
                   (df_.shift(1) >= ds.shift(1)) & (df_ < ds)

        elif indicator == 'TEMA':
            tf_ = pd.Series(talib.TEMA(close, p['tema_fast']), index=close.index)
            ts  = pd.Series(talib.TEMA(close, p['tema_slow']), index=close.index)
            return (tf_.shift(1) <= ts.shift(1)) & (tf_ > ts), \
                   (tf_.shift(1) >= ts.shift(1)) & (tf_ < ts)

        elif indicator == 'KAMA':
            kf = pd.Series(talib.KAMA(close, p['kama_fast']), index=close.index)
            ks = pd.Series(talib.KAMA(close, p['kama_slow']), index=close.index)
            return (kf.shift(1) <= ks.shift(1)) & (kf > ks), \
                   (kf.shift(1) >= ks.shift(1)) & (kf < ks)

        elif indicator == 'STC':
            macd_r, _, _ = talib.MACD(close, fastperiod=p['stc_fast'], slowperiod=p['stc_slow'])
            ms  = talib.EMA(talib.EMA(macd_r, timeperiod=p['stc_cycle']), timeperiod=p['stc_smooth'])
            hi  = talib.MAX(ms, timeperiod=p['stc_cycle'])
            lo  = talib.MIN(ms, timeperiod=p['stc_cycle'])
            stc = 100 * ((ms - lo) / (hi - lo))
            sl  = pd.Series(talib.SMA(stc, timeperiod=3), index=close.index)
            return (sl.shift(1) <= 50) & (sl > 50), (sl.shift(1) >= 50) & (sl < 50)

        elif indicator == 'BBANDS':
            u, m, _ = talib.BBANDS(close, timeperiod=p['bb_len'],
                                    nbdevup=p['bb_std'], nbdevdn=p['bb_std'], matype=0)
            us = pd.Series(u, index=close.index)
            ms = pd.Series(m, index=close.index)
            return (close.shift(1) <= us.shift(1)) & (close > us), \
                   (close.shift(1) >= ms.shift(1)) & (close < ms)

        elif indicator == 'KELTNER':
            ema = pd.Series(talib.EMA(close, p['kc_len']), index=close.index)
            atr = pd.Series(talib.ATR(high, low, close, p['kc_len']), index=close.index)
            upper = ema + p['kc_mult'] * atr
            return (close.shift(1) <= upper.shift(1)) & (close > upper), \
                   (close.shift(1) >= ema.shift(1)) & (close < ema)

        elif indicator == 'DONCHIAN':
            highest  = high.rolling(p['don_len']).max()
            exit_low = low.rolling(p['exit_len']).min()
            return close > highest.shift(1), close < exit_low.shift(1)

        elif indicator == 'CHANDELIER':
            atr    = pd.Series(talib.ATR(high, low, close, p['chand_len']), index=close.index)
            ema_tr = pd.Series(talib.EMA(close, p.get('trend_ema', 200)), index=close.index)
            l_stop = high.rolling(p['chand_len']).max() - atr * p['chand_mult']
            s_stop = low.rolling(p['chand_len']).min()  + atr * p['chand_mult']
            in_up  = close > ema_tr
            return (close.shift(1) <= l_stop.shift(1)) & (close > l_stop) & in_up, \
                   (close.shift(1) >= s_stop.shift(1)) & (close < s_stop)

        elif indicator == 'ADX':
            adx    = pd.Series(talib.ADX(high, low, close, p['adx_len']), index=close.index)
            pdi    = pd.Series(talib.PLUS_DI(high, low, close, p['adx_len']), index=close.index)
            mdi    = pd.Series(talib.MINUS_DI(high, low, close, p['adx_len']), index=close.index)
            ema_ex = pd.Series(talib.EMA(close, p['adx_exit']), index=close.index)
            return (pdi > mdi) & (adx > p['adx_thresh']), (mdi > pdi) | (close < ema_ex)

        elif indicator == 'SUPERTREND':
            h = high.values.astype(np.float64)
            l = low.values.astype(np.float64)
            c = close.values.astype(np.float64)
            atr_v = talib.ATR(h, l, c, timeperiod=p['st_len'])
            hl2   = (h + l) / 2
            ub    = hl2 + p['st_mult'] * atr_v
            lb    = hl2 - p['st_mult'] * atr_v
            n     = len(c)
            fub   = np.full(n, np.nan)
            flb   = np.full(n, np.nan)
            dire  = np.ones(n, dtype=np.int8)
            st    = p['st_len']
            fub[st] = ub[st]; flb[st] = lb[st]
            for i in range(st+1, n):
                fub[i] = ub[i] if c[i-1] > fub[i-1] else (min(ub[i], fub[i-1]) if not np.isnan(fub[i-1]) else ub[i])
                flb[i] = lb[i] if c[i-1] < flb[i-1] else (max(lb[i], flb[i-1]) if not np.isnan(flb[i-1]) else lb[i])
                if   c[i] > fub[i]: dire[i] = 1
                elif c[i] < flb[i]: dire[i] = -1
                else:                dire[i] = dire[i-1]
            dir_s = pd.Series(dire, index=close.index)
            cp    = p.get('confirm_p', 1)
            cl, cs = True, True
            for lag in range(1, cp+1):
                cl = cl & (dir_s.shift(lag) == 1)
                cs = cs & (dir_s.shift(lag) == -1)
            return (dir_s.shift(cp+1) == -1) & cl, (dir_s.shift(cp+1) == 1) & cs

        elif indicator == 'RSI':
            rsi = pd.Series(talib.RSI(close, p['rsi_len']), index=close.index)
            return (rsi.shift(1) <= p['oversold']) & (rsi > p['oversold']), \
                   (rsi.shift(1) >= p['overbought']) & (rsi < p['overbought'])

        elif indicator == 'STOCH':
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=p['stoch_k'],
                                        slowk_period=p['stoch_d'], slowd_period=p['stoch_d'])
            k_s = pd.Series(slowk, index=close.index)
            d_s = pd.Series(slowd, index=close.index)
            return (k_s.shift(1) <= d_s.shift(1)) & (k_s > d_s) & (k_s < p['stoch_os']), \
                   (k_s.shift(1) >= d_s.shift(1)) & (k_s < d_s) & (k_s > p['stoch_ob'])

        elif indicator == 'SAR':
            sar = pd.Series(talib.SAR(high, low, acceleration=p['sar_step'], maximum=p['sar_max']), index=close.index)
            return (close.shift(1) <= sar.shift(1)) & (close > sar), \
                   (close.shift(1) >= sar.shift(1)) & (close < sar)

        elif indicator == 'CCI':
            cci = pd.Series(talib.CCI(high, low, close, p['cci_len']), index=close.index)
            return (cci.shift(1) <= -p['cci_level']) & (cci > -p['cci_level']), \
                   (cci.shift(1) >= p['cci_level']) & (cci < p['cci_level'])

        elif indicator == 'WILLR':
            willr = pd.Series(talib.WILLR(high, low, close, p['will_len']), index=close.index)
            return (willr.shift(1) <= p['will_os']) & (willr > p['will_os']), \
                   (willr.shift(1) >= p['will_ob']) & (willr < p['will_ob'])

        elif indicator == 'AROON':
            d, u = talib.AROON(high, low, timeperiod=p['aroon_len'])
            u_s, d_s = pd.Series(u, index=close.index), pd.Series(d, index=close.index)
            return (u_s.shift(1) <= d_s.shift(1)) & (u_s > d_s), \
                   (u_s.shift(1) >= d_s.shift(1)) & (u_s < d_s)

        # ============= INDICADORES DEFENSIVOS (5) =============

        elif indicator == 'GOLDEN_CROSS':
            f = pd.Series(talib.SMA(close, p['ma_fast']), index=close.index)
            s = pd.Series(talib.SMA(close, p['ma_slow']), index=close.index)
            return (f.shift(1) <= s.shift(1)) & (f > s), \
                   (f.shift(1) >= s.shift(1)) & (f < s)

        elif indicator == 'EMA_LARGO':
            f = pd.Series(talib.EMA(close, p['ema_fast']), index=close.index)
            s = pd.Series(talib.EMA(close, p['ema_slow']), index=close.index)
            return (f.shift(1) <= s.shift(1)) & (f > s), \
                   (f.shift(1) >= s.shift(1)) & (f < s)

        elif indicator == 'DONCHIAN_LARGO':
            highest  = high.rolling(p['don_len']).max()
            exit_low = low.rolling(p['exit_len']).min()
            return close > highest.shift(1), close < exit_low.shift(1)

        elif indicator == 'ADX_LARGO':
            adx    = pd.Series(talib.ADX(high, low, close, p['adx_len']), index=close.index)
            pdi    = pd.Series(talib.PLUS_DI(high, low, close, p['adx_len']), index=close.index)
            mdi    = pd.Series(talib.MINUS_DI(high, low, close, p['adx_len']), index=close.index)
            ema_ex = pd.Series(talib.EMA(close, p['adx_exit']), index=close.index)
            return (pdi > mdi) & (adx > p['adx_thresh']), (mdi > pdi) | (close < ema_ex)

        elif indicator == 'KAMA_LARGO':
            kf = pd.Series(talib.KAMA(close, p['kama_fast']), index=close.index)
            ks = pd.Series(talib.KAMA(close, p['kama_slow']), index=close.index)
            return (kf.shift(1) <= ks.shift(1)) & (kf > ks), \
                   (kf.shift(1) >= ks.shift(1)) & (kf < ks)

        # ============= INDICADORES MR (mean-reversion, sufijo _SHORT = lado short) =============

        elif indicator in ('RSI2_MR', 'RSI2_MR_SHORT'):
            rsi = pd.Series(talib.RSI(close.values, p['rsi_period']), index=close.index)
            if indicator.endswith('_SHORT'):
                return rsi.shift(1) > p['umbral_high'], rsi.shift(1) < (100 - p['exit_mid'])
            return rsi.shift(1) < p['umbral_low'], rsi.shift(1) > p['exit_mid']

        elif indicator in ('RSI_MR', 'RSI_MR_SHORT'):
            rsi = pd.Series(talib.RSI(close.values, p['rsi_period']), index=close.index)
            if indicator.endswith('_SHORT'):
                return rsi.shift(1) > p['umbral_high'], rsi.shift(1) < p['umbral_low']
            return rsi.shift(1) < p['umbral_low'], rsi.shift(1) > p['umbral_high']

        elif indicator in ('CCI_MR', 'CCI_MR_SHORT'):
            cci = pd.Series(talib.CCI(high.values, low.values, close.values, p['cci_period']),
                            index=close.index)
            u, c = p['umbral'], p['confirmacion']
            if indicator.endswith('_SHORT'):
                return (cci.shift(c+1) > u) & (cci.shift(1) < u), cci.shift(1) < 0
            return (cci.shift(c+1) < -u) & (cci.shift(1) > -u), cci.shift(1) > 0

        elif indicator in ('WILLR_MR', 'WILLR_MR_SHORT'):
            willr = pd.Series(talib.WILLR(high.values, low.values, close.values, p['willr_period']),
                              index=close.index)
            if indicator.endswith('_SHORT'):
                return willr.shift(1) > p['umbral_high'], willr.shift(1) < p['umbral_low']
            return willr.shift(1) < p['umbral_low'], willr.shift(1) > p['umbral_high']

        elif indicator in ('BBANDS_REV', 'BBANDS_REV_SHORT'):
            upper, mid, lower = talib.BBANDS(close.values, timeperiod=p['bb_len'],
                                             nbdevup=p['bb_std'], nbdevdn=p['bb_std'])
            upper = pd.Series(upper, index=close.index)
            mid   = pd.Series(mid,   index=close.index)
            lower = pd.Series(lower, index=close.index)
            if indicator.endswith('_SHORT'):
                exit_s = (close.shift(1) <= mid.shift(1)) if p['exit_at_mid'] else (close.shift(1) <= lower.shift(1))
                return close.shift(1) >= upper.shift(1), exit_s
            exit_l = (close.shift(1) >= mid.shift(1)) if p['exit_at_mid'] else (close.shift(1) >= upper.shift(1))
            return close.shift(1) <= lower.shift(1), exit_l

        elif indicator in ('ZSCORE_MR', 'ZSCORE_MR_SHORT'):
            lb = p['lookback']
            roll_mean = close.rolling(lb).mean()
            roll_std  = close.rolling(lb).std()
            z = (close - roll_mean) / roll_std.replace(0, np.nan)
            if indicator.endswith('_SHORT'):
                return z.shift(1) > p['z_entry'], z.shift(1) < p['z_exit']
            return z.shift(1) < -p['z_entry'], z.shift(1) > -p['z_exit']

        elif indicator in ('STOCH_MR', 'STOCH_MR_SHORT'):
            slowk, slowd = talib.STOCH(high.values, low.values, close.values,
                                        fastk_period=p['fastk'], slowk_period=p['slowk'],
                                        slowd_period=p['slowd'])
            sk = pd.Series(slowk, index=close.index)
            sd = pd.Series(slowd, index=close.index)
            u = p['umbral']
            if indicator.endswith('_SHORT'):
                return (sk.shift(2) > 100-u) & (sd.shift(2) > 100-u) & (sk.shift(1) < sd.shift(1)), \
                       sk.shift(1) < u
            return (sk.shift(2) < u) & (sd.shift(2) < u) & (sk.shift(1) > sd.shift(1)), \
                   sk.shift(1) > (100 - u)

        elif indicator in ('ESTACIONAL_MENSUAL', 'ESTACIONAL_MENSUAL_SHORT'):
            mes = close.index.month
            meses = [m for m in range(1, 13) if p.get(f'mes_{m:02d}') ==
                     ('SHORT' if indicator.endswith('_SHORT') else 'LONG')]
            en_mes = pd.Series(pd.Index(mes).isin(meses), index=close.index)
            entries = en_mes & ~en_mes.shift(1).fillna(False)
            exits   = ~en_mes
            return entries, exits

        elif indicator in ('DOW_EFFECT', 'DOW_EFFECT_SHORT'):
            nombres = {0: 'Lun', 1: 'Mar', 2: 'Mie', 3: 'Jue', 4: 'Vie'}
            dow = close.index.dayofweek
            dias = [d for d in range(5) if p.get(f'dia_{nombres[d]}') ==
                    ('SHORT' if indicator.endswith('_SHORT') else 'LONG')]
            en_dia = pd.Series(pd.Index(dow).isin(dias), index=close.index)
            entries = en_dia & ~en_dia.shift(1).fillna(False)
            exits   = ~en_dia
            return entries, exits

        elif indicator == 'COMBO_ESTAC_RSI':
            rsi = pd.Series(talib.RSI(close.values, p['rsi_period']), index=close.index)
            mes = close.index.month
            meses_activos = [m for m in range(1, 13) if p.get(f'mes_{m:02d}') == 'SI']
            en_mes_valido = pd.Series(pd.Index(mes).isin(meses_activos), index=close.index)
            entries = en_mes_valido & (rsi.shift(1) < p['rsi_umbral'])
            exits   = ~en_mes_valido | (rsi.shift(1) > 60)
            return entries, exits

        else:
            return None, None

    except Exception:
        return None, None


# ==============================================================================
# 13. RE-SELECCIÓN SIN RE-MINAR  (fusión 2026-07 — antes re_seleccionar.py +
#     completar_long_files.py)
# ==============================================================================
# Re-selecciona el top-3 por ticker desde los CSVs crudos del minado aplicando
# los fixes de la auditoría (#5 umbral por sleeve, #6 dedup por familia), y
# regenera COMPLETOS los long files (oos_returns_long, equity_curves_long)
# re-simulando las estrategias seleccionadas. Precios via precios_cache.py
# (cache local en disco — no depende de que internet aguante toda la corrida).
#
# Uso:  py Quant_Miner.py reseleccionar          (completo)
#       py Quant_Miner.py reseleccionar --dry    (solo filtros, no escribe)

RESEL_UMBRAL_SLEEVE = {
    'DIVERSIFIERS':       0.30,   # bonos/oro/MF/FX ETFs — Sharpe histórico modesto
    'DEFENSIVE_QUALITY':  0.30,   # (sleeve excluido del portafolio, por si acaso)
    'CRYPTO':             0.30,
    'CYCLICAL_COMMODITY': 0.30,
    'GROWTH_BETA':        0.40,   # equities de crecimiento — pool grande y rendidor
}
RESEL_UMBRAL_DEFAULT = 0.35
RESEL_PRE_CORTE      = 50        # top-N por Sharpe IS antes de re-simular


def _resel_cargar_pool():
    import glob
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, 'estrategias_*.csv')))
    print(f"Encontrados {len(files)} CSVs crudos en {OUTPUT_DIR}/")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                dfs.append(df)
        except Exception as e:
            print(f"  ⚠  {os.path.basename(f)}: no pude leer ({e})")
    df = pd.concat(dfs, ignore_index=True)
    print(f"Pool crudo: {len(df):,} estrategias sobre {df['ticker'].nunique()} tickers")
    return df


def _resel_prefiltrar(df):
    """FIX #5: umbral de Sharpe IS adaptado por SLEEVE (no por asset_group —
    DIVERSIFIERS queda mezclado dentro de STOCKS_ETF y un umbral uniforme
    castiga a los defensivos y regala a los de growth)."""
    antes = len(df)
    df = df.copy()
    df['sleeve'] = df['ticker'].map(lambda t: TICKER_SLEEVE.get(t, 'UNASSIGNED'))
    umbral = df['sleeve'].map(lambda s: RESEL_UMBRAL_SLEEVE.get(s, RESEL_UMBRAL_DEFAULT))
    df = df[df['is_sharpe'] >= umbral]
    df = df[(df['is_dd']              <= MAX_IS_DD)
          & (df['is_profit_factor']   >= MIN_IS_PF)
          & (df['sensitivity_drop']   <= MAX_SENS_DROP)
          & (df['is_trades_per_year'] >= MIN_TRADES_PER_YEAR)
          & (df['oos_trades_per_year']>= MIN_TRADES_PER_YEAR)]
    print(f"Pre-filtro (Sharpe-sleeve + otros gates IS): {antes:,} -> {len(df):,} estrategias")
    for s in sorted(RESEL_UMBRAL_SLEEVE):
        n = int((df['sleeve'] == s).sum())
        print(f"    {s:20s} umbral IS>={RESEL_UMBRAL_SLEEVE[s]:.2f}  sobrevivientes {n:6d}")
    df = (df.sort_values('is_sharpe', ascending=False)
             .groupby('ticker', group_keys=False)
             .head(RESEL_PRE_CORTE))
    print(f"  Pre-corte top-{RESEL_PRE_CORTE} por ticker: quedan {len(df):,} estrategias "
          f"({df['ticker'].nunique()} tickers)")
    return df


def _resel_resimular_is(df):
    """Re-simula cada estrategia sobreviviente para obtener sus retornos IS
    (necesarios para el dedup por correlación). Precios desde cache local."""
    from precios_cache import obtener_precios

    tickers = sorted(df['ticker'].unique())
    print(f"\nRe-simulando IS: {len(df)} estrategias / {len(tickers)} tickers")
    is_rets, sin_precio = {}, []
    t0 = time.time()

    for i, ticker in enumerate(tickers, 1):
        grupo = df[df['ticker'] == ticker]['asset_group'].iloc[0]
        start = GROUP_START.get(grupo, '2010-01-01')
        fees  = GROUP_FEES.get(grupo, 0.0005)
        try:
            data = obtener_precios(ticker, start)
            if data is None or len(data) < 300:
                sin_precio.append(ticker)
                continue
            close = data['Close'].squeeze()
            high, low = data['High'].squeeze(), data['Low'].squeeze()
            split_idx = int(len(close) * SPLIT)
            for idx, row in df[df['ticker'] == ticker].iterrows():
                try:
                    params = json.loads(row['params']) if isinstance(row['params'], str) else row['params']
                    entries, exits = reconstruir_senales(row['indicator'], params, close, high, low)
                    if entries is None:
                        continue
                    pf = vbt.Portfolio.from_signals(close, entries.fillna(False),
                                                    exits.fillna(False),
                                                    init_cash=10000, fees=fees, freq='1d')
                    r = pf.returns().fillna(0.0).iloc[:split_idx]
                    is_rets[idx] = [round(float(x), 6) for x in r.tolist()]
                except Exception:
                    continue
        except Exception as e:
            print(f"  ⚠  {ticker}: {e}")
            continue
        if i % 20 == 0 or i == len(tickers):
            el = time.time() - t0
            rest = el / i * (len(tickers) - i)
            print(f"  [{i}/{len(tickers)}] {ticker:12s} ({el:.0f}s transcurridos, "
                  f"~{rest:.0f}s restantes)")

    df = df.copy()
    df['is_returns'] = df.index.map(lambda i: is_rets.get(i))
    df = df.dropna(subset=['is_returns'])
    df['is_returns'] = df['is_returns'].apply(json.dumps)
    print(f"Re-simulación completa: {len(df):,} estrategias con retornos IS")
    if sin_precio:
        print(f"⚠  {len(sin_precio)} tickers sin precio (cache vacío + descarga falló): "
              f"{sin_precio}")
        print("   Corré 'py precios_cache.py precargar' y reintentá.")
    return df


def _resel_seleccionar_top(df):
    """FIX #6: dedup por familia de indicador además del dedup por correlación
    IS. Score con pesos documentados (#8)."""
    mc = df['is_monthly_consistency'].fillna(0.5)
    tr = df['is_tail_ratio'].fillna(1.0).clip(upper=3.0)
    df = df.copy()
    df['selection_score'] = (
        df['is_sharpe']                          * 0.45 +
        df['is_profit_factor'].clip(upper=3.0)   * 0.12 +
        mc                                        * 0.15 +
        (tr / 3.0)                                * 0.08 -
        df['sensitivity_drop']                    * 0.10 -
        df['is_dd']                               * 0.10
    )
    out = []
    for ticker, g in df.groupby('ticker'):
        g = g.sort_values('selection_score', ascending=False)
        selected = []
        for _, row in g.iterrows():
            if len(selected) >= TOP_N:
                break
            fam_curr = FAMILIA_INDICADOR.get(row['indicator'], 'OTRA')
            n_misma = sum(FAMILIA_INDICADOR.get(s['indicator'], 'OTRA') == fam_curr
                           for s in selected)
            if len(selected) >= 2 and n_misma >= 2:
                continue
            try:
                r_curr = pd.Series(json.loads(row['is_returns']))
            except Exception:
                continue
            too_similar = False
            for s in selected:
                try:
                    r_prev = pd.Series(json.loads(s['is_returns']))
                    n = min(len(r_curr), len(r_prev))
                    if n < 20:
                        continue
                    corr = r_curr.iloc[-n:].corr(r_prev.iloc[-n:])
                    if not np.isnan(corr) and corr > CORR_MAX:
                        too_similar = True
                        break
                except Exception:
                    pass
            if not too_similar:
                selected.append(row)
        out.extend(selected)
    top = pd.DataFrame(out)
    print(f"\nTop-3 nuevo: {len(top)} estrategias en {top['ticker'].nunique()} tickers")
    return top


def _resel_regenerar_outputs(top):
    """Escribe top3_global.csv y regenera COMPLETOS los long files re-simulando
    las estrategias seleccionadas sobre toda su historia (segmentos IS+OOS).

    FIX 2026-07: la versión anterior (re_seleccionar.py) solo FILTRABA los
    long files existentes por intersección de strategy_id — como la
    re-selección genera ids nuevos, la intersección colapsaba a ~35 tickers y
    el Assembler salteaba el resto en silencio. Regenerar de verdad garantiza
    cobertura 1:1 con top3_global.csv."""
    from precios_cache import obtener_precios

    top = top.copy()
    top['strategy_id'] = top.apply(lambda r: build_strategy_id(
        r['ticker'], r['indicator'], r['params']), axis=1)
    top['sleeve'] = top['ticker'].map(lambda t: TICKER_SLEEVE.get(t, 'UNASSIGNED'))
    top['sector'] = top['ticker'].map(lambda t: TICKER_SECTOR.get(t, 'UNKNOWN'))

    if os.path.exists(OUT_TOP3):
        os.replace(OUT_TOP3, OUT_TOP3.replace('.csv', '_bak.csv'))
        print(f"  (backup: {OUT_TOP3} -> {OUT_TOP3.replace('.csv', '_bak.csv')})")
    top.drop(columns=['is_returns'], errors='ignore').to_csv(OUT_TOP3, index=False)
    print(f"✅ {OUT_TOP3} ({len(top)} estrategias)")

    print(f"\nRegenerando long files ({top['ticker'].nunique()} tickers)...")
    buf_ret, buf_eq, sin_precio = [], [], []
    t0 = time.time()
    tickers = sorted(top['ticker'].unique())
    for i, ticker in enumerate(tickers, 1):
        g = top[top['ticker'] == ticker]
        grupo = g['asset_group'].iloc[0]
        start = GROUP_START.get(grupo, '2010-01-01')
        fees  = GROUP_FEES.get(grupo, 0.0005)
        data = obtener_precios(ticker, start)
        if data is None or len(data) < 300:
            sin_precio.append(ticker)
            continue
        close = data['Close'].squeeze()
        high, low = data['High'].squeeze(), data['Low'].squeeze()
        split_idx = int(len(close) * SPLIT)
        fechas = close.index
        for _, row in g.iterrows():
            try:
                params = json.loads(row['params']) if isinstance(row['params'], str) else row['params']
                entries, exits = reconstruir_senales(row['indicator'], params, close, high, low)
                if entries is None:
                    continue
                pf = vbt.Portfolio.from_signals(close, entries.fillna(False), exits.fillna(False),
                                                init_cash=10000, fees=fees, freq='1d')
                equity = pf.value()
                rets   = pf.returns().fillna(0.0)
                sid = row['strategy_id']
                for j, fecha in enumerate(fechas):
                    buf_eq.append({'strategy_id': sid, 'fecha': fecha.strftime('%Y-%m-%d'),
                                   'equity': round(float(equity.iloc[j]), 2),
                                   'segmento': 'IS' if j < split_idx else 'OOS'})
                for j in range(split_idx, len(fechas)):
                    buf_ret.append({'strategy_id': sid, 'fecha': fechas[j].strftime('%Y-%m-%d'),
                                    'retorno_oos': round(float(rets.iloc[j]), 6)})
            except Exception:
                continue
        if i % 40 == 0 or i == len(tickers):
            el = time.time() - t0
            print(f"  [{i}/{len(tickers)}] {ticker:12s} ({el:.0f}s)")

    if sin_precio:
        print(f"⚠  {len(sin_precio)} tickers sin precio: {sin_precio}")

    for base, buf in ((OUT_RETURNS, buf_ret), (OUT_EQUITY, buf_eq)):
        df_b = pd.DataFrame(buf)
        for ext in ('parquet', 'csv'):
            path = f"{base}.{ext}"
            if os.path.exists(path):
                bak = f"{base}_bak.{ext}"
                if not os.path.exists(bak):
                    os.replace(path, bak)
                    print(f"  (backup: {path} -> {bak})")
                break
        try:
            df_b.to_parquet(f"{base}.parquet", index=False)
            print(f"✅ {base}.parquet: {len(df_b):,} filas")
        except Exception:
            df_b.to_csv(f"{base}.csv", index=False)
            print(f"✅ {base}.csv: {len(df_b):,} filas")


def main_reseleccionar(dry=False):
    print("=" * 70)
    print("RE-SELECCIÓN SIN RE-MINAR — fixes #5 (Sharpe por sleeve) + #6 (familia)")
    print("=" * 70)
    df_pool = _resel_cargar_pool()
    df = _resel_prefiltrar(df_pool)
    if dry:
        print("\n(--dry: no re-simulo ni escribo. Acá terminaría.)")
        return
    df = _resel_resimular_is(df)
    top = _resel_seleccionar_top(df)
    _resel_regenerar_outputs(top)
    print("\n✅ Re-selección completa. Siguientes pasos:")
    print("     py Strategy_Assembler.py")
    print("     py Portfolio_Balancer.py")
    print("     py validation.py")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'reseleccionar':
        main_reseleccionar(dry='--dry' in sys.argv)
    else:
        main()
