# ==============================================================================
# 1) KÜTÜPHANELER
# ==============================================================================


import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Callable
import json, os
from datetime import datetime
from joblib import Parallel, delayed
import openpyxl

# ==============================================================================
# 2) VERİ MODELLERİ
# ==============================================================================
@dataclass
class TradeDetail:
    ticker: str
    combination: str
    direction: str   # 'bullish' | 'bearish'
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    pnl_pct: float
    hit_tp50: bool
    max_runup_pct: float
    max_drawdown_pct: float
    # Ek alanlar (raporlama)
    ulke: str = ""
    borsa_kategori: str = ""
    zaman_dilimi: str = ""
    sl_yuzde: float = 0.0
    tp_yuzde: float = 0.0
    tp_kismi_yuzde: int = 0
    hacim_lb: int = 0
    hacim_artis_yuzde: int = 0
    sektor: str = ""
    trend: str = ""         # Up/Down/Side
    atr_yuzde: float = 0.0  # ATR(14)/Close
    risk_odul: float = 0.0  # TP/SL

# ==============================================================================
# 3) YARDIMCILAR: kolon normalizasyonu + yeniden örnekleme + indirme + timestamp
# ==============================================================================
def normalize_ohlcv_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    wanted = {"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}
    if isinstance(df.columns, pd.MultiIndex):
        levels0 = set(df.columns.get_level_values(0))
        levels1 = set(df.columns.get_level_values(1))
        if ticker in levels1:
            sub = df.xs(ticker, axis=1, level=1, drop_level=True)
        elif ticker in levels0:
            sub = df.xs(ticker, axis=1, level=0, drop_level=True)
        else:
            sub = df.copy(); sub.columns = [str(a) for a, *_ in sub.columns.to_list()]
    else:
        sub = df
    lower_map = {c.lower(): c for c in sub.columns}
    out = pd.DataFrame(index=sub.index)
    for k, proper in {"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}.items():
        if k in lower_map: out[proper] = sub[lower_map[k]]
        elif k == "close" and "adj close" in lower_map: out["Close"] = sub[lower_map["adj close"]]
        else: out[proper] = pd.NA
    if out[["Open","High","Low","Close","Volume"]].isna().all().all(): return df
    return out

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = df["Open"].resample(rule).first()
    h = df["High"].resample(rule).max()
    l = df["Low"].resample(rule).min()
    c = df["Close"].resample(rule).last()
    v = df["Volume"].resample(rule).sum()
    out = pd.concat([o,h,l,c,v], axis=1); out.columns = ["Open","High","Low","Close","Volume"]
    return out.dropna(how="any")

# Türkçe etiket -> (yfinance taban aralık, resample kuralı)
INTERVAL_MAP: Dict[str, Tuple[str, str | None]] = {
    "1 Dakika": ("1m", None),
    "3 Dakika": ("1m", "3T"),
    "5 Dakika": ("5m", None),
    "15 Dakika": ("15m", None),
    "30 Dakika": ("30m", None),
    "45 Dakika": ("15m", "45T"),
    "1 Saat": ("60m", None),
    "2 Saat": ("60m", "2H"),
    "3 Saat": ("60m", "3H"),
    "4 Saat": ("60m", "4H"),
    "1 Gün": ("1d", None),
    "1 Hafta": ("1wk", None),
    "1 Ay": ("1mo", None),
}

def download_with_label(ticker: str, period: str, label: str) -> pd.DataFrame:
    yf_interval, resample_rule = INTERVAL_MAP[label]
    df = yf.download(ticker, period=period, interval=yf_interval, auto_adjust=True, progress=False)
    if df is None or df.empty: return df
    df = normalize_ohlcv_columns(df, ticker)
    if resample_rule: df = resample_ohlcv(df, resample_rule)
    return df

def ts_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# Basit sektör alma: yfinance → info; olmazsa fallback sözlük
FALLBACK_SECTOR = {
    "THYAO.IS": "Ulaşım", "TUPRS.IS": "Enerji", "AAPL": "Bilgi Teknolojileri", "MSFT": "Bilgi Teknolojileri",
    "AMZN": "Tüketici", "GOOG": "İletişim", "NVDA": "Bilgi Teknolojileri", "JPM": "Finans",
}
def get_sector_safe(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).get_info()
        sec = info.get("sector") or info.get("industry") or ""
        if sec: return str(sec)
    except Exception:
        pass
    return FALLBACK_SECTOR.get(ticker, "")

def trend_and_atr(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    # Dow/PA için kaba trend etiketi + ATR%
    c = df["Close"]
    ma50 = c.rolling(50).mean()
    ma200 = c.rolling(200).mean()
    trend = pd.Series("Side", index=c.index)
    trend[(ma50 > ma200)] = "Up"
    trend[(ma50 < ma200)] = "Down"
    # ATR(14)
    h,l,o,c = df["High"], df["Low"], df["Open"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    atr_pct = (atr / c).fillna(0.0)
    return trend.fillna("Side"), atr_pct

def infer_country_from_suffix(ticker: str) -> str:
    if ticker.endswith(".IS"): return "Türkiye"
    if ticker.endswith(".DE"): return "Almanya"
    if ticker.endswith(".PA"): return "Fransa"
    if ticker.endswith(".L"):  return "Birleşik Krallık"
    if ticker.endswith(".AS"): return "Hollanda"
    if ticker.endswith(".SW"): return "İsviçre"
    if ticker.endswith(".MC"): return "İspanya"
    if ticker.endswith(".MI"): return "İtalya"
    if ticker.endswith(".TO"): return "Kanada"
    if ticker.endswith(".HK"): return "Hong Kong"
    if ticker.endswith(".T"):  return "Japonya"
    if ticker.endswith(".KS"): return "Güney Kore"
    if ticker.endswith(".SS") or ticker.endswith(".SZ"): return "Çin"
    if ticker.endswith(".AX"): return "Avustralya"
    return "Amerika"  # US default

# ==============================================================================
# 4) FORMASYONLAR (İngilizce adlar) + YÖN
# ==============================================================================
def bullish_engulfing(df):
    o,c = df["Open"], df["Close"]
    return ((c.shift(1) < o.shift(1)) & (c > o) & (c >= o.shift(1)) & (o <= c.shift(1))).fillna(False)
def bearish_engulfing(df):
    o,c = df["Open"], df["Close"]
    return ((c.shift(1) > o.shift(1)) & (c < o) & (c <= o.shift(1)) & (o >= c.shift(1))).fillna(False)
def hammer(df):
    o,h,l,c = df["Open"], df["High"], df["Low"], df["Close"]
    body=(c-o).abs(); rng=h-l; lower=np.minimum(c,o)-l; upper=h-np.maximum(c,o)
    valid=rng>1e-6; res=pd.Series(False,index=df.index)
    res.loc[valid]=(body[valid]/rng[valid]<=0.3)&(lower[valid]>=body[valid]*2.0)&(upper[valid]<=body[valid])
    return res.fillna(False)
def inverted_hammer(df):
    o,h,l,c = df["Open"], df["High"], df["Low"], df["Close"]
    body=(c-o).abs(); rng=h-l; upper=h-np.maximum(c,o); lower=np.minimum(c,o)-l
    valid=rng>1e-6; res=pd.Series(False,index=df.index)
    res.loc[valid]=(body[valid]/rng[valid]<=0.3)&(upper[valid]>=body[valid]*2.0)&(lower[valid]<=body[valid])
    return res.fillna(False)
def piercing_pattern(df):
    o,c=df["Open"], df["Close"]; prev_red=c.shift(1)<o.shift(1); green=c>o
    prev_mid=(o.shift(1)+c.shift(1))/2
    return (prev_red & green & (c>=prev_mid) & (o < c.shift(1))).fillna(False)
def dark_cloud_cover(df):
    o,c=df["Open"], df["Close"]; prev_green=c.shift(1)>o.shift(1); red=c<o
    prev_mid=(o.shift(1)+c.shift(1))/2
    return (prev_green & red & (c<=prev_mid) & (o > c.shift(1))).fillna(False)
def morning_star(df):
    o,c=df["Open"], df["Close"]
    red1=c.shift(2)<o.shift(2)
    small2=(c.shift(1)-o.shift(1)).abs()<((c.shift(2)-o.shift(2)).abs()*0.6)
    green3=c>o; retrace=c>=(o.shift(2)+c.shift(2))/2; gap=o.shift(1)<c.shift(2)
    return (red1 & small2 & green3 & retrace & gap).fillna(False)
def evening_star(df):
    o,c=df["Open"], df["Close"]
    green1=c.shift(2)>o.shift(2)
    small2=(c.shift(1)-o.shift(1)).abs()<((c.shift(2)-o.shift(2)).abs()*0.6)
    red3=c<o; retrace=c<=(o.shift(2)+c.shift(2))/2; gap=o.shift(1)>c.shift(2)
    return (green1 & small2 & red3 & retrace & gap).fillna(False)
def three_inside_up(df):
    o,c=df["Open"], df["Close"]
    harami=(c.shift(1)<o.shift(1)) & ((o>c.shift(1)) & (c<o.shift(1)))
    return (harami & (c>c.shift(1))).fillna(False)
def three_inside_down(df):
    o,c=df["Open"], df["Close"]
    harami=(c.shift(1)>o.shift(1)) & ((o<c.shift(1)) & (c>o.shift(1)))
    return (harami & (c<c.shift(1))).fillna(False)
def three_white_soldiers(df):
    o,c=df["Open"], df["Close"]
    g0=c>o; g1=c.shift(1)>o.shift(1); g2=c.shift(2)>o.shift(2)
    higher=(c>c.shift(1))&(c.shift(1)>c.shift(2))
    body_ok=((c-o)>(c-o).rolling(10).mean()*0.2)
    return (g0 & g1 & g2 & higher & body_ok).fillna(False)
def three_black_crows(df):
    o,c=df["Open"], df["Close"]
    r0=c<o; r1=c.shift(1)<o.shift(1); r2=c.shift(2)<o.shift(2)
    lower=(c<c.shift(1))&(c.shift(1)<c.shift(2))
    body_ok=((o-c)>(o-c).rolling(10).mean()*0.2)
    return (r0 & r1 & r2 & lower & body_ok).fillna(False)
def bullish_harami(df):
    o,c=df["Open"], df["Close"]
    return ((c.shift(1)<o.shift(1)) & (o>=c.shift(1)) & (c<=o.shift(1)) & (c>o)).fillna(False)
def bearish_harami(df):
    o,c=df["Open"], df["Close"]
    return ((c.shift(1)>o.shift(1)) & (o<=c.shift(1)) & (c>=o.shift(1)) & (c<o)).fillna(False)
def bullish_kicker(df):
    o,c=df["Open"], df["Close"]
    return ((o>c.shift(1)) & (c>o)).fillna(False)
def bearish_kicker(df):
    o,c=df["Open"], df["Close"]
    return ((o<c.shift(1)) & (c<o)).fillna(False)
def bullish_marubozu(df):
    o,h,l,c=df["Open"], df["High"], df["Low"], df["Close"]
    body=(c-o).abs(); rng=h-l; green=c>o
    return (green & (body/rng>=0.8)).fillna(False)
def bearish_marubozu(df):
    o,h,l,c=df["Open"], df["High"], df["Low"], df["Close"]
    body=(o-c).abs(); rng=h-l; red=c<o
    return (red & (body/rng>=0.8)).fillna(False)
def hammer_uptrend(df):  # Hanging Man
    cond = hammer(df); c=df["Close"]; up=c>c.rolling(5).mean(); return (cond & up).fillna(False)
def invhammer_uptrend(df):  # Shooting Star
    cond = inverted_hammer(df); c=df["Close"]; up=c>c.rolling(5).mean(); return (cond & up).fillna(False)
def bullish_sandwich(df):
    o,c=df["Open"], df["Close"]
    return ((c.shift(2)<o.shift(2)) & (c.shift(1)>o.shift(1)) & (c>o.shift(2))).fillna(False)
def bearish_sandwich(df):
    o,c=df["Open"], df["Close"]
    return ((c.shift(2)>o.shift(2)) & (c.shift(1)<o.shift(1)) & (c<o.shift(2))).fillna(False)
def abandoned_baby_bullish(df):
    o,h,l,c=df["Open"], df["High"], df["Low"], df["Close"]
    small2=(c.shift(1)-o.shift(1)).abs() <= (h.shift(1)-l.shift(1))*0.1
    return (small2 & (h.shift(1)<l.shift(2)) & (l>h.shift(1)) & (c>o)).fillna(False)
def abandoned_baby_bearish(df):
    o,h,l,c=df["Open"], df["High"], df["Low"], df["Close"]
    small2=(c.shift(1)-o.shift(1)).abs() <= (h.shift(1)-l.shift(1))*0.1
    return (small2 & (l.shift(1)>h.shift(2)) & (h<l.shift(1)) & (c<o)).fillna(False)

PATTERNS: Dict[str, Tuple[Callable[[pd.DataFrame], pd.Series], str]] = {
    # Bullish
    "Bullish Engulfing": (bullish_engulfing, "bullish"),
    "Bullish Harami": (bullish_harami, "bullish"),
    "Hammer": (hammer, "bullish"),
    "Inverted Hammer": (inverted_hammer, "bullish"),
    "Piercing Pattern": (piercing_pattern, "bullish"),
    "Morning Star": (morning_star, "bullish"),
    "Three Inside Up": (three_inside_up, "bullish"),
    "Three White Soldiers": (three_white_soldiers, "bullish"),
    "Abandoned Baby (Bullish)": (abandoned_baby_bullish, "bullish"),
    "Bullish Kicker": (bullish_kicker, "bullish"),
    "Bullish Marubozu": (bullish_marubozu, "bullish"),
    "Bullish Sandwich": (bullish_sandwich, "bullish"),
    # Bearish
    "Bearish Engulfing": (bearish_engulfing, "bearish"),
    "Bearish Harami": (bearish_harami, "bearish"),
    "Shooting Star": (invhammer_uptrend, "bearish"),
    "Hanging Man": (hammer_uptrend, "bearish"),
    "Dark Cloud Cover": (dark_cloud_cover, "bearish"),
    "Evening Star": (evening_star, "bearish"),
    "Three Inside Down": (three_inside_down, "bearish"),
    "Three Black Crows": (three_black_crows, "bearish"),
    "Abandoned Baby (Bearish)": (abandoned_baby_bearish, "bearish"),
    "Bearish Kicker": (bearish_kicker, "bearish"),
    "Bearish Marubozu": (bearish_marubozu, "bearish"),
    "Bearish Sandwich": (bearish_sandwich, "bearish"),
}
BULLISH_NAMES = [k for k,(_,d) in PATTERNS.items() if d=="bullish"]
BEARISH_NAMES = [k for k,(_,d) in PATTERNS.items() if d=="bearish"]

# ==============================================================================
# 5) HACİM FİLTRESİ ve BACKTEST (long/short) — merkezi PnL
# ==============================================================================
def apply_volume_condition(df: pd.DataFrame, pct_increase: int, lookback: int) -> pd.Series:
    base = df["Volume"].shift(1).rolling(lookback).mean()
    return (df["Volume"] > (base * (1 + pct_increase / 100.0))).fillna(False)

def run_trade_simulation(
    df: pd.DataFrame,
    signal_indices: np.ndarray,
    lookahead_bars: int,
    tp_pct: float,
    sl_pct: float,
    tp_partial_pct: int,
    direction: str,
    meta: dict
) -> List[dict]:
    out: List[dict] = []
    close, high, low, dates, n = df["Close"].values, df["High"].values, df["Low"].values, df.index, len(df)
    trend_ser, atr_pct_ser = trend_and_atr(df)
    risk_odul = (tp_pct/sl_pct) if sl_pct>0 else np.nan

    for i in signal_indices:
        if i+1 >= n: continue
        entry = float(close[i]); entry_date = str(dates[i].date())
        hit_tp50 = False; max_runup=0.0; max_drawdown=0.0
        exit_price=None; exit_date=None
        end_idx = min(i + lookahead_bars, n-1)

        if direction=="bullish":
            tp_price=entry*(1+tp_pct); sl_price=entry*(1-sl_pct)
            for j in range(i+1, end_idx+1):
                max_runup=max(max_runup,(high[j]-entry)/entry)
                max_drawdown=min(max_drawdown,(low[j]-entry)/entry)
                if low[j] <= sl_price: exit_price=sl_price; exit_date=str(dates[j].date()); break
                if not hit_tp50 and high[j] >= tp_price: hit_tp50=True
        else:
            tp_price=entry*(1-tp_pct); sl_price=entry*(1+sl_pct)
            for j in range(i+1, end_idx+1):
                max_runup=max(max_runup,(entry-low[j])/entry)
                max_drawdown=min(max_drawdown,-(high[j]-entry)/entry)
                if high[j] >= sl_price: exit_price=sl_price; exit_date=str(dates[j].date()); break
                if not hit_tp50 and low[j] <= tp_price: hit_tp50=True

        if exit_price is None:
            exit_date=str(dates[end_idx].date()); final=float(close[end_idx])
            if direction=="bullish":
                if hit_tp50:
                    partial=tp_pct*(tp_partial_pct/100.0); rem=1-(tp_partial_pct/100.0)
                    pnl=partial+((final-entry)/entry)*rem; exit_price=entry*(1+pnl)
                else: exit_price=final
            else:
                if hit_tp50:
                    partial=tp_pct*(tp_partial_pct/100.0); rem=1-(tp_partial_pct/100.0)
                    pnl=partial+((entry-final)/entry)*rem; exit_price=entry*(1-pnl)
                else: exit_price=final

        pnl_pct = (exit_price/entry - 1.0) if direction=="bullish" else (entry/exit_price - 1.0)

        out.append(asdict(TradeDetail(
            ticker=meta["ticker"], combination=meta["combo"], direction=direction,
            entry_date=entry_date, entry_price=entry, exit_date=exit_date, exit_price=float(exit_price),
            pnl_pct=float(pnl_pct), hit_tp50=bool(hit_tp50), max_runup_pct=float(max_runup), max_drawdown_pct=float(max_drawdown),
            ulke=meta["ulke"], borsa_kategori=meta["borsa_kategori"], zaman_dilimi=meta["zaman_dilimi"],
            sl_yuzde=meta["sl_yuzde"], tp_yuzde=meta["tp_yuzde"], tp_kismi_yuzde=meta["tp_kismi_yuzde"],
            hacim_lb=meta["hacim_lb"], hacim_artis_yuzde=meta["hacim_artis_yuzde"],
            sektor=meta["sektor"], trend=str(trend_ser.iloc[i]) if pd.notna(trend_ser.iloc[i]) else "Side",
            atr_yuzde=float(atr_pct_ser.iloc[i]) if pd.notna(atr_pct_ser.iloc[i]) else 0.0,
            risk_odul=float(risk_odul) if risk_odul==risk_odul else 0.0
        )))
    return out

def process_ticker_dynamic(params: Tuple) -> List[dict]:
    (
        ticker, selected_patterns, interval_labels, period,
        vol_pcts, lookbacks, sl_pct, tp_pct, tp_partial_pct, lookahead_bars,
        ulke, borsa_kategori, get_sector_flag
    ) = params
    out: List[dict] = []
    sektor = get_sector_safe(ticker) if get_sector_flag else ""

    for label in interval_labels:
        try:
            df = download_with_label(ticker, period, label)
        except Exception as e:
            print(f"[UYARI] {ticker} ({label}) indirme hatası: {e}"); continue
        if df is None or df.empty:
            print(f"[BİLGİ] {ticker} ({label}) boş veri."); continue
        if not {"Open","High","Low","Close","Volume"}.issubset(df.columns):
            print(f"[UYARI] {ticker} ({label}) beklenen kolonlar eksik: {set(df.columns)}"); continue

        for pat_name, (pat_fn, direction) in selected_patterns.items():
            try:
                pat_mask = pat_fn(df).fillna(False)
            except Exception as e:
                print(f"[UYARI] {ticker} ({label}) '{pat_name}' hesaplanamadı: {e}"); continue
            if not pat_mask.any(): continue

            for vol_pct in vol_pcts:
                for lb in lookbacks:
                    try:
                        vol_mask = apply_volume_condition(df, vol_pct, lb).fillna(False)
                        sig_mask = (pat_mask & vol_mask).fillna(False)
                        idx = np.where(sig_mask.values)[0]
                        if idx.size == 0: continue
                        combo = f"{pat_name} + %{vol_pct} Hacim (LB={lb}) - {label}"
                        meta = dict(
                            ticker=ticker, combo=combo, ulke=ulke or infer_country_from_suffix(ticker),
                            borsa_kategori=borsa_kategori, zaman_dilimi=label,
                            sl_yuzde=sl_pct*100, tp_yuzde=tp_pct*100, tp_kismi_yuzde=tp_partial_pct,
                            hacim_lb=lb, hacim_artis_yuzde=vol_pct, sektor=sektor
                        )
                        logs = run_trade_simulation(df, idx, lookahead_bars, tp_pct, sl_pct, tp_partial_pct, direction, meta)
                        out.extend(logs)
                    except Exception as e:
                        print(f"[UYARI] {ticker} ({label}) '{pat_name}' vol%={vol_pct} lb={lb}: {e}")
                        continue
    return out

# ==============================================================================
# 6) SABİT STRATEJİLER (örnek, long-only)
# ==============================================================================
def strategy_yutan_boga(df): return (bullish_engulfing(df) & apply_volume_condition(df,49,4)).fillna(False)
def strategy_cebic(df): return (hammer(df) & apply_volume_condition(df,49,4)).fillna(False)
FIXED_STRATEGIES = {"Temel Strateji - Yutan Boğa": strategy_yutan_boga, "Temel Strateji - Çekiç (Hammer)": strategy_cebic}

# ==============================================================================
# 7) VARSAYILAN VERİ SETLERİ — Ülke/Borsa/Kategori
# (Temsili alt-kümeler; gerekirse istediğin kadar genişletebiliriz)
# ==============================================================================
CACHE_FILE = "borsa_hisse_cache.json"
PREDEFINED_TICKERS = {
    "Türkiye": {
        "BIST 30": ["AKBNK.IS","ARCLK.IS","ASELS.IS","BIMAS.IS","EREGL.IS","GARAN.IS","KCHOL.IS","PETKM.IS","SAHOL.IS","SASA.IS","SISE.IS","TCELL.IS","THYAO.IS","TUPRS.IS","YKBNK.IS"],
        "BIST 100 (Seçme)": ["AKBNK.IS","EREGL.IS","THYAO.IS","TUPRS.IS","ASELS.IS","BIMAS.IS","SISE.IS","FROTO.IS","KOZAL.IS","PGSUS.IS"]
    },
    "Amerika": {
        "S&P 100 (Seçme)": ["AAPL","MSFT","AMZN","GOOG","JPM","NVDA","V"],
        "Dow 30 (Seçme)": ["AAPL","MSFT","JPM","V","KO","PG","NKE"],
        "Nasdaq 100 (Seçme)": ["AAPL","MSFT","AMZN","NVDA","META","ADBE","NFLX"]
    },
    "Kanada": {
        "TSX 60 (Seçme)": ["RY.TO","TD.TO","ENB.TO","SHOP.TO","CNQ.TO","BNS.TO"]
    },
    "Birleşik Krallık": {
        "FTSE 100 (Seçme)": ["HSBA.L","BP.L","AZN.L","ULVR.L","RIO.L","GSK.L"]
    },
    "Almanya": {
        "DAX (Seçme)": ["SAP.DE","ALV.DE","BAYN.DE","BMW.DE","SIE.DE","ADS.DE","BAS.DE"]
    },
    "Fransa": {
        "CAC 40 (Seçme)": ["OR.PA","AI.PA","SAN.PA","AIR.PA","BNP.PA","ENGI.PA"]
    },
    "İspanya": {
        "IBEX 35 (Seçme)": ["ITX.MC","SAN.MC","BBVA.MC","IBE.MC","TEF.MC"]
    },
    "İtalya": {
        "FTSE MIB (Seçme)": ["ENI.MI","ISP.MI","UCG.MI","STLA.MI","ENEL.MI"]
    },
    "Hollanda": {
        "AEX (Seçme)": ["ASML.AS","HEIA.AS","AD.AS","NN.AS","RAND.AS"]
    },
    "İsviçre": {
        "SMI (Seçme)": ["NESN.SW","ROG.SW","NOVN.SW","UBSG.SW","CSGN.SW"]
    },
    "Japonya": {
        "Nikkei 225 (Seçme)": ["7203.T","9984.T","6758.T","9432.T","8306.T"]  # Toyota, SoftBank, Sony, KDDI, MUFG
    },
    "Hong Kong": {
        "Hang Seng 50 (Seçme)": ["0005.HK","0939.HK","0388.HK","0700.HK","1299.HK"]
    },
    "Güney Kore": {
        "KOSPI (Seçme)": ["005930.KS","000660.KS","035420.KS","051910.KS"]  # Samsung, SK Hynix, Naver, LG Chem
    },
    "Çin": {
        "SSE 50 (Seçme)": ["600519.SS","601318.SS","601857.SS","601988.SS"]
    },
    "Avustralya": {
        "ASX 50 (Seçme)": ["BHP.AX","CBA.AX","WBC.AX","CSL.AX","NAB.AX"]
    },
}

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE,"r",encoding="utf-8") as f: return json.load(f)
        except Exception: pass
    return PREDEFINED_TICKERS

# ==============================================================================
# 8) ARAYÜZ (Dinamik: iki kutu; Sabit: ülke/borsa/hisse+TV zamanları)
# ==============================================================================
def create_dual_listbox(options, description):
    options_widget = widgets.SelectMultiple(options=options, rows=8, layout=widgets.Layout(width='220px'))
    selected_widget = widgets.SelectMultiple(options=[], rows=8, layout=widgets.Layout(width='220px'))
    add_button = widgets.Button(description='>>'); remove_button = widgets.Button(description='<<')
    def add_items(b):
        new_opts = list(selected_widget.options) + [opt for opt in options_widget.value if opt not in selected_widget.options]
        selected_widget.options = sorted(new_opts)
    def remove_items(b):
        selected_widget.options = [opt for opt in selected_widget.options if opt not in selected_widget.value]
    add_button.on_click(add_items); remove_button.on_click(remove_items)
    buttons = widgets.VBox([add_button, remove_button], layout=widgets.Layout(align_items='center', justify_content='center'))
    return widgets.VBox([widgets.Label(value=description), widgets.HBox([options_widget, buttons, selected_widget])]), options_widget, selected_widget

all_data = load_cache()

# Dinamik
country_dlb, country_opts, country_sel = create_dual_listbox(list(all_data.keys()), "Ülkeler:")
exchange_dlb, exchange_opts, exchange_sel = create_dual_listbox([], "Borsalar/Kategoriler:")
ticker_dlb, ticker_opts, ticker_sel = create_dual_listbox([], "Hisseler:")

bullish_dlb, bullish_opts, bullish_sel = create_dual_listbox(BULLISH_NAMES, "Yükseliş Formasyonları (Bullish):")
bearish_dlb, bearish_opts, bearish_sel = create_dual_listbox(BEARISH_NAMES, "Düşüş Formasyonları (Bearish):")

style={'description_width':'initial'}
vol_lookback_widget = widgets.IntRangeSlider(value=[3,6], min=1, max=50, step=1, description='Hacim Geri Bakma (LB):', style=style)
vol_pct_widget      = widgets.IntRangeSlider(value=[0,26], min=0, max=500, step=1, description='Hacim % Artış:', style=style)
sl_widget           = widgets.FloatSlider(value=7.0, min=0.0, max=30.0, step=0.5, description='Stop-Loss %:', readout_format='.1f', style=style)
tp_widget           = widgets.FloatSlider(value=15.0, min=0.0, max=100.0, step=1, description='Take-Profit %:', readout_format='.0f', style=style)
tp_partial_widget   = widgets.IntSlider(value=50, min=0, max=100, step=10, description='Kısmi TP %:', style=style)
tp_window_widget    = widgets.IntSlider(value=5, min=1, max=50, step=1, description='TP Penceresi (Bar):', style=style)
period_widget       = widgets.Dropdown(options=['2y','5y','10y','max'], value='5y', description='Veri Periyodu:', style=style)
interval_widget     = widgets.SelectMultiple(options=list(INTERVAL_MAP.keys()), value=['1 Gün'], description='Zaman Dilimi:', style=style)
parallel_widget     = widgets.Checkbox(value=True, description='Paralel Çalıştırma')
run_dynamic_button  = widgets.Button(description="Optimizasyonu Başlat", icon='play', button_style='success')

# Sabit
fixed_strategy_dlb, _, fixed_strategy_sel = create_dual_listbox(list(FIXED_STRATEGIES.keys()), "Stratejiler:")

fixed_country_dlb, fixed_country_opts, fixed_country_sel = create_dual_listbox(list(all_data.keys()), "Ülkeler:")
fixed_exchange_dlb, fixed_exchange_opts, fixed_exchange_sel = create_dual_listbox([], "Borsalar/Kategoriler:")
fixed_ticker_dlb, fixed_ticker_opts, fixed_ticker_sel = create_dual_listbox([], "Hisseler:")
fixed_interval_widget = widgets.SelectMultiple(options=list(INTERVAL_MAP.keys()), value=['1 Gün'], description='Zaman Dilimi:', style=style)
run_fixed_button = widgets.Button(description="Sabit Stratejileri Test Et", icon='play', button_style='info')

output_area = widgets.Output()

# Layout – Dinamik
col1_dyn = widgets.VBox([country_dlb, exchange_dlb, ticker_dlb])
col2_dyn = widgets.VBox([bullish_dlb, bearish_dlb])
params1_dyn = widgets.VBox([vol_lookback_widget, vol_pct_widget, sl_widget, tp_widget])
params2_dyn = widgets.VBox([tp_partial_widget, tp_window_widget, period_widget, interval_widget])
main_box_dyn = widgets.HBox([col1_dyn, col2_dyn])
params_box_dyn = widgets.HBox([params1_dyn, params2_dyn])
tab1_content = widgets.VBox([main_box_dyn, params_box_dyn, widgets.HBox([parallel_widget, run_dynamic_button])])

# Layout – Sabit
selections_box_fixed = widgets.HBox([fixed_strategy_dlb])
fixed_row2 = widgets.HBox([fixed_country_dlb, fixed_exchange_dlb, fixed_ticker_dlb])
tab2_content = widgets.VBox([selections_box_fixed, fixed_row2, fixed_interval_widget, run_fixed_button])

tab_container = widgets.Tab(children=[tab1_content, tab2_content])
tab_container.set_title(0, 'Dinamik Optimizasyon'); tab_container.set_title(1, 'Sabit Strateji Testi')
interface_layout = widgets.VBox([widgets.HTML("<h2>Strateji Test ve Optimizasyon Sistemi</h2>"), tab_container])

# ==============================================================================
# 9) ETKİLEŞİM + Raporlama (renklendirme + timestamp dosya + üstte top listeler)
# ==============================================================================
def update_exchanges(change):
    exchanges = {e for c in country_sel.options for e in all_data.get(c, {})}
    exchange_opts.options, exchange_sel.options = sorted(list(exchanges)), []
country_sel.observe(update_exchanges, names='options')

def update_tickers(change):
    tickers = {t for c, es in all_data.items() for e, ts in es.items() if e in exchange_sel.options for t in ts}
    ticker_opts.options, ticker_sel.options = sorted(list(tickers)), []
exchange_sel.observe(update_tickers, names='options')

def update_fixed_exchanges(change):
    exchanges = {e for c in fixed_country_sel.options for e in all_data.get(c, {})}
    fixed_exchange_opts.options, fixed_exchange_sel.options = sorted(list(exchanges)), []
fixed_country_sel.observe(update_fixed_exchanges, names='options')

def update_fixed_tickers(change):
    tickers = {t for c, es in all_data.items() for e, ts in es.items() if e in fixed_exchange_sel.options for t in ts}
    fixed_ticker_opts.options, fixed_ticker_sel.options = sorted(list(tickers)), []
fixed_exchange_sel.observe(update_fixed_tickers, names='options')

def _color_direction(val):
    if isinstance(val, str):
        if val.lower()=="bullish": return "color: green; font-weight:600;"
        if val.lower()=="bearish": return "color: red; font-weight:600;"
    return ""

def build_summaries(df_trades: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # Türkçe başlıklarla özet (kombinasyon+dönem parametreleriyle)
    agg = {
        "pnl_pct":["mean","median","count"],
        "hit_tp50":["mean"],
        "max_runup_pct":["max"],
        "max_drawdown_pct":["min"],
    }
    gb_cols = ["ticker","combination","direction","ulke","borsa_kategori","zaman_dilimi",
               "sl_yuzde","tp_yuzde","tp_kismi_yuzde","hacim_lb","hacim_artis_yuzde","sektor","trend"]
    g = df_trades.groupby(gb_cols).agg(agg)
    g.columns = ["Ort. Getiri %","Medyan Getiri %","İşlem Sayısı","TP50 Oranı","Maks. Yükseliş %","Maks. Düşüş %"]
    g = g.reset_index()
    g["Ort. Getiri %"] = (g["Ort. Getiri %"]*100).round(2)
    g["Medyan Getiri %"] = (g["Medyan Getiri %"]*100).round(2)
    g["TP50 Oranı"] = (g["TP50 Oranı"]*100).round(2)
    g["Maks. Yükseliş %"] = (g["Maks. Yükseliş %"]*100).round(2)
    g["Maks. Düşüş %"] = (g["Maks. Düşüş %"]*100).round(2)

    # Kombinasyon bazlı "en çok işlem" ve "en yüksek ort. getiri"
    top_trades_combo = g.sort_values(["İşlem Sayısı","Ort. Getiri %"], ascending=[False, False]).head(20)
    top_return_combo = g[g["İşlem Sayısı"]>=3].sort_values(["Ort. Getiri %","İşlem Sayısı"], ascending=[False, False]).head(20)

    # Hisse bazlı özet
    gb_sym = ["ticker","ulke","borsa_kategori","sektor"]
    gs = df_trades.groupby(gb_sym).agg(
        İşlem_Sayısı=("pnl_pct","count"),
        Ort_Getiri_Pct=("pnl_pct","mean"),
        Medyan_Getiri_Pct=("pnl_pct","median"),
        TP50_Oranı=("hit_tp50","mean")
    ).reset_index()
    gs["Ort_Getiri_Pct"]=(gs["Ort_Getiri_Pct"]*100).round(2)
    gs["Medyan_Getiri_Pct"]=(gs["Medyan_Getiri_Pct"]*100).round(2)
    gs["TP50_Oranı"]=(gs["TP50_Oranı"]*100).round(2)

    top_trades_symbol = gs.sort_values(["İşlem_Sayısı","Ort_Getiri_Pct"], ascending=[False, False]).head(20)
    top_return_symbol = gs[gs["İşlem_Sayısı"]>=3].sort_values(["Ort_Getiri_Pct","İşlem_Sayısı"], ascending=[False, False]).head(20)

    return {
        "summary": g,
        "top_trades_combo": top_trades_combo,
        "top_return_combo": top_return_combo,
        "top_trades_symbol": top_trades_symbol,
        "top_return_symbol": top_return_symbol
    }

def show_tables(summ: Dict[str, pd.DataFrame], df_trades: pd.DataFrame):
    print("=== EN ÇOK İŞLEM (Kombinasyon) ===")
    display(summ["top_trades_combo"].style.applymap(_color_direction, subset=["direction"]))
    print("\n=== EN YÜKSEK ORTALAMA GETİRİ (Kombinasyon) ===")
    display(summ["top_return_combo"].style.applymap(_color_direction, subset=["direction"]))
    print("\n=== EN ÇOK İŞLEM (Hisse) ===")
    display(summ["top_trades_symbol"])
    print("\n=== EN YÜKSEK ORTALAMA GETİRİ (Hisse) ===")
    display(summ["top_return_symbol"])
    print("\n=== ÖZET – GENEL ===")
    display(summ["summary"].style.applymap(_color_direction, subset=["direction"]))
    print("\n=== İŞLEM GÜNLÜĞÜ (Detay) ===")
    display(df_trades.style.applymap(_color_direction, subset=["direction"]))

def process_and_display_results(all_trade_details: List[dict], stamp: str):
    if not all_trade_details:
        print("--- Backtest Tamamlandı: Hiçbir işlem bulunamadı. ---"); return
    df_trades = pd.DataFrame(all_trade_details)

    # Türkçe kolonlar için küçük dokunuşlar
    df_trades["İşlem Tarihi"] = df_trades["entry_date"]
    df_trades["Yön"] = df_trades["direction"].str.title()
    df_trades["ATR(14) %"] = (df_trades["atr_yuzde"]*100).round(2)
    df_trades["PnL %"] = (df_trades["pnl_pct"]*100).round(2)

    summaries = build_summaries(df_trades)
    show_tables(summaries, df_trades)

    fname = f"backtest_results_{stamp}.xlsx"
    try:
        with pd.ExcelWriter(fname, engine='openpyxl') as writer:
            summaries["summary"].to_excel(writer, sheet_name='Özet - Genel', index=False)
            summaries["top_trades_combo"].to_excel(writer, sheet_name='Özet - En Çok İşlem (Kombo)', index=False)
            summaries["top_return_combo"].to_excel(writer, sheet_name='Özet - En Yüksek Getiri (Kombo)', index=False)
            summaries["top_trades_symbol"].to_excel(writer, sheet_name='Özet - En Çok İşlem (Hisse)', index=False)
            summaries["top_return_symbol"].to_excel(writer, sheet_name='Özet - En Yüksek Getiri (Hisse)', index=False)
            df_trades.to_excel(writer, sheet_name='İşlem Günlüğü (Detay)', index=False)
        print(f"\n[BAŞARILI] Rapor '{fname}' olarak kaydedildi.")
    except Exception as e:
        print(f"\n[HATA] Excel kaydedilemedi: {e}")

# ==============================================================================
# 10) ÇALIŞTIRMA BUTONLARI
# ==============================================================================
def on_run_dynamic_button_click(b):
    with output_area:
        clear_output(wait=True)
        stamp = ts_stamp()
        print(f"--- Dinamik Optimizasyon Başlatılıyor ({stamp}) ---")

        selected_tickers = list(ticker_sel.options)
        sel_bull = {p: PATTERNS[p] for p in bullish_sel.options}
        sel_bear = {p: PATTERNS[p] for p in bearish_sel.options}
        selected_patterns = {**sel_bull, **sel_bear}
        interval_labels = tuple(interval_widget.value)

        if not all([selected_tickers, selected_patterns, interval_labels]):
            print("HATA: Hisse, Formasyon ve Zaman Dilimi seçimi zorunludur."); return

        lookbacks = list(range(vol_lookback_widget.value[0], vol_lookback_widget.value[1]+1))
        vol_pcts  = list(range(vol_pct_widget.value[0], vol_pct_widget.value[1]+1))
        lookahead = int(tp_window_widget.value)

        # Seçilen ülke/borsa bilgilerini aktar
        sel_countries = list(country_sel.options)
        sel_exchanges = list(exchange_sel.options)
        ulke_label = ", ".join(sel_countries) if sel_countries else ""
        borsa_label = ", ".join(sel_exchanges) if sel_exchanges else ""

        tasks = []
        for t in selected_tickers:
            for pat in selected_patterns.items():
                pass  # sadece task boyut hesap için istersen kullan
        total = len(selected_tickers)*len(interval_labels)*len(selected_patterns)*len(lookbacks)*len(vol_pcts)
        print(f"Toplam {total} kombinasyon test edilecek...")

        get_sector_flag = True  # sektörü çekmeye çalış
        param_list = [
            (t, selected_patterns, interval_labels, period_widget.value, list(vol_pcts), list(lookbacks),
             sl_widget.value/100.0, tp_widget.value/100.0, int(tp_partial_widget.value), lookahead,
             ulke_label, borsa_label, get_sector_flag)
            for t in selected_tickers
        ]

        if parallel_widget.value:
            print("Paralel çalışma modu aktif...")
            results_list = Parallel(n_jobs=-1, verbose=0)(delayed(process_ticker_dynamic)(args) for args in param_list)
        else:
            print("Sıralı çalışma modu aktif...")
            results_list = [process_ticker_dynamic(args) for args in param_list]

        all_trade_details = [x for sub in results_list for x in sub]
        process_and_display_results(all_trade_details, stamp)

def on_run_fixed_button_click(b):
    with output_area:
        clear_output(wait=True)
        stamp = ts_stamp()
        print(f"--- Sabit Strateji Testi Başlatılıyor ({stamp}) ---")

        names = list(fixed_strategy_sel.options)
        tickers = list(fixed_ticker_sel.options)
        label_intervals = tuple(fixed_interval_widget.value)

        if not all([names, tickers, label_intervals]):
            print("HATA: Strateji, Hisse ve Zaman Dilimi seçimi zorunludur."); return

        sel_countries = list(fixed_country_sel.options)
        sel_exchanges = list(fixed_exchange_sel.options)
        ulke_label = ", ".join(sel_countries) if sel_countries else ""
        borsa_label = ", ".join(sel_exchanges) if sel_exchanges else ""

        all_trade_details: List[dict] = []
        for ticker in tickers:
            sektor = get_sector_safe(ticker)
            for label in label_intervals:
                print(f"Test ediliyor: {ticker} - {label}")
                try:
                    df = download_with_label(ticker, '5y', label)
                except Exception as e:
                    print(f"[UYARI] {ticker} ({label}) indirme hatası: {e}"); continue
                if df is None or df.empty:
                    print(f"[BİLGİ] {ticker} ({label}) boş veri."); continue
                if not {"Open","High","Low","Close","Volume"}.issubset(df.columns):
                    print(f"[UYARI] {ticker} ({label}) beklenen kolonlar eksik: {set(df.columns)}"); continue

                trend_ser, atr_pct_ser = trend_and_atr(df)

                for strat_name in names:
                    try:
                        sig_mask = FIXED_STRATEGIES[strat_name](df).fillna(False)
                        idx = np.where(sig_mask.values)[0]
                        if idx.size == 0: continue
                        combo = f"{strat_name} - {label}"
                        meta = dict(
                            ticker=ticker, combo=combo, ulke=ulke_label or infer_country_from_suffix(ticker),
                            borsa_kategori=borsa_label, zaman_dilimi=label,
                            sl_yuzde=7.0, tp_yuzde=15.0, tp_kismi_yuzde=50,
                            hacim_lb=4, hacim_artis_yuzde=49, sektor=sektor
                        )
                        logs = run_trade_simulation(df, idx, 5, 0.15, 0.07, 50, "bullish", meta)
                        all_trade_details.extend(logs)
                    except Exception as e:
                        print(f"[UYARI] {ticker} ({label}) '{strat_name}' hesaplanamadı: {e}")
                        continue

        process_and_display_results(all_trade_details, stamp)

run_dynamic_button.on_click(on_run_dynamic_button_click)
run_fixed_button.on_click(on_run_fixed_button_click)

# ==============================================================================
# 11) ARAYÜZÜ GÖSTER
# ==============================================================================
display(interface_layout, output_area)
