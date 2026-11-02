"""Descarga y normalizacion de datos diarios desde Yahoo Finance."""

from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf

OHLCV = ["Open", "High", "Low", "Close", "Volume"]

DEFAULT_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA", "MDY", "EFA", "EEM", "FXI", "EWJ", "EWG",
    "TLT", "IEF", "LQD", "HYG", "GLD", "SLV", "VNQ",
    "XLE", "XLF", "XLK", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB",
]

DEFAULT_SELECTION = [
    "SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG",
    "GLD", "VNQ", "XLE", "XLF", "XLK", "XLV",
]

SECTOR_ETFS = [
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
]

ARTICLE_ETFS = [
    "AGG", "ARKK", "AIA", "BATT", "BITO", "COPX", "DBC", "ECH", "EEM", "EFA",
    "EMLC", "EWY", "FCOM", "FXI", "GDX", "GLD", "GREK", "GRID", "GUNR", "HYG",
    "ICLN", "IDV", "IEF", "IEV", "IGPT", "IGV", "IHI", "ILF", "INDA", "ITA",
    "ITB", "IWM", "IYH", "IYR", "IYZ", "KBE", "KIE", "LIT", "LQD", "MGK",
    "MOO", "MTUM", "NLR", "PDP", "PFF", "QQQ", "REMX", "ROBO", "SDS", "SHY",
    "SIL", "SKYY", "SOXX", "SPYG", "TAN", "TLT", "UFO", "UNG", "URA", "USO",
    "VGT", "VIXM", "VTV", "VUG", "XBI", "XME", "XOP", "XRT", "XTL", "XTN",
]

PORTFOLIOS = {
    f"Sector ETFs ({len(SECTOR_ETFS)})": SECTOR_ETFS,
    f"Article ETFs ({len(ARTICLE_ETFS)})": ARTICLE_ETFS,
}


def _clean_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


def _extract(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        level1 = raw.columns.get_level_values(1)
        if ticker in level0:
            df = raw[ticker]
        elif ticker in level1:
            df = raw.xs(ticker, axis=1, level=1)
        else:
            return None
    else:
        df = raw
    if not set(OHLCV).issubset(df.columns):
        return None
    df = df[OHLCV].copy()
    df = _clean_index(df)
    df = df.dropna(subset=["Close"])
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df if not df.empty else None


def _download_one(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception:
        return None
    return _extract(raw, ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(
    tickers: tuple[str, ...],
    start: str,
    end: str,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Devuelve (datos por ticker, tickers sin datos suficientes)."""
    tickers = tuple(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    if not tickers:
        return {}, []

    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        raw = yf.download(
            list(tickers),
            start=start,
            end=end_exclusive,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        raw = pd.DataFrame()

    out: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for ticker in tickers:
        df = _extract(raw, ticker)
        if df is None or len(df) < 120:
            df = _download_one(ticker, start, end_exclusive)
        if df is None or len(df) < 120:
            missing.append(ticker)
        else:
            out[ticker] = df

    return out, missing
