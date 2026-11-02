"""A Low-Risk ETF Trading Strategy - panel Streamlit.

Ejecutar con:  streamlit run app.py
"""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from plotly.subplots import make_subplots

from data import DEFAULT_UNIVERSE, PORTFOLIOS, load_prices
from strategy import Filters, StrategyParams, run_backtest

st.set_page_config(
    page_title="Low-Risk ETF Strategy",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }
    [data-testid="stMetric"] {
        background: rgba(148, 163, 184, .08);
        border: 1px solid rgba(148, 163, 184, .18);
        border-radius: 12px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { font-size: .82rem; }
    h1 { font-size: 1.95rem !important; letter-spacing: -.02em; }
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

INDICATORS = {
    "rsmk": "RSMK vs índice",
    "rsi": "RSI",
    "corr": "Correlación",
    "volume": "Volumen",
    "trend": "Tendencia (EMA/MA)",
}

COLORS = {
    "up": "#22c55e",
    "down": "#ef4444",
    "blue": "#38bdf8",
    "grey": "#94a3b8",
    "amber": "#f59e0b",
    "violet": "#a78bfa",
    "teal": "#2dd4bf",
}


@dataclass
class UIConfig:
    universe: list[str]
    portfolio_name: str
    start: date
    end: date
    params: StrategyParams
    filters: Filters
    chart_symbol: str
    chart_window: str
    n_panels: int
    chart_indicators: list[str]
    show_trades: bool
    show_signals: bool
    show_ma: bool
    show_eqw: bool
    shade_trades: bool
    log_equity: bool


def fmt_pct(value, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(value):
        return "—"
    return f"{value * 100:,.{digits}f}%"


def fmt_num(value, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(value):
        return "—"
    return f"{value:,.{digits}f}"


def fmt_pct_points(value, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(value):
        return "—"
    return f"{value:,.{digits}f}%"


def render_sidebar() -> UIConfig:
    with st.sidebar:
        st.markdown("## Configuración")
        st.caption("Estrategia RS/RSI long-only · Markos Katsanos")

        st.markdown("### Datos")
        portfolio_name = st.radio(
            "Portfolio",
            options=list(PORTFOLIOS.keys()),
            index=0,
            help="Dos universos del script original: ETFs sectoriales o la lista completa del artículo.",
        )
        portfolio_etfs = PORTFOLIOS[portfolio_name]
        with st.expander("Personalizar universo", expanded=False):
            universe = st.multiselect(
                "ETFs incluidos",
                options=sorted(set(portfolio_etfs + DEFAULT_UNIVERSE)),
                default=portfolio_etfs,
                key=f"universe_{portfolio_name}",
                help="Quita ETFs del portfolio o añade otros de la lista base.",
            )
            st.caption(
                "Para analizar un solo ETF, deja únicamente ese ticker y pon "
                "1 posición simultánea."
            )
        st.caption(f"{len(universe)} ETFs en el portfolio activo.")
        extra_raw = st.text_input(
            "Tickers adicionales",
            value="",
            placeholder="Ej.: ARKK, SOXX, IBIT",
            help="Separa los tickers por comas. Deben cotizar en EE. UU. y tener datos en Yahoo Finance.",
        )
        col_a, col_b = st.columns(2)
        start_date = col_a.date_input(
            "Inicio",
            value=date(2010, 1, 1),
            min_value=date(2000, 1, 1),
            max_value=date.today(),
        )
        end_date = col_b.date_input(
            "Fin",
            value=date.today(),
            min_value=date(2000, 1, 2),
            max_value=date.today(),
        )

        st.markdown("### Cartera")
        positions = st.selectbox(
            "Nº de posiciones simultáneas",
            options=[1, 2, 3, 4, 5],
            index=1,
            help="POSQTY del código AFL. El artículo optimiza entre 2 y 3.",
        )
        initial_capital = st.number_input(
            "Capital inicial (USD)",
            min_value=10_000,
            max_value=10_000_000,
            value=100_000,
            step=10_000,
            help="INITIALEQUITY del AFL.",
        )
        commission_pct = st.number_input(
            "Comisión por operación (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.005,
            step=0.001,
            format="%.4f",
            help="CommissionMode 3 / CommissionAmount 0.005 del AFL (0,005 % por lado).",
        )
        slippage_bps = st.number_input(
            "Deslizamiento (puntos básicos)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
            help="0 = backtest del artículo. 5-10 pb es una hipótesis más realista.",
        )
        priority = st.selectbox(
            "Prioridad de entrada",
            options=["Alfabética (AmiBroker)", "RSMK más fuerte"],
            index=0,
            help="Cuando hay más señales que huecos, AmiBroker entra por orden de símbolo. También puedes priorizar la señal de mayor RSMK.",
        )
        compound = st.checkbox(
            "Reinvertir beneficios",
            value=False,
            help="Desactivado: tamaño fijo = capital inicial / nº posiciones (como el AFL). Activado: equity actual / nº posiciones.",
        )

        with st.expander("Parámetros de la estrategia", expanded=False):
            time_exit = st.number_input("Salida temporal (barras)", 1, 180, 30, 1)
            rs_bars = st.number_input("RS BARS", 2, 200, 35, 1)
            rs_smoothing = st.number_input("Suavizado RSMK (EMA)", 1, 50, 4, 1)
            rs_ma = st.number_input("RS MA BARS", 5, 200, 50, 1)
            rs_crit = st.number_input("RSCRIT (umbral RS)", 0.5, 30.0, 3.0, 0.5)
            rsi_bars = st.number_input("RSI BARS", 2, 30, 10, 1)
            ma_bars = st.number_input("MA BARS (tendencia)", 5, 100, 15, 1)
            rsi_os = st.number_input("RSI OS (sobreventa)", 5.0, 60.0, 35.0, 1.0)
            corr_period = st.number_input("Ventana de correlación", 20, 300, 100, 10)
            corr_min = st.number_input("CORMIN", -1.0, 1.0, 0.60, 0.05, format="%.2f")
            profit_target = st.number_input("Objetivo de beneficio (%)", 1.0, 100.0, 25.0, 1.0)
            min_volume = st.number_input(
                "Volumen mínimo (acciones/día)",
                0,
                5_000_000,
                100_000,
                10_000,
            )
            reentry_delay = st.number_input(
                "Bloqueo tras objetivo (barras)",
                0,
                60,
                10,
                1,
                help="REENTRYDELAY del ApplyStop de beneficio.",
            )
            index_symbol = st.text_input("Índice de referencia", value="SPY").strip().upper()
            corr_method = st.selectbox(
                "Método de correlación",
                options=["Precios (AFL)", "Retornos"],
                index=0,
                help="AmiBroker Correlation() usa los precios; la alternativa usa retornos diarios.",
            )
            target_exit = st.selectbox(
                "Ejecución del objetivo",
                options=["Apertura siguiente (AFL)", "Precio objetivo"],
                index=0,
                help="ExitAtStop = 0 en el AFL: la salida se ejecuta en la apertura siguiente.",
            )

        with st.expander("Filtros de entrada / salida", expanded=False):
            f_volume = st.checkbox("Volumen medio 5d > mínimo", value=True)
            f_rs_ma = st.checkbox("RSMK > su media (MARS)", value=True)
            f_rs_crit = st.checkbox("RSMK > RSCRIT", value=True)
            f_rs_high = st.checkbox("RSMK en máximo de 3 barras", value=True)
            f_rs_cap = st.checkbox("RSMK < 25 (no sobrecompra)", value=True)
            f_regime = st.checkbox("Régimen índice + correlación", value=True)
            f_rsi = st.checkbox("RSI sobreventa y giro al alza", value=True)
            f_trend = st.checkbox("EMA(5) > MA(MABARS) desplazada", value=True)
            st.caption("Salidas")
            f_exit_signal = st.checkbox("Señal de venta RS/correlación", value=True)
            f_exit_target = st.checkbox("Objetivo de beneficio", value=True)
            f_exit_time = st.checkbox("Salida temporal", value=True)

        st.markdown("### Panel técnico")
        reference = index_symbol or "SPY"
        option_pool = [t for t in dict.fromkeys(universe + [reference]) if t]
        if not option_pool:
            option_pool = ["SPY"]
        if st.session_state.get("chart_symbol") not in option_pool:
            preferred = next(
                (t for t in ("QQQ", "XLK") if t in option_pool and t != reference),
                None,
            )
            if preferred is None:
                preferred = next(
                    (t for t in option_pool if t != reference), option_pool[0]
                )
            st.session_state["chart_symbol"] = preferred
        chart_symbol = st.selectbox("Activo a graficar", options=option_pool, key="chart_symbol")
        chart_window = st.selectbox(
            "Ventana del gráfico",
            options=["6 meses", "1 año", "2 años", "3 años", "5 años", "Todo el periodo"],
            index=2,
            help="Afecta solo a la visualización; el backtest usa todo el rango de fechas.",
        )
        n_panels = st.selectbox(
            "Nº de indicadores",
            options=[1, 2, 3, 4, 5],
            index=3,
            help="Número de paneles que se dibujan bajo las velas.",
        )
        chart_indicators = st.multiselect(
            "Indicadores a mostrar",
            options=list(INDICATORS.keys()),
            default=["rsmk", "rsi", "corr", "volume"],
            format_func=lambda k: INDICATORS[k],
        )
        show_trades = st.checkbox("Marcar operaciones", value=True)
        show_signals = st.checkbox("Mostrar señales teóricas", value=False)
        show_ma = st.checkbox("Medias en el gráfico de precio", value=True)
        shade_trades = st.checkbox("Sombrear periodos en posición", value=True)
        show_eqw = st.checkbox("Comparar con cartera equiponderada", value=False)
        log_equity = st.checkbox(
            "Escala logarítmica en la curva",
            value=False,
            help="Útil para comparar rentabilidades compuestas muy distintas.",
        )

        st.divider()
        st.caption(
            "Datos: Yahoo Finance (precios ajustados). "
            "Backtest con señales al cierre y ejecución a la apertura siguiente."
        )

    params = StrategyParams(
        rs_bars=int(rs_bars),
        rs_smoothing=int(rs_smoothing),
        rs_ma=int(rs_ma),
        rs_crit=float(rs_crit),
        index_symbol=index_symbol or "SPY",
        corr_period=int(corr_period),
        corr_min=float(corr_min),
        roc_bars=6,
        corr_method="returns" if corr_method.startswith("Retornos") else "price",
        rsi_bars=int(rsi_bars),
        rsi_os=float(rsi_os),
        ma_bars=int(ma_bars),
        profit_target=float(profit_target),
        time_exit=int(time_exit),
        reentry_delay=int(reentry_delay),
        min_volume=float(min_volume),
        initial_capital=float(initial_capital),
        positions=int(positions),
        commission_pct=float(commission_pct),
        slippage_bps=float(slippage_bps),
        compound=bool(compound),
        priority="rsmk" if priority.startswith("RSMK") else "alphabetical",
        target_exit="target" if target_exit.startswith("Precio") else "next_open",
    )
    filters = Filters(
        volume=f_volume,
        rs_above_ma=f_rs_ma,
        rs_above_crit=f_rs_crit,
        rs_new_high=f_rs_high,
        rs_cap=f_rs_cap,
        regime=f_regime,
        rsi=f_rsi,
        trend=f_trend,
        exit_signal=f_exit_signal,
        exit_target=f_exit_target,
        exit_time=f_exit_time,
    )
    extra_tickers = [
        t.strip().upper()
        for t in re.split(r"[,;\s]+", extra_raw)
        if t.strip()
    ]
    extra_tickers = [t for t in extra_tickers if re.fullmatch(r"[A-Z0-9.\-=^]{1,15}", t)]
    clean_universe = [t.strip().upper() for t in universe if t.strip()]
    clean_universe += [t for t in extra_tickers if t not in clean_universe]

    return UIConfig(
        universe=clean_universe,
        portfolio_name=portfolio_name,
        start=start_date,
        end=end_date,
        params=params,
        filters=filters,
        chart_symbol=chart_symbol,
        chart_window=chart_window,
        n_panels=int(n_panels),
        chart_indicators=chart_indicators,
        show_trades=show_trades,
        show_signals=show_signals,
        show_ma=show_ma,
        show_eqw=show_eqw,
        shade_trades=shade_trades,
        log_equity=log_equity,
    )


def _layout(fig: go.Figure, height: int, legend: bool = True) -> None:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=10, r=10, t=34, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.004,
            x=0,
            font=dict(size=10),
        ),
        hovermode="x unified",
        dragmode="pan",
        font=dict(size=11),
    )
    fig.update_xaxes(
        showgrid=False,
        rangebreaks=[dict(bounds=["sat", "mon"])],
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikecolor="rgba(148,163,184,0.5)",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.12)",
        zeroline=False,
    )


def chart_window_bounds(cfg: UIConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    end_ts = pd.Timestamp(cfg.end)
    if cfg.chart_window == "Todo el periodo":
        start_ts = pd.Timestamp(cfg.start)
    elif cfg.chart_window == "6 meses":
        start_ts = end_ts - pd.DateOffset(months=6)
    else:
        years = {
            "1 año": 1,
            "2 años": 2,
            "3 años": 3,
            "5 años": 5,
        }.get(cfg.chart_window, 3)
        start_ts = end_ts - pd.DateOffset(years=years)
    start_ts = max(start_ts, pd.Timestamp(cfg.start))
    return start_ts, end_ts


def build_technical_chart(
    ind: pd.DataFrame,
    trades: pd.DataFrame,
    open_positions: pd.DataFrame,
    symbol: str,
    cfg: UIConfig,
    params: StrategyParams,
) -> go.Figure:
    start_ts, end_ts = chart_window_bounds(cfg)
    df = ind.loc[(ind.index >= start_ts) & (ind.index <= end_ts)]
    panels = [k for k in cfg.chart_indicators][: cfg.n_panels]
    rows = 1 + len(panels)
    if panels:
        price_share = {1: 0.68, 2: 0.60, 3: 0.52, 4: 0.46, 5: 0.42}.get(
            len(panels), 0.46
        )
        row_heights = [price_share] + [(1.0 - price_share) / len(panels)] * len(panels)
    else:
        row_heights = [1.0]
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.018,
        row_heights=row_heights,
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=symbol,
            increasing_line_color=COLORS["up"],
            increasing_fillcolor=COLORS["up"],
            decreasing_line_color=COLORS["down"],
            decreasing_fillcolor=COLORS["down"],
            whiskerwidth=0.4,
        ),
        row=1,
        col=1,
    )

    if cfg.show_ma:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["ema5"],
                name=f"EMA 5",
                line=dict(color=COLORS["blue"], width=1.1),
                hovertemplate="EMA5 %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["ma_trend"],
                name=f"MA {params.ma_bars}",
                line=dict(color=COLORS["amber"], width=1.1),
                hovertemplate="MA %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    frames = [df for df in (trades, open_positions) if not df.empty]
    ops = pd.concat(frames, ignore_index=True) if frames else trades
    if cfg.show_trades and not ops.empty:
        tr = ops[ops["symbol"] == symbol].copy()
        overlap = (tr["entry_date"] <= end_ts) & (
            tr["exit_date"].isna() | (tr["exit_date"] >= start_ts)
        )
        tr = tr[overlap]
    else:
        tr = ops.iloc[0:0]

    if cfg.shade_trades and not tr.empty:
        for _, row in tr.iterrows():
            x0 = max(pd.Timestamp(row["entry_date"]), start_ts)
            x1 = row["exit_date"] if pd.notna(row["exit_date"]) else end_ts
            x1 = min(pd.Timestamp(x1), end_ts)
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor="rgba(34,197,94,0.07)",
                line_width=0,
                layer="below",
                row=1,
                col=1,
            )

    if not tr.empty:
        fig.add_trace(
            go.Scatter(
                x=tr["entry_date"],
                y=tr["entry_price"],
                mode="markers",
                name="Compra",
                marker=dict(
                    symbol="triangle-up",
                    size=14,
                    color="#3b82f6",
                    line=dict(color="#0b1220", width=1),
                ),
                customdata=np.stack(
                    [tr["pnl_pct"], tr["shares"], tr["exit_reason"]], axis=1
                ),
                hovertemplate=(
                    "<b>Compra %{x|%d/%m/%Y}</b><br>"
                    "Precio %{y:.2f}<br>Títulos %{customdata[1]}"
                    "<br>P&L %{customdata[0]:.2f}%<br>%{customdata[2]}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        closed_tr = tr[tr["exit_date"].notna()]
        if not closed_tr.empty:
            fig.add_trace(
                go.Scatter(
                    x=closed_tr["exit_date"],
                    y=closed_tr["exit_price"],
                    mode="markers",
                    name="Venta",
                    marker=dict(
                        symbol="triangle-down",
                        size=14,
                        color="#f97316",
                        line=dict(color="#0b1220", width=1),
                    ),
                    customdata=np.stack(
                        [closed_tr["pnl_pct"], closed_tr["bars"], closed_tr["exit_reason"]],
                        axis=1,
                    ),
                    hovertemplate=(
                        "<b>Venta %{x|%d/%m/%Y}</b><br>"
                        "Precio %{y:.2f}<br>Barras %{customdata[1]}"
                        "<br>P&L %{customdata[0]:.2f}%<br>%{customdata[2]}<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

    if cfg.show_signals:
        buys = df[df["buy"]]
        sells = df[df["sell"]]
        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys.index,
                    y=buys["Low"] * 0.995,
                    mode="markers",
                    name="Señal compra (teórica)",
                    marker=dict(symbol="circle-open", size=7, color=COLORS["blue"]),
                    hovertemplate="Señal compra %{x|%d/%m/%Y}<extra></extra>",
                ),
                row=1,
                col=1,
            )
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells.index,
                    y=sells["High"] * 1.005,
                    mode="markers",
                    name="Señal venta (teórica)",
                    marker=dict(symbol="x", size=7, color=COLORS["amber"]),
                    hovertemplate="Señal venta %{x|%d/%m/%Y}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    fig.update_yaxes(title_text=symbol, row=1, col=1)

    for j, key in enumerate(panels, start=2):
        if key == "rsmk":
            rs = df["rsmk"]
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=rs.where(rs >= 0),
                    name="RSMK > 0",
                    line=dict(color=COLORS["up"], width=1.4),
                    connectgaps=False,
                    hovertemplate="RSMK %{y:.2f}<extra></extra>",
                ),
                row=j,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=rs.where(rs < 0),
                    name="RSMK < 0",
                    line=dict(color=COLORS["down"], width=1.4),
                    connectgaps=False,
                    hovertemplate="RSMK %{y:.2f}<extra></extra>",
                ),
                row=j,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["rs_ma"],
                    name=f"MARS ({params.rs_ma})",
                    line=dict(color=COLORS["amber"], width=1.2, dash="dash"),
                    hovertemplate="MARS %{y:.2f}<extra></extra>",
                ),
                row=j,
                col=1,
            )
            fig.add_hline(
                y=params.rs_crit,
                line=dict(color="rgba(34,197,94,.55)", width=1, dash="dot"),
                row=j,
                col=1,
            )
            fig.add_hline(
                y=-params.rs_crit,
                line=dict(color="rgba(239,68,68,.55)", width=1, dash="dot"),
                row=j,
                col=1,
            )
            fig.add_hline(
                y=params.rs_cap,
                line=dict(color="rgba(148,163,184,.5)", width=1, dash="dot"),
                row=j,
                col=1,
            )
            fig.add_hline(
                y=0,
                line=dict(color="rgba(148,163,184,.4)", width=1),
                row=j,
                col=1,
            )
            fig.update_yaxes(title_text="RSMK", row=j, col=1)

        elif key == "rsi":
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["rsi"],
                    name=f"RSI {params.rsi_bars}",
                    line=dict(color=COLORS["violet"], width=1.4),
                    hovertemplate="RSI %{y:.2f}<extra></extra>",
                ),
                row=j,
                col=1,
            )
            fig.add_hline(
                y=params.rsi_os,
                line=dict(color="rgba(245,158,11,.7)", width=1, dash="dash"),
                annotation_text=f"OS {params.rsi_os:g}",
                annotation_position="top left",
                annotation_font_size=10,
                row=j,
                col=1,
            )
            fig.add_hline(
                y=50,
                line=dict(color="rgba(148,163,184,.35)", width=1),
                row=j,
                col=1,
            )
            fig.add_hline(
                y=70,
                line=dict(color="rgba(239,68,68,.35)", width=1, dash="dot"),
                row=j,
                col=1,
            )
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=j, col=1)

        elif key == "corr":
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["corr"],
                    name=f"Correlación {params.corr_period}d",
                    line=dict(color=COLORS["teal"], width=1.4),
                    hovertemplate="Corr %{y:.3f}<extra></extra>",
                ),
                row=j,
                col=1,
            )
            for level, color in [
                (params.corr_min, "rgba(34,197,94,.55)"),
                (-params.corr_min, "rgba(34,197,94,.35)"),
                (params.sell_corr, "rgba(239,68,68,.5)"),
                (0, "rgba(148,163,184,.35)"),
            ]:
                fig.add_hline(
                    y=level,
                    line=dict(color=color, width=1, dash="dot"),
                    row=j,
                    col=1,
                )
            fig.update_yaxes(title_text="Correlación", range=[-1, 1], row=j, col=1)

        elif key == "volume":
            up = df["Close"] >= df["Open"]
            bar_colors = np.where(up, "rgba(34,197,94,.55)", "rgba(239,68,68,.55)")
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df["Volume"],
                    name="Volumen",
                    marker_color=bar_colors,
                    hovertemplate="Volumen %{y:,.0f}<extra></extra>",
                ),
                row=j,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["vol_ma5"],
                    name="Vol. MA5",
                    line=dict(color=COLORS["blue"], width=1.1),
                    hovertemplate="MA5 %{y:,.0f}<extra></extra>",
                ),
                row=j,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["vol_ma50"],
                    name="Vol. MA50",
                    line=dict(color=COLORS["amber"], width=1.1),
                    hovertemplate="MA50 %{y:,.0f}<extra></extra>",
                ),
                row=j,
                col=1,
            )
            fig.add_hline(
                y=params.min_volume,
                line=dict(color="rgba(148,163,184,.6)", width=1, dash="dot"),
                row=j,
                col=1,
            )
            fig.update_yaxes(title_text="Volumen", tickformat=".2s", row=j, col=1)

        elif key == "trend":
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["ema5"],
                    name="EMA 5",
                    line=dict(color=COLORS["blue"], width=1.4),
                    hovertemplate="EMA5 %{y:.2f}<extra></extra>",
                ),
                row=j,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["ma_trend_lag5"],
                    name=f"MA {params.ma_bars} (-5)",
                    line=dict(color=COLORS["amber"], width=1.4, dash="dash"),
                    hovertemplate="MA-5 %{y:.2f}<extra></extra>",
                ),
                row=j,
                col=1,
            )
            fig.update_yaxes(title_text="Precio", row=j, col=1)

    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_xaxes(range=[start_ts, end_ts])
    height = 440 + 155 * len(panels) if panels else 520
    _layout(fig, height)
    return fig


def build_equity_chart(result, cfg: UIConfig) -> go.Figure:
    eq = result.equity
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.74, 0.26],
        vertical_spacing=0.06,
    )
    fig.add_trace(
        go.Scatter(
            x=eq.index,
            y=eq["strategy"],
            name="Estrategia",
            line=dict(color=COLORS["blue"], width=2),
            hovertemplate="Estrategia %{y:,.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=eq.index,
            y=eq["benchmark"],
            name=f"{cfg.params.index_symbol} B&H",
            line=dict(color=COLORS["grey"], width=1.5, dash="dash"),
            hovertemplate=f"{cfg.params.index_symbol} %{{y:,.0f}}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    if cfg.show_eqw:
        fig.add_trace(
            go.Scatter(
                x=eq.index,
                y=eq["equal_weight"],
                name="Cartera equiponderada B&H",
                line=dict(color=COLORS["teal"], width=1.2, dash="dot"),
                hovertemplate="Equiponderada %{y:,.0f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    dd = eq["strategy"] / eq["strategy"].cummax() - 1.0
    dd_b = eq["benchmark"] / eq["benchmark"].cummax() - 1.0
    fig.add_trace(
        go.Scatter(
            x=eq.index,
            y=dd,
            name="Drawdown estrategia",
            fill="tozeroy",
            line=dict(color=COLORS["down"], width=1),
            fillcolor="rgba(239,68,68,.25)",
            hovertemplate="DD %{y:.2%}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=eq.index,
            y=dd_b,
            name="Drawdown B&H",
            line=dict(color=COLORS["grey"], width=1, dash="dot"),
            hovertemplate="DD B&H %{y:.2%}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text="Capital (USD)",
        tickformat=",.0f",
        type="log" if cfg.log_equity else "linear",
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    _layout(fig, 640)
    return fig


def metrics_table(result, bench_label: str) -> pd.DataFrame:
    m = result.metrics
    b = result.bench_metrics
    avg = m.get("avg_pnl_pct")
    best = m.get("best_pnl_pct")
    worst = m.get("worst_pnl_pct")
    rows = [
        ("Rentabilidad total", fmt_pct(m["total_return"]), fmt_pct(b["total_return"])),
        ("CAGR", fmt_pct(m["cagr"]), fmt_pct(b["cagr"])),
        ("Volatilidad anual", fmt_pct(m["volatility"]), fmt_pct(b["volatility"])),
        ("Sharpe", fmt_num(m["sharpe"]), fmt_num(b["sharpe"])),
        ("Sortino", fmt_num(m["sortino"]), fmt_num(b["sortino"])),
        ("Máxima caída", fmt_pct(m["max_drawdown"]), fmt_pct(b["max_drawdown"])),
        ("Calmar", fmt_num(m["calmar"]), fmt_num(b["calmar"])),
        ("Exposición media", fmt_pct(m.get("exposure")), "100%"),
        ("Operaciones cerradas", f"{int(m.get('trades', 0)):,}", "—"),
        ("Tasa de acierto", fmt_pct(m.get("win_rate")), "—"),
        ("Profit factor", fmt_num(m.get("profit_factor")), "—"),
        (
            "Media por operación",
            f"{avg:,.2f}%" if avg is not None and math.isfinite(avg) else "—",
            "—",
        ),
        (
            "Mejor operación",
            f"{best:,.2f}%" if best is not None and math.isfinite(best) else "—",
            "—",
        ),
        (
            "Peor operación",
            f"{worst:,.2f}%" if worst is not None and math.isfinite(worst) else "—",
            "—",
        ),
    ]
    return pd.DataFrame(rows, columns=["Métrica", "Estrategia", f"{bench_label} B&H"])


def yearly_table(result, bench_label: str) -> pd.DataFrame:
    out = result.yearly.reset_index().rename(columns={"Ano": "Año"})
    out["Año"] = out["Año"].astype(int).astype(str)
    cols = ["Estrategia", f"{bench_label} B&H", "Diferencia"]
    for col in cols:
        out[col] = out[col].map(
            lambda v: f"{v:+.2f}%" if pd.notna(v) and math.isfinite(float(v)) else "—"
        )
    return out[["Año", "Estrategia", f"{bench_label} B&H", "Diferencia"]]


def operations_view(trades: pd.DataFrame, include_open: bool = False) -> pd.DataFrame:
    view = trades.copy()
    view["entry_date"] = pd.to_datetime(view["entry_date"]).dt.strftime("%Y-%m-%d")
    view["exit_date"] = pd.to_datetime(view["exit_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    view = view.rename(
        columns={
            "symbol": "Activo",
            "entry_date": "Entrada",
            "entry_price": "Precio entrada",
            "exit_date": "Salida",
            "exit_price": "Precio salida",
            "shares": "Títulos",
            "entry_value": "Importe entrada",
            "exit_value": "Importe salida",
            "commission": "Comisiones",
            "pnl": "P&L",
            "pnl_pct": "P&L (%)",
            "bars": "Barras",
            "exit_reason": "Motivo",
            "entry_rs": "RS entrada",
            "exit_rs": "RS salida",
            "entry_rsi": "RSI entrada",
            "entry_corr": "Corr. entrada",
        }
    )
    if include_open:
        view["Salida"] = view["Salida"].fillna("Abierta")
    return view


def _style_sheet(ws) -> None:
    ws.freeze_panes = "A2"
    header_fill = PatternFill("solid", fgColor="1F4E78")
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
    for col in ws.columns:
        max_len = 0
        for cell in col:
            if cell.value is not None:
                max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(
            max(10, max_len + 2), 32
        )


def build_excel(result, cfg: UIConfig, bench_label: str) -> bytes:
    buf = io.BytesIO()
    trades = operations_view(result.trades)
    open_pos = operations_view(result.open_positions, include_open=True)
    yearly = yearly_table(result, bench_label)
    metrics = metrics_table(result, bench_label)
    equity = result.equity.reset_index().rename(
        columns={
            "date": "Fecha",
            "strategy": "Estrategia",
            "benchmark": f"{bench_label} B&H",
            "equal_weight": "Equiponderada",
            "cash": "Efectivo",
            "invested": "Invertido",
            "positions": "Posiciones",
        }
    )
    param_labels = {
        "rs_bars": "RS BARS",
        "rs_smoothing": "Suavizado RSMK",
        "rs_ma": "RS MA BARS",
        "rs_crit": "RSCRIT",
        "rs_cap": "RSMK máximo",
        "min_volume": "Volumen mínimo",
        "index_symbol": "Índice de referencia",
        "corr_period": "Ventana correlación",
        "corr_min": "CORMIN",
        "sell_corr": "Correlación de venta",
        "roc_bars": "ROC barras",
        "roc_threshold": "ROC umbral (%)",
        "corr_method": "Método correlación",
        "rsi_bars": "RSI BARS",
        "rsi_os": "RSI sobreventa",
        "ma_bars": "MA BARS",
        "profit_target": "Objetivo beneficio (%)",
        "time_exit": "Salida temporal (barras)",
        "reentry_delay": "Bloqueo tras objetivo",
        "initial_capital": "Capital inicial",
        "positions": "Nº posiciones",
        "commission_pct": "Comisión (%)",
        "slippage_bps": "Deslizamiento (pb)",
        "lot_size": "Lote",
        "min_position_value": "Importe mínimo posición",
        "min_shares": "Títulos mínimos",
        "compound": "Reinvertir",
        "priority": "Prioridad",
        "target_exit": "Ejecución objetivo",
    }
    params_rows = [
        {"Parámetro": param_labels.get(k, k), "Valor": v}
        for k, v in result.params.items()
    ]
    filters = cfg.filters
    for key, label in [
        ("volume", "Filtro volumen"),
        ("rs_above_ma", "Filtro RSMK > MARS"),
        ("rs_above_crit", "Filtro RSMK > RSCRIT"),
        ("rs_new_high", "Filtro máximo 3 barras"),
        ("rs_cap", "Filtro RSMK < 25"),
        ("regime", "Filtro régimen"),
        ("rsi", "Filtro RSI"),
        ("trend", "Filtro tendencia"),
        ("exit_signal", "Salida señal"),
        ("exit_target", "Salida objetivo"),
        ("exit_time", "Salida temporal"),
    ]:
        params_rows.append(
            {"Parámetro": label, "Valor": "Sí" if getattr(filters, key) else "No"}
        )
    params_df = pd.DataFrame(params_rows)
    params_df["Valor"] = params_df["Valor"].astype(str)

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        trades.to_excel(writer, sheet_name="Operaciones", index=False)
        open_pos.to_excel(writer, sheet_name="Posiciones abiertas", index=False)
        equity.to_excel(writer, sheet_name="Curva de capital", index=False)
        metrics.to_excel(writer, sheet_name="Métricas", index=False)
        yearly.to_excel(writer, sheet_name="Rentabilidad anual", index=False)
        params_df.to_excel(writer, sheet_name="Parámetros", index=False)
        for ws in writer.book.worksheets:
            _style_sheet(ws)
    return buf.getvalue()


def render_trades(result, cfg: UIConfig, bench_label: str) -> None:
    st.markdown("### Operaciones")
    closed = result.trades.copy()
    open_pos = result.open_positions.copy()

    if closed.empty and open_pos.empty:
        st.info("La estrategia no generó operaciones en el periodo seleccionado.")
        return

    c1, c2, c3 = st.columns([2, 2, 1])
    if not closed.empty:
        sym_options = sorted(closed["symbol"].unique())
        reason_options = sorted(closed["exit_reason"].unique())
    else:
        sym_options, reason_options = [], []
    sel_syms = c1.multiselect(
        "Filtrar por activo", sym_options, default=[], placeholder="Todos"
    )
    sel_reasons = c2.multiselect(
        "Filtrar por motivo", reason_options, default=[], placeholder="Todos"
    )
    with c3:
        st.metric("Cerradas", f"{len(closed):,}", label_visibility="visible")
        st.caption(f"{len(open_pos)} abiertas al final")

    filtered = closed
    if sel_syms:
        filtered = filtered[filtered["symbol"].isin(sel_syms)]
    if sel_reasons:
        filtered = filtered[filtered["exit_reason"].isin(sel_reasons)]

    view = operations_view(filtered)
    display_cols = [
        "Activo",
        "Entrada",
        "Salida",
        "Precio entrada",
        "Precio salida",
        "Títulos",
        "Importe entrada",
        "Comisiones",
        "P&L",
        "P&L (%)",
        "Barras",
        "Motivo",
        "RS entrada",
    ]
    display = view[[c for c in display_cols if c in view.columns]]
    st.dataframe(
        display,
        use_container_width=True,
        height=430,
        hide_index=True,
        column_config={
            "Entrada": st.column_config.TextColumn(width="small"),
            "Salida": st.column_config.TextColumn(width="small"),
            "Precio entrada": st.column_config.NumberColumn(format="%.2f"),
            "Precio salida": st.column_config.NumberColumn(format="%.2f"),
            "Títulos": st.column_config.NumberColumn(format="%d"),
            "Importe entrada": st.column_config.NumberColumn(format="%.2f"),
            "Comisiones": st.column_config.NumberColumn(format="%.2f"),
            "P&L": st.column_config.NumberColumn(format="%.2f"),
            "P&L (%)": st.column_config.NumberColumn(format="%.2f"),
            "RS entrada": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    stamp = pd.Timestamp.today().strftime("%Y%m%d")
    d1, d2, d3 = st.columns(3)
    csv_bytes = view.to_csv(index=False).encode("utf-8-sig")
    d1.download_button(
        "Descargar operaciones (CSV)",
        data=csv_bytes,
        file_name=f"operaciones_{stamp}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    xlsx_bytes = build_excel(result, cfg, bench_label)
    d2.download_button(
        "Descargar todo (XLSX)",
        data=xlsx_bytes,
        file_name=f"alowrisk_backtest_{stamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    equity_csv = (
        result.equity.reset_index()
        .rename(
            columns={
                "date": "Fecha",
                "strategy": "Estrategia",
                "benchmark": "Benchmark",
                "cash": "Efectivo",
                "invested": "Invertido",
                "positions": "Posiciones",
                "equal_weight": "Equiponderada",
            }
        )
        .to_csv(index=False)
        .encode("utf-8-sig")
    )
    d3.download_button(
        "Descargar curva (CSV)",
        data=equity_csv,
        file_name=f"curva_capital_{stamp}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if not open_pos.empty:
        with st.expander(f"Posiciones abiertas al cierre del periodo ({len(open_pos)})"):
            st.dataframe(
                operations_view(open_pos, include_open=True),
                use_container_width=True,
                hide_index=True,
            )


def main() -> None:
    cfg = render_sidebar()

    st.title("A Low-Risk ETF Trading Strategy")
    st.caption(
        "Sistema dual RS/RSI long-only sobre ETFs · réplica del código AFL de Markos Katsanos. "
        "Señales al cierre, ejecución a la apertura siguiente."
    )
    st.caption(
        f"Portfolio activo: **{cfg.portfolio_name}** · {len(cfg.universe)} ETFs · "
        f"índice {cfg.params.index_symbol}"
    )

    if not cfg.universe:
        st.error("Selecciona al menos un ETF del universo.")
        st.stop()
    if cfg.end <= cfg.start:
        st.error("La fecha fin debe ser posterior a la fecha inicio.")
        st.stop()

    tickers = sorted(set(cfg.universe + [cfg.params.index_symbol]))
    warm_start = (pd.Timestamp(cfg.start) - pd.Timedelta(days=550)).strftime("%Y-%m-%d")

    try:
        with st.spinner(
            f"Descargando datos de {len(tickers)} tickers de Yahoo Finance…"
        ):
            prices, missing = load_prices(tuple(tickers), warm_start, str(cfg.end))
    except Exception as exc:
        st.error(f"No se pudieron descargar los datos: {exc}")
        st.stop()

    if cfg.params.index_symbol not in prices:
        st.error(
            f"No hay datos suficientes para el índice de referencia "
            f"**{cfg.params.index_symbol}**. Prueba con SPY."
        )
        st.stop()

    available = [t for t in cfg.universe if t in prices]
    if not available:
        st.error("Ninguno de los ETFs seleccionados tiene datos suficientes en el periodo.")
        st.stop()

    if missing:
        st.warning(
            "Sin datos suficientes para: " + ", ".join(missing) + ". Se han excluido del backtest."
        )

    try:
        result = run_backtest(prices, cfg.params, cfg.filters, cfg.start, cfg.end)
    except Exception as exc:
        st.error(f"Error en el backtest: {exc}")
        st.stop()

    bench_label = cfg.params.index_symbol
    m = result.metrics
    b = result.bench_metrics

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric(
        "Rentabilidad total",
        fmt_pct(m["total_return"]),
        delta=f"{(m['total_return'] - b['total_return']) * 100:+.1f} pp vs {bench_label}",
    )
    k2.metric(
        "CAGR",
        fmt_pct(m["cagr"]),
        delta=f"{(m['cagr'] - b['cagr']) * 100:+.1f} pp",
    )
    k3.metric(
        "Sharpe",
        fmt_num(m["sharpe"]),
        delta=f"{m['sharpe'] - b['sharpe']:+.2f}",
    )
    k4.metric(
        "Máxima caída",
        fmt_pct(m["max_drawdown"]),
        delta=f"{(m['max_drawdown'] - b['max_drawdown']) * 100:+.1f} pp",
    )
    k5.metric("Operaciones", f"{int(m.get('trades', 0)):,}", help="Operaciones cerradas.")
    k6.metric(
        "Acierto",
        fmt_pct(m.get("win_rate")),
        help="Porcentaje de operaciones cerradas con beneficio.",
    )
    st.caption(
        f"Exposición media {fmt_pct(m.get('exposure'))} · "
        f"{int(m.get('trades', 0))} operaciones cerradas · "
        f"{int(m.get('open_positions', 0))} posiciones abiertas al final · "
        f"periodo {cfg.start:%d/%m/%Y} → {cfg.end:%d/%m/%Y}"
    )

    if not cfg.filters.entry_enabled():
        st.warning("Todos los filtros de entrada están desactivados: no se generarán compras.")

    if cfg.chart_symbol not in result.indicators:
        cfg.chart_symbol = (
            bench_label if bench_label in result.indicators else next(iter(result.indicators))
        )

    st.markdown("### Panel técnico")
    shown = cfg.chart_indicators[: cfg.n_panels]
    chart_start, chart_end = chart_window_bounds(cfg)
    caption = f"{cfg.chart_symbol} · {chart_start:%d/%m/%Y} → {chart_end:%d/%m/%Y}"
    caption += f" · backtest {cfg.start:%d/%m/%Y} → {cfg.end:%d/%m/%Y}"
    if shown:
        caption += " · paneles: " + ", ".join(INDICATORS[k] for k in shown)
        if len(cfg.chart_indicators) > len(shown):
            caption += f" (mostrando {len(shown)} de {len(cfg.chart_indicators)})"
    st.caption(caption)
    if cfg.chart_symbol == bench_label:
        st.caption(
            "El RSMK del índice frente a sí mismo es 0 y su correlación es 1; "
            "elige otro ETF para ver la fuerza relativa."
        )
    fig = build_technical_chart(
        result.indicators[cfg.chart_symbol],
        result.trades,
        result.open_positions,
        cfg.chart_symbol,
        cfg,
        cfg.params,
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )

    st.markdown("### Backtest vs Buy & Hold")
    st.plotly_chart(build_equity_chart(result, cfg), use_container_width=True, config={"displaylogo": False})

    left, right = st.columns([1.05, 1.0])
    with left:
        st.markdown("**Métricas**")
        st.dataframe(
            metrics_table(result, bench_label),
            use_container_width=True,
            hide_index=True,
            height=520,
        )
    with right:
        st.markdown("**Rentabilidad por año natural**")
        st.dataframe(
            yearly_table(result, bench_label),
            use_container_width=True,
            hide_index=True,
            height=520,
        )

    render_trades(result, cfg, bench_label)

    with st.expander("Universo y calidad de datos"):
        rows = []
        for ticker, df in sorted(prices.items()):
            rows.append(
                {
                    "Ticker": ticker,
                    "Barras": len(df),
                    "Primera fecha": df.index[0].date(),
                    "Última fecha": df.index[-1].date(),
                    "Último cierre": round(float(df["Close"].iloc[-1]), 2),
                    "Volumen medio 60d": int(df["Volume"].tail(60).mean()),
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            height=360,
        )
        st.caption(
            "Precios ajustados por dividendos y splits (Yahoo Finance). "
            "Si un ETF empieza después de la fecha de inicio, sus indicadores tardan "
            f"{max(cfg.params.corr_period, cfg.params.rs_bars + cfg.params.rs_smoothing, cfg.params.rs_ma)} "
            "barras en estar disponibles."
        )

    with st.expander("Metodología y limitaciones"):
        st.markdown(
            f"""
**Indicador RSMK**

- `RS1 = ln(C / {bench_label}) - ln(C / {bench_label})[RSBARS]`
- `RSMK = EMA(RS1, {cfg.params.rs_smoothing}) × 100`
- `MARS = MA(RSMK, {cfg.params.rs_ma})`

**Entrada (todas las condiciones, al cierre, ejecución a la apertura siguiente)**

1. `MA(Volumen, 5) > {cfg.params.min_volume:,.0f}`
2. `RSMK > MARS`
3. `RSMK > {cfg.params.rs_crit:g}`
4. `RSMK > máximo de RSMK de las 3 barras anteriores`
5. `RSMK < {cfg.params.rs_cap:g}`
6. `(ROC({bench_label}, {cfg.params.roc_bars}) > {cfg.params.roc_threshold:g} y Corr > {cfg.params.corr_min:g})`
   o `(ROC({bench_label}, {cfg.params.roc_bars}) < {cfg.params.roc_threshold:g} y Corr < -{cfg.params.corr_min:g})`
7. `mínimo de RSI({cfg.params.rsi_bars}) en 20 barras < {cfg.params.rsi_os:g}` y `RSI > máximo de RSI de las 2 barras anteriores`
8. `EMA(5) > MA({cfg.params.ma_bars}) de hace 5 barras`

**Salidas**

- Señal: `RSMK < -{cfg.params.rs_crit:g}` y `Corr > {cfg.params.sell_corr:g}`.
- Objetivo de beneficio: +{cfg.params.profit_target:g} %.
- Salida temporal: {cfg.params.time_exit} barras.
- Máximo {cfg.params.positions} posiciones, tamaño {'equity/nº posiciones' if cfg.params.compound else 'capital inicial/nº posiciones'}.

**Limitaciones**

- Datos de Yahoo Finance: posibles huecos, splits mal ajustados o ETFs desaparecidos (sesgo de supervivencia).
- El backtest no modela liquidez intradía ni impacto de mercado; el deslizamiento es una hipótesis.
- Los resultados del artículo original dependen de la lista de ETFs y del periodo exactos; esta app permite reproducir la lógica, no garantiza cifras idénticas.
- Uso educativo; no es asesoramiento financiero.
"""
        )


if __name__ == "__main__":
    main()
