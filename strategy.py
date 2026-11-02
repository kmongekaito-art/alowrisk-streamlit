"""Motor de la estrategia "A Low-Risk ETF Trading Strategy" (Markos Katsanos).

Replica en Python/AFL-to-pandas la estrategia RS/RSI long-only para ETFs:

    BUY  = volumen medio > MinVol
           AND RSMK > MA(RSMK, MAR)
           AND RSMK > RSCRIT
           AND RSMK > HHV(RSMK, 3) del dia anterior
           AND RSMK < 25
           AND ((ROC(SPY, 6) > -6 AND CORR > CORMIN)
                OR (ROC(SPY, 6) < -6 AND CORR < -CORMIN))
           AND LLV(RSI, 20) < RSIOS
           AND RSI > HHV(RSI, 2) del dia anterior
           AND EMA(C, 5) > MA(C, MABARS) de hace 5 barras

    SELL = RSMK < -RSCRIT AND CORR > 0.8

Las senales se evaluan al cierre y se ejecutan a la apertura siguiente
(SetTradeDelays(1,1,1,1), BuyPrice/SellPrice = OPEN).  El backtest es de
cartera: varias ETFs, un maximo de POSQTY posiciones y tamano fijo
(INITIALEQUITY / POSQTY), como en AmiBroker.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyParams:
    # Indicador RSMK / RS
    rs_bars: int = 35
    rs_smoothing: int = 4
    rs_ma: int = 50
    rs_crit: float = 3.0
    rs_cap: float = 25.0
    min_volume: float = 100_000.0
    index_symbol: str = "SPY"
    corr_period: int = 100
    corr_min: float = 0.60
    sell_corr: float = 0.80
    roc_bars: int = 6
    roc_threshold: float = -6.0
    corr_method: str = "price"

    # Indicadores RSI / tendencia
    rsi_bars: int = 10
    rsi_os: float = 35.0
    ma_bars: int = 15

    # Salidas
    profit_target: float = 25.0
    time_exit: int = 30
    reentry_delay: int = 10

    # Cartera / ejecucion
    initial_capital: float = 100_000.0
    positions: int = 2
    commission_pct: float = 0.005
    slippage_bps: float = 0.0
    lot_size: int = 10
    min_position_value: float = 1_000.0
    min_shares: int = 10
    compound: bool = False
    priority: str = "alphabetical"
    target_exit: str = "next_open"


@dataclass(frozen=True)
class Filters:
    volume: bool = True
    rs_above_ma: bool = True
    rs_above_crit: bool = True
    rs_new_high: bool = True
    rs_cap: bool = True
    regime: bool = True
    rsi: bool = True
    trend: bool = True
    exit_signal: bool = True
    exit_target: bool = True
    exit_time: bool = True

    def entry_enabled(self) -> bool:
        return any(
            (
                self.volume,
                self.rs_above_ma,
                self.rs_above_crit,
                self.rs_new_high,
                self.rs_cap,
                self.regime,
                self.rsi,
                self.trend,
            )
        )


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    open_positions: pd.DataFrame
    indicators: dict[str, pd.DataFrame]
    metrics: dict
    bench_metrics: dict
    yearly: pd.DataFrame
    params: dict


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    """RSI de Wilder, equivalente a RSI() de AmiBroker."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return rsi


def compute_indicators(
    df: pd.DataFrame,
    bench_close: pd.Series,
    params: StrategyParams,
) -> pd.DataFrame:
    """Calcula RSMK, MARS, RSI, correlacion y las condiciones elementales."""
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float)
    bench = bench_close.reindex(c.index).ffill()

    rel = np.log(c / bench)
    rs_change = rel - rel.shift(params.rs_bars)
    rsmk = ema(rs_change, params.rs_smoothing) * 100.0
    rs_ma = rsmk.rolling(params.rs_ma, min_periods=params.rs_ma).mean()

    if params.corr_method == "returns":
        corr = c.pct_change(fill_method=None).rolling(params.corr_period).corr(bench.pct_change(fill_method=None))
    else:
        corr = c.rolling(params.corr_period).corr(bench)

    roc_index = (bench / bench.shift(params.roc_bars) - 1.0) * 100.0
    rsi = wilder_rsi(c, params.rsi_bars)
    ema5 = ema(c, 5)
    ma_trend = c.rolling(params.ma_bars, min_periods=params.ma_bars).mean()

    cond_volume = v.rolling(5, min_periods=5).mean() > params.min_volume
    cond_rs_ma = rsmk > rs_ma
    cond_rs_crit = rsmk > params.rs_crit
    cond_rs_high = rsmk > rsmk.shift(1).rolling(3, min_periods=3).max()
    cond_rs_cap = rsmk < params.rs_cap
    cond_regime = (
        ((roc_index > params.roc_threshold) & (corr > params.corr_min))
        | ((roc_index < params.roc_threshold) & (corr < -params.corr_min))
    )
    cond_rsi = (rsi.rolling(20, min_periods=20).min() < params.rsi_os) & (
        rsi > rsi.shift(1).rolling(2, min_periods=2).max()
    )
    cond_trend = ema5 > ma_trend.shift(5)

    out = pd.DataFrame(index=c.index)
    out["Open"] = df["Open"].astype(float)
    out["High"] = df["High"].astype(float)
    out["Low"] = df["Low"].astype(float)
    out["Close"] = c
    out["Volume"] = v
    out["rsmk"] = rsmk
    out["rs_ma"] = rs_ma
    out["rsi"] = rsi
    out["corr"] = corr
    out["roc_index"] = roc_index
    out["ema5"] = ema5
    out["ma_trend"] = ma_trend
    out["ma_trend_lag5"] = ma_trend.shift(5)
    out["vol_ma5"] = v.rolling(5, min_periods=5).mean()
    out["vol_ma50"] = v.rolling(50, min_periods=50).mean()
    out["cond_volume"] = cond_volume.fillna(False)
    out["cond_rs_ma"] = cond_rs_ma.fillna(False)
    out["cond_rs_crit"] = cond_rs_crit.fillna(False)
    out["cond_rs_high"] = cond_rs_high.fillna(False)
    out["cond_rs_cap"] = cond_rs_cap.fillna(False)
    out["cond_regime"] = cond_regime.fillna(False)
    out["cond_rsi"] = cond_rsi.fillna(False)
    out["cond_trend"] = cond_trend.fillna(False)
    return out


def apply_filters(
    ind: pd.DataFrame,
    params: StrategyParams,
    filters: Filters,
) -> pd.DataFrame:
    """Anade las columnas booleanas buy/sell aplicando los filtros elegidos."""
    out = ind.copy()
    buy = pd.Series(True, index=out.index)
    if filters.volume:
        buy &= out["cond_volume"]
    if filters.rs_above_ma:
        buy &= out["cond_rs_ma"]
    if filters.rs_above_crit:
        buy &= out["cond_rs_crit"]
    if filters.rs_new_high:
        buy &= out["cond_rs_high"]
    if filters.rs_cap:
        buy &= out["cond_rs_cap"]
    if filters.regime:
        buy &= out["cond_regime"]
    if filters.rsi:
        buy &= out["cond_rsi"]
    if filters.trend:
        buy &= out["cond_trend"]
    if not filters.entry_enabled():
        buy[:] = False

    sell = (out["rsmk"] < -params.rs_crit) & (out["corr"] > params.sell_corr)
    out["buy"] = buy.fillna(False).astype(bool)
    out["sell"] = sell.fillna(False).astype(bool)
    return out


def build_indicators(
    prices: dict[str, pd.DataFrame],
    bench_close: pd.Series,
    params: StrategyParams,
    filters: Filters,
) -> dict[str, pd.DataFrame]:
    return {
        sym: apply_filters(compute_indicators(df, bench_close, params), params, filters)
        for sym, df in prices.items()
    }


def _series_stats(eq: pd.Series, initial: float) -> dict:
    eq = eq.dropna()
    if len(eq) < 2:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "volatility": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
        }
    ret = eq.pct_change(fill_method=None).dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total = eq.iloc[-1] / initial - 1.0
    cagr = (eq.iloc[-1] / initial) ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else np.nan
    vol = ret.std() * math.sqrt(252.0)
    sharpe = ret.mean() / ret.std() * math.sqrt(252.0) if ret.std() > 0 else np.nan
    downside = ret[ret < 0].std() * math.sqrt(252.0)
    sortino = ret.mean() * 252.0 / downside if downside and downside > 0 else np.nan
    dd = eq / eq.cummax() - 1.0
    maxdd = float(dd.min())
    calmar = cagr / abs(maxdd) if maxdd < 0 else np.nan
    return {
        "total_return": float(total),
        "cagr": float(cagr),
        "volatility": float(vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": maxdd,
        "calmar": float(calmar),
    }


def _yearly_returns(eq: pd.Series) -> pd.Series:
    ret = eq.pct_change(fill_method=None).dropna()
    return (1.0 + ret).groupby(ret.index.year).prod() - 1.0


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "symbol",
            "entry_date",
            "entry_price",
            "exit_date",
            "exit_price",
            "shares",
            "entry_value",
            "exit_value",
            "commission",
            "pnl",
            "pnl_pct",
            "bars",
            "exit_reason",
            "entry_rs",
            "exit_rs",
            "entry_rsi",
            "entry_corr",
        ]
    )


def run_backtest(
    prices: dict[str, pd.DataFrame],
    params: StrategyParams,
    filters: Filters,
    start,
    end,
) -> BacktestResult:
    if params.index_symbol not in prices:
        raise ValueError(f"No hay datos para el indice de referencia {params.index_symbol}.")

    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if end <= start:
        raise ValueError("La fecha fin debe ser posterior a la fecha inicio.")

    bench_df = prices[params.index_symbol]
    bench_close = bench_df["Close"].astype(float)
    ind_all = build_indicators(prices, bench_close, params, filters)

    master = bench_df.index
    master = master[(master >= start) & (master <= end)]
    if len(master) < 2:
        raise ValueError("No hay suficientes datos de mercado en el rango elegido.")

    symbols = [s for s in prices if s in ind_all]
    arrays: dict[str, dict] = {}
    for sym in symbols:
        ind = ind_all[sym]
        has_bar = ind["Close"].reindex(master).notna().to_numpy()
        o = ind["Open"].reindex(master).to_numpy(dtype=float)
        h = ind["High"].reindex(master).to_numpy(dtype=float)
        c_ff = ind["Close"].reindex(master).ffill().to_numpy(dtype=float)
        buy = ind["buy"].reindex(master).fillna(False).to_numpy(dtype=bool)
        sell = ind["sell"].reindex(master).fillna(False).to_numpy(dtype=bool)
        arrays[sym] = {
            "has_bar": has_bar,
            "open": o,
            "high": h,
            "close": c_ff,
            "buy": buy,
            "sell": sell,
            "rsmk": ind["rsmk"].reindex(master).ffill().to_numpy(dtype=float),
            "rsi": ind["rsi"].reindex(master).ffill().to_numpy(dtype=float),
            "corr": ind["corr"].reindex(master).ffill().to_numpy(dtype=float),
        }

    bench_close_master = bench_close.reindex(master).ffill().to_numpy(dtype=float)
    symbol_order = {s: i for i, s in enumerate(symbols)}

    initial = float(params.initial_capital)
    pos_limit = max(1, int(params.positions))
    comm_rate = float(params.commission_pct) / 100.0
    slip_rate = float(params.slippage_bps) / 10_000.0
    lot = max(1, int(params.lot_size))
    min_shares = max(1, int(params.min_shares))
    min_value = float(params.min_position_value)

    cash = initial
    positions: dict[str, dict] = {}
    pending_exits: dict[str, str] = {}
    pending_entries: dict[str, dict] = {}
    blocked_until: dict[str, int] = {}
    trades: list[dict] = []
    daily: list[dict] = []

    def last_close(sym: str, idx: int) -> float:
        val = arrays[sym]["close"][idx]
        if not np.isfinite(val):
            j = idx
            while j >= 0 and not np.isfinite(arrays[sym]["close"][j]):
                j -= 1
            val = arrays[sym]["close"][j] if j >= 0 else np.nan
        return float(val)

    def close_position(sym: str, idx: int, price: float, reason: str) -> None:
        nonlocal cash
        pos = positions.pop(sym)
        gross = pos["shares"] * price
        comm = gross * comm_rate
        cash += gross - comm
        total_comm = pos["entry_comm"] + comm
        pnl = gross - comm - pos["entry_cost"] - pos["entry_comm"]
        cost_basis = pos["entry_cost"] + pos["entry_comm"]
        trades.append(
            {
                "symbol": sym,
                "entry_date": pos["entry_date"],
                "entry_price": pos["entry_price"],
                "exit_date": master[idx],
                "exit_price": float(price),
                "shares": int(pos["shares"]),
                "entry_value": round(pos["entry_cost"], 2),
                "exit_value": round(gross, 2),
                "commission": round(total_comm, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl / cost_basis * 100.0, 4) if cost_basis else 0.0,
                "bars": int(idx - pos["entry_idx"]),
                "exit_reason": reason,
                "entry_rs": round(float(pos["entry_rs"]), 4),
                "exit_rs": round(float(arrays[sym]["rsmk"][idx]), 4),
                "entry_rsi": round(float(pos["entry_rsi"]), 4),
                "entry_corr": round(float(pos["entry_corr"]), 4),
            }
        )
        if reason == "Objetivo":
            blocked_until[sym] = idx + max(0, int(params.reentry_delay))

    def open_position(sym: str, idx: int, price: float, signal: dict) -> None:
        nonlocal cash
        equity_now = cash + sum(
            p["shares"] * last_close(s, idx) for s, p in positions.items()
        )
        base = equity_now if params.compound else initial
        target_value = base / pos_limit
        exec_price = float(price) * (1.0 + slip_rate)
        denom = exec_price * (1.0 + comm_rate)
        wanted = math.floor(target_value / denom / lot) * lot
        affordable = math.floor(cash / denom / lot) * lot
        shares = int(min(wanted, affordable))
        if shares < min_shares or shares * exec_price < min_value:
            return
        cost = shares * exec_price
        comm = cost * comm_rate
        if cost + comm > cash:
            shares -= lot
            if shares < min_shares:
                return
            cost = shares * exec_price
            comm = cost * comm_rate
        cash -= cost + comm
        positions[sym] = {
            "entry_idx": idx,
            "entry_date": master[idx],
            "entry_price": exec_price,
            "shares": shares,
            "entry_cost": cost,
            "entry_comm": comm,
            "target": exec_price * (1.0 + float(params.profit_target) / 100.0),
            "entry_rs": signal.get("rsmk", np.nan),
            "entry_rsi": signal.get("rsi", np.nan),
            "entry_corr": signal.get("corr", np.nan),
        }

    last_i = len(master) - 1
    for i in range(len(master)):
        date = master[i]

        # 1) Salidas pendientes a la apertura.
        for sym in list(pending_exits):
            if sym not in positions:
                pending_exits.pop(sym, None)
                continue
            if not arrays[sym]["has_bar"][i]:
                continue
            raw_open = arrays[sym]["open"][i]
            if not np.isfinite(raw_open):
                continue
            reason = pending_exits.pop(sym)
            close_position(sym, i, float(raw_open) * (1.0 - slip_rate), reason)

        # 2) Entradas pendientes a la apertura (solo validas para la barra siguiente).
        if pending_entries:
            pending_entries = {
                s: v for s, v in pending_entries.items() if v["signal_idx"] == i - 1
            }
        if pending_entries and len(positions) < pos_limit:
            def priority_key(s: str):
                if params.priority == "rsmk":
                    rs = pending_entries[s].get("rsmk", np.nan)
                    return (-rs if np.isfinite(rs) else 1e12, symbol_order.get(s, 0))
                return (symbol_order.get(s, 0),)

            for sym in sorted(pending_entries, key=priority_key):
                if len(positions) >= pos_limit:
                    break
                if sym in positions:
                    pending_entries.pop(sym, None)
                    continue
                if not arrays[sym]["has_bar"][i]:
                    continue
                raw_open = arrays[sym]["open"][i]
                if not np.isfinite(raw_open):
                    continue
                signal = pending_entries.pop(sym)
                open_position(sym, i, float(raw_open), signal)

        # 3) Objetivo de beneficio (intradia).
        if filters.exit_target:
            for sym, pos in list(positions.items()):
                if sym in pending_exits:
                    continue
                if not arrays[sym]["has_bar"][i]:
                    continue
                high = arrays[sym]["high"][i]
                if not np.isfinite(high) or high < pos["target"]:
                    continue
                if params.target_exit == "target":
                    raw_open = arrays[sym]["open"][i]
                    px = max(float(raw_open), pos["target"]) if np.isfinite(raw_open) else pos["target"]
                    close_position(sym, i, px * (1.0 - slip_rate), "Objetivo")
                else:
                    pending_exits[sym] = "Objetivo"

        # 4) Senales de venta, salida temporal y nuevo maximo de RSMK al cierre.
        for sym, pos in positions.items():
            if sym in pending_exits:
                continue
            if not arrays[sym]["has_bar"][i]:
                continue
            if filters.exit_signal and arrays[sym]["sell"][i]:
                pending_exits[sym] = "Senal de venta"
                continue
            if filters.exit_time and (i - pos["entry_idx"]) >= max(1, int(params.time_exit)) - 1:
                pending_exits[sym] = "Salida temporal"

        # 5) Senales de compra al cierre para la apertura siguiente.
        if i < last_i:
            candidates = []
            for sym in symbols:
                if sym in positions or sym in pending_entries:
                    continue
                if blocked_until.get(sym, -1) > i:
                    continue
                if not arrays[sym]["has_bar"][i]:
                    continue
                if not arrays[sym]["buy"][i]:
                    continue
                candidates.append(sym)
            for sym in candidates:
                pending_entries[sym] = {
                    "rsmk": float(arrays[sym]["rsmk"][i]),
                    "rsi": float(arrays[sym]["rsi"][i]),
                    "corr": float(arrays[sym]["corr"][i]),
                    "signal_idx": i,
                }

        # 6) Valoracion de la cartera al cierre.
        invested = sum(p["shares"] * last_close(s, i) for s, p in positions.items())
        daily.append(
            {
                "date": date,
                "strategy": cash + invested,
                "cash": cash,
                "invested": invested,
                "positions": len(positions),
            }
        )

    equity = pd.DataFrame(daily).set_index("date")
    equity["benchmark"] = initial * bench_close_master / bench_close_master[0]

    norm_curves = []
    for sym in symbols:
        c = arrays[sym]["close"]
        valid = np.isfinite(c)
        if not valid.any():
            continue
        base_idx = int(np.argmax(valid))
        base = c[base_idx]
        if not np.isfinite(base) or base <= 0:
            continue
        norm_curves.append(c / base)
    if norm_curves:
        eqw = np.nanmean(np.vstack(norm_curves), axis=0)
        equity["equal_weight"] = initial * eqw
    else:
        equity["equal_weight"] = initial

    trades_df = pd.DataFrame(trades) if trades else _empty_trades()
    if not trades_df.empty:
        trades_df = trades_df.sort_values(["entry_date", "symbol"]).reset_index(drop=True)

    open_rows = []
    for sym, pos in positions.items():
        last_price = last_close(sym, last_i)
        gross = pos["shares"] * last_price
        pnl = gross - pos["entry_cost"] - pos["entry_comm"]
        cost_basis = pos["entry_cost"] + pos["entry_comm"]
        open_rows.append(
            {
                "symbol": sym,
                "entry_date": pos["entry_date"],
                "entry_price": pos["entry_price"],
                "exit_date": pd.NaT,
                "exit_price": round(float(last_price), 4),
                "shares": int(pos["shares"]),
                "entry_value": round(pos["entry_cost"], 2),
                "exit_value": round(gross, 2),
                "commission": round(pos["entry_comm"], 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl / cost_basis * 100.0, 4) if cost_basis else 0.0,
                "bars": int(last_i - pos["entry_idx"]),
                "exit_reason": "Abierta",
                "entry_rs": round(float(pos["entry_rs"]), 4),
                "exit_rs": round(float(arrays[sym]["rsmk"][last_i]), 4),
                "entry_rsi": round(float(pos["entry_rsi"]), 4),
                "entry_corr": round(float(pos["entry_corr"]), 4),
            }
        )
    open_df = pd.DataFrame(open_rows) if open_rows else _empty_trades()

    strategy_stats = _series_stats(equity["strategy"], initial)
    bench_stats = _series_stats(equity["benchmark"], initial)

    exposure = float((equity["invested"] / equity["strategy"]).clip(lower=0).mean())

    closed = trades_df
    if not closed.empty:
        wins = closed[closed["pnl"] > 0]
        losses = closed[closed["pnl"] < 0]
        gross_profit = float(wins["pnl"].sum())
        gross_loss = float(-losses["pnl"].sum())
        strategy_stats.update(
            {
                "trades": int(len(closed)),
                "win_rate": float(len(wins) / len(closed)),
                "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else np.nan,
                "avg_pnl_pct": float(closed["pnl_pct"].mean()),
                "best_pnl_pct": float(closed["pnl_pct"].max()),
                "worst_pnl_pct": float(closed["pnl_pct"].min()),
                "avg_bars": float(closed["bars"].mean()),
                "gross_profit": gross_profit,
                "gross_loss": gross_loss,
            }
        )
    else:
        strategy_stats.update(
            {
                "trades": 0,
                "win_rate": np.nan,
                "profit_factor": np.nan,
                "avg_pnl_pct": np.nan,
                "best_pnl_pct": np.nan,
                "worst_pnl_pct": np.nan,
                "avg_bars": np.nan,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
            }
        )
    strategy_stats["exposure"] = exposure
    strategy_stats["open_positions"] = int(len(open_df))

    yearly = pd.DataFrame(
        {
            params.index_symbol + " B&H": _yearly_returns(equity["benchmark"]) * 100.0,
            "Estrategia": _yearly_returns(equity["strategy"]) * 100.0,
        }
    )
    yearly.index.name = "Ano"
    yearly["Diferencia"] = yearly["Estrategia"] - yearly[params.index_symbol + " B&H"]

    return BacktestResult(
        equity=equity,
        trades=trades_df,
        open_positions=open_df,
        indicators=ind_all,
        metrics=strategy_stats,
        bench_metrics=bench_stats,
        yearly=yearly,
        params=asdict(params),
    )
