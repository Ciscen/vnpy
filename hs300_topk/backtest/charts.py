"""
hs300_topk/backtest/charts.py

Plotly 图表构建函数：权益曲线、超额收益、交易信号、个股详情等。
"""
from __future__ import annotations

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vnpy.alpha.strategy.backtesting import BacktestingEngine


def build_equity_chart(df: pl.DataFrame) -> go.Figure:
    """构建权益曲线 + 回撤四合一图"""
    dates = df["date"].to_list()
    balance = df["balance"].to_list()
    drawdown = df["drawdown"].to_list()
    net_pnl = df["net_pnl"].to_list()

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=["资金曲线 (Balance)", "回撤 (Drawdown)",
                        "每日盈亏 (Daily PnL)", "盈亏分布 (PnL Distribution)"],
        vertical_spacing=0.06,
    )

    fig.add_trace(
        go.Scatter(x=dates, y=balance, mode="lines", name="Balance"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=dates, y=drawdown, fill="tozeroy",
                   fillcolor="rgba(255,0,0,0.15)", mode="lines",
                   line=dict(color="red"), name="Drawdown"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Bar(x=dates, y=net_pnl, name="Daily PnL"),
        row=3, col=1,
    )
    fig.add_trace(
        go.Histogram(x=net_pnl, nbinsx=80, name="PnL Dist"),
        row=4, col=1,
    )

    fig.update_layout(
        height=1200, width=1100,
        title_text="HS300 Top-K 回测权益曲线",
        showlegend=False,
        plot_bgcolor="white",
    )
    for i in range(1, 5):
        fig.update_xaxes(showgrid=True, gridcolor="LightGray", row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="LightGray", row=i, col=1)

    return fig


def build_excess_return_chart(
    engine: BacktestingEngine,
    benchmark_symbol: str = "000300.SSE",
) -> go.Figure | None:
    """构建策略 vs 基准超额收益图（含成本）"""
    from vnpy.trader.object import BarData as _BarData

    benchmark_bars: list[_BarData] = engine.lab.load_bar_data(
        benchmark_symbol, engine.interval, engine.start, engine.end
    )
    if not benchmark_bars:
        print(f"  [评估] 基准数据 {benchmark_symbol} 不可用，跳过超额收益图")
        return None

    benchmark_prices = [b.close_price for b in benchmark_bars]
    df = engine.daily_df

    if len(benchmark_prices) != df.height:
        print(f"  [评估] 基准长度({len(benchmark_prices)})与回测天数({df.height})不匹配，跳过")
        return None

    perf = (
        df.with_columns(
            cumulative_return=pl.col("balance").pct_change().cum_sum(),
            cumulative_cost=(pl.col("commission") / pl.col("balance").shift(1)).cum_sum(),
        ).with_columns(
            benchmark_price=pl.Series(values=benchmark_prices, dtype=pl.Float64),
        ).with_columns(
            benchmark_return=pl.col("benchmark_price").pct_change().cum_sum(),
        ).with_columns(
            excess_return=pl.col("cumulative_return") - pl.col("benchmark_return"),
        ).with_columns(
            net_excess_return=pl.col("excess_return") - pl.col("cumulative_cost"),
        ).with_columns(
            excess_dd=pl.col("excess_return") - pl.col("excess_return").cum_max(),
            net_excess_dd=pl.col("net_excess_return") - pl.col("net_excess_return").cum_max(),
        )
    )

    fig = make_subplots(
        rows=5, cols=1,
        subplot_titles=[
            "累计收益 vs 基准", "超额收益 (Alpha)",
            "换手率", "Alpha 回撤", "Alpha 回撤 (含成本)",
        ],
        vertical_spacing=0.06,
    )

    p_dates = perf["date"].to_list()
    cum_ret = perf["cumulative_return"].to_list()
    cum_cost = perf["cumulative_cost"].to_list()
    cum_ret_net = (perf["cumulative_return"] - perf["cumulative_cost"]).to_list()
    bench_ret = perf["benchmark_return"].to_list()
    excess_ret = perf["excess_return"].to_list()
    net_excess_ret = perf["net_excess_return"].to_list()
    turnover = (df["turnover"] / df["balance"].shift(1)).to_list()
    excess_dd_list = perf["excess_dd"].to_list()
    net_excess_dd_list = perf["net_excess_dd"].to_list()
    d_dates = df["date"].to_list()

    fig.add_trace(go.Scatter(x=p_dates, y=cum_ret,
                             mode="lines", name="策略"), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_dates, y=cum_ret_net,
                             mode="lines", name="策略(含成本)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_dates, y=bench_ret,
                             mode="lines", name="沪深300"), row=1, col=1)

    fig.add_trace(go.Scatter(x=p_dates, y=excess_ret,
                             mode="lines", name="Alpha"), row=2, col=1)
    fig.add_trace(go.Scatter(x=p_dates, y=net_excess_ret,
                             mode="lines", name="Alpha(含成本)"), row=2, col=1)

    fig.add_trace(go.Scatter(x=d_dates, y=turnover,
                             name="换手率"), row=3, col=1)

    fig.add_trace(go.Scatter(x=p_dates, y=excess_dd_list,
                             fill="tozeroy", mode="lines",
                             name="Alpha DD"), row=4, col=1)
    fig.add_trace(go.Scatter(x=p_dates, y=net_excess_dd_list,
                             fill="tozeroy", mode="lines",
                             name="Alpha DD(含成本)"), row=5, col=1)

    fig.update_layout(
        height=1500, width=1200, title_text="HS300 Top-K 超额收益分析",
        plot_bgcolor="white", paper_bgcolor="white",
    )
    for i in range(1, 6):
        fig.update_xaxes(showgrid=True, gridcolor="LightGray", row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="LightGray", row=i, col=1)

    return fig


def build_pnl_chart(df: pl.DataFrame) -> go.Figure:
    """构建月度盈亏柱状图"""
    monthly = (
        df.with_columns(
            pl.col("date").cast(pl.Utf8).str.slice(0, 7).alias("month")
        )
        .group_by("month")
        .agg(pl.col("net_pnl").sum().alias("monthly_pnl"))
        .sort("month")
    )

    months = monthly["month"].to_list()
    monthly_pnl = monthly["monthly_pnl"].to_list()
    colors = ["green" if v >= 0 else "red" for v in monthly_pnl]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=months,
            y=monthly_pnl,
            marker_color=colors,
            name="月度盈亏",
        )
    )
    fig.update_layout(
        title_text="HS300 Top-K 月度盈亏",
        xaxis_title="月份",
        yaxis_title="净盈亏 (元)",
        height=500, width=1000,
        plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="LightGray"),
        yaxis=dict(showgrid=True, gridcolor="LightGray"),
    )
    return fig


REASON_COLORS: dict[str, str] = {
    "signal_buy": "#2ca02c",
    "rebalance_out": "#1f77b4",
    "stop_loss": "#d62728",
    "abs_stop_cap": "#e377c2",
    "trailing_tp": "#ff7f0e",
    "max_hold": "#9467bd",
    "portfolio_stop": "#8c564b",
    "market_filter": "#7f7f7f",
    "unknown": "#bcbd22",
}

REASON_LABELS: dict[str, str] = {
    "signal_buy": "买入（信号）",
    "rebalance_out": "卖出（调仓换股）",
    "stop_loss": "卖出（止损）",
    "abs_stop_cap": "卖出（绝对止损）",
    "trailing_tp": "卖出（追踪止盈）",
    "max_hold": "卖出（超时退出）",
    "portfolio_stop": "卖出（组合止损）",
    "market_filter": "卖出（市场过滤）",
    "unknown": "卖出（未知）",
}


def classify_reason(reason_str: str) -> str:
    """将详细原因字符串归类到大类"""
    if not reason_str:
        return "unknown"
    for key in ["abs_stop_cap", "stop_loss", "trailing_tp", "max_hold",
                "portfolio_stop", "market_filter", "rebalance_out", "signal_buy"]:
        if key in reason_str:
            return key
    return "unknown"


def detect_direction_values(tlog: pl.DataFrame) -> tuple[str, str]:
    """自动检测 direction 列中买入/卖出的值"""
    vals = set(tlog["direction"].unique().to_list())
    if "Long" in vals:
        return "Long", "Short"
    return "多", "空"


def build_trade_signal_chart(
    engine: BacktestingEngine,
    trade_log_df: pl.DataFrame | None,
) -> go.Figure | None:
    """构建交易信号可视化图

    Panel 1: 资金曲线 + 买卖点标记
    Panel 2: 持仓数量随时间变化
    Panel 3: 卖出原因统计（按月分布堆叠柱状图）
    Panel 4: 单笔交易盈亏散点图（按卖出原因着色）
    """
    if trade_log_df is None or trade_log_df.is_empty():
        print("  [报告] 无交易日志，跳过交易信号图")
        return None

    df = engine.daily_df

    tlog = trade_log_df.with_columns(
        pl.col("datetime").str.slice(0, 10).alias("date_str"),
        pl.col("reason").map_elements(classify_reason, return_dtype=pl.Utf8).alias("reason_cat"),
    )

    long_val, short_val = detect_direction_values(tlog)

    buys = tlog.filter(pl.col("direction") == long_val)
    sells = tlog.filter(pl.col("direction") == short_val)

    buy_by_date = buys.group_by("date_str").agg(
        pl.col("vt_symbol").count().alias("count"),
        pl.col("vt_symbol").alias("symbols"),
        pl.col("price").alias("prices"),
    )
    sell_by_date = sells.group_by("date_str").agg(
        pl.col("vt_symbol").count().alias("count"),
        pl.col("vt_symbol").alias("symbols"),
        pl.col("reason").alias("reasons"),
        pl.col("pnl_pct").alias("pnls"),
    )

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[
            "资金曲线与交易时点",
            "每日持仓股票数",
            "月度卖出原因分布",
            "单笔交易盈亏（按卖出原因着色）",
        ],
        vertical_spacing=0.07,
        row_heights=[0.3, 0.2, 0.25, 0.25],
    )

    # ── Panel 1: 资金曲线 + 买卖标记 ──
    _dates_list = df["date"].to_list()
    _balance_list = df["balance"].to_list()
    fig.add_trace(
        go.Scatter(x=_dates_list, y=_balance_list, mode="lines",
                   line=dict(color="#333", width=1.5), name="Balance",
                   showlegend=False),
        row=1, col=1,
    )

    balance_map = {str(d): b for d, b in zip(_dates_list, _balance_list)}

    if not buy_by_date.is_empty():
        bx, by, bt = [], [], []
        for row in buy_by_date.iter_rows(named=True):
            d = row["date_str"]
            if d in balance_map:
                bx.append(d)
                by.append(balance_map[d])
                syms = row["symbols"][:5]
                extra = f" +{row['count'] - 5}..." if row["count"] > 5 else ""
                bt.append(f"买入 {row['count']} 只: {', '.join(syms)}{extra}")
        fig.add_trace(
            go.Scatter(x=bx, y=by, mode="markers", name="买入",
                       marker=dict(symbol="triangle-up", size=8, color="#2ca02c", opacity=0.8),
                       text=bt, hovertemplate="%{text}<br>日期: %{x}<br>资金: %{y:,.0f}<extra></extra>"),
            row=1, col=1,
        )

    if not sell_by_date.is_empty():
        sx, sy, st = [], [], []
        for row in sell_by_date.iter_rows(named=True):
            d = row["date_str"]
            if d in balance_map:
                sx.append(d)
                sy.append(balance_map[d])
                reasons = row["reasons"][:5]
                syms = row["symbols"][:5]
                detail_parts = [f"{s}({r})" for s, r in zip(syms, reasons)]
                extra = f" +{row['count'] - 5}..." if row["count"] > 5 else ""
                st.append(f"卖出 {row['count']} 只: {', '.join(detail_parts)}{extra}")
        fig.add_trace(
            go.Scatter(x=sx, y=sy, mode="markers", name="卖出",
                       marker=dict(symbol="triangle-down", size=8, color="#d62728", opacity=0.8),
                       text=st, hovertemplate="%{text}<br>日期: %{x}<br>资金: %{y:,.0f}<extra></extra>"),
            row=1, col=1,
        )

    # ── Panel 2: 持仓数量（基于逐笔交易重建） ──
    dates = [str(d) for d in df["date"].to_list()]
    holdings: set[str] = set()
    trade_events: dict[str, int] = {}
    for row in tlog.sort("datetime").iter_rows(named=True):
        d = row["date_str"]
        sym = row["vt_symbol"]
        if row["direction"] == long_val:
            holdings.add(sym)
        else:
            holdings.discard(sym)
        trade_events[d] = len(holdings)

    pos_count_series = []
    running_count = 0
    for d in dates:
        if d in trade_events:
            running_count = trade_events[d]
        pos_count_series.append(running_count)

    fig.add_trace(
        go.Scatter(x=dates, y=pos_count_series, mode="lines",
                   fill="tozeroy", fillcolor="rgba(31,119,180,0.15)",
                   line=dict(color="#1f77b4", width=1), name="持仓数",
                   showlegend=False),
        row=2, col=1,
    )

    # ── Panel 3: 月度卖出原因堆叠柱状图 ──
    if not sells.is_empty():
        monthly_reasons = (
            sells.with_columns(
                pl.col("datetime").str.slice(0, 7).alias("month"),
            )
            .group_by(["month", "reason_cat"])
            .agg(pl.len().alias("count"))
            .sort("month")
        )

        all_reasons = ["stop_loss", "abs_stop_cap", "trailing_tp", "max_hold",
                       "rebalance_out", "portfolio_stop", "market_filter"]
        for reason_key in all_reasons:
            subset = monthly_reasons.filter(pl.col("reason_cat") == reason_key)
            if not subset.is_empty():
                fig.add_trace(
                    go.Bar(
                        x=subset["month"].to_list(),
                        y=subset["count"].to_list(),
                        name=REASON_LABELS.get(reason_key, reason_key),
                        marker_color=REASON_COLORS.get(reason_key, "#999"),
                    ),
                    row=3, col=1,
                )
        fig.update_layout(barmode="stack")

    # ── Panel 4: 单笔交易盈亏散点图 ──
    if not sells.is_empty() and "pnl_pct" in sells.columns:
        all_reasons = ["stop_loss", "abs_stop_cap", "trailing_tp", "max_hold",
                       "rebalance_out", "portfolio_stop", "market_filter"]
        for reason_key in all_reasons:
            subset = sells.filter(pl.col("reason_cat") == reason_key)
            if subset.is_empty():
                continue
            fig.add_trace(
                go.Scatter(
                    x=subset["date_str"].to_list(),
                    y=subset["pnl_pct"].to_list(),
                    mode="markers",
                    name=REASON_LABELS.get(reason_key, reason_key),
                    marker=dict(
                        color=REASON_COLORS.get(reason_key, "#999"),
                        size=5,
                        opacity=0.7,
                    ),
                    text=[f"{s} ({r})" for s, r in zip(
                        subset["vt_symbol"].to_list(), subset["reason"].to_list())],
                    hovertemplate="%{text}<br>盈亏: %{y:.1f}%<extra></extra>",
                    showlegend=False,
                ),
                row=4, col=1,
            )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=4, col=1)

    fig.update_layout(
        height=1600, width=1200,
        title_text="HS300 Top-K 交易信号分析",
        plot_bgcolor="white", paper_bgcolor="white",
    )
    for i in range(1, 5):
        fig.update_xaxes(showgrid=True, gridcolor="LightGray", row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="LightGray", row=i, col=1)
    fig.update_yaxes(title_text="资金", row=1, col=1)
    fig.update_yaxes(title_text="持仓数", row=2, col=1)
    fig.update_yaxes(title_text="笔数", row=3, col=1)
    fig.update_yaxes(title_text="盈亏 (%)", row=4, col=1)

    return fig


def build_stock_detail_chart(
    engine: BacktestingEngine,
    trade_log_df: pl.DataFrame,
    vt_symbol: str,
) -> go.Figure | None:
    """构建个股维度交易详情图

    Panel 1: 价格走势 + 买入/卖出标记（hover 显示原因和盈亏）
    Panel 2: 持仓量变化
    """
    from vnpy.trader.object import BarData as _BarData

    bars: list[_BarData] = engine.lab.load_bar_data(
        vt_symbol, engine.interval, engine.start, engine.end
    )
    if not bars:
        return None

    bar_dates = [str(b.datetime)[:10] for b in bars]
    bar_open = [b.open_price for b in bars]
    bar_high = [b.high_price for b in bars]
    bar_low = [b.low_price for b in bars]
    bar_close = [b.close_price for b in bars]
    bar_vol = [b.volume for b in bars]

    long_val, short_val = detect_direction_values(trade_log_df)
    stock_trades = trade_log_df.filter(pl.col("vt_symbol") == vt_symbol).sort("datetime")
    if stock_trades.is_empty():
        return None

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            f"{vt_symbol} 价格与交易标记",
            "成交量",
            "持仓量变化",
        ],
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25],
        shared_xaxes=True,
    )

    # Panel 1: K 线
    fig.add_trace(
        go.Candlestick(
            x=bar_dates, open=bar_open, high=bar_high, low=bar_low, close=bar_close,
            name="K线", increasing_line_color="#d62728", decreasing_line_color="#2ca02c",
            showlegend=False,
        ),
        row=1, col=1,
    )

    buys = stock_trades.filter(pl.col("direction") == long_val)
    sells = stock_trades.filter(pl.col("direction") == short_val)

    if not buys.is_empty():
        buy_dates_list = buys["datetime"].str.slice(0, 10).to_list()
        buy_prices_list = buys["price"].to_list()
        fig.add_trace(
            go.Scatter(
                x=buy_dates_list,
                y=buy_prices_list,
                mode="markers",
                name="买入",
                marker=dict(symbol="triangle-up", size=12, color="#2ca02c",
                            line=dict(width=1, color="black")),
                text=[f"买入 vol={v:.0f}" for v in buys["volume"].to_list()],
                hovertemplate="%{text}<br>价格: %{y:.2f}<br>日期: %{x}<extra></extra>",
            ),
            row=1, col=1,
        )

    if not sells.is_empty():
        sell_texts = []
        for row in sells.iter_rows(named=True):
            sell_texts.append(
                f"{row['reason']}<br>"
                f"盈亏: {row['pnl_pct']:.1f}%, 持仓: {row['hold_days']}天"
            )
        sell_dates_list = sells["datetime"].str.slice(0, 10).to_list()
        sell_prices_list = sells["price"].to_list()
        fig.add_trace(
            go.Scatter(
                x=sell_dates_list,
                y=sell_prices_list,
                mode="markers",
                name="卖出",
                marker=dict(symbol="triangle-down", size=12, color="#d62728",
                            line=dict(width=1, color="black")),
                text=sell_texts,
                hovertemplate="%{text}<br>价格: %{y:.2f}<br>日期: %{x}<extra></extra>",
            ),
            row=1, col=1,
        )

        # 持仓区间高亮
        for i, buy_row in enumerate(buys.iter_rows(named=True)):
            buy_date = buy_row["datetime"][:10]
            matched_sell = sells.filter(
                (pl.col("datetime") > buy_row["datetime"])
            ).head(1)
            if matched_sell.is_empty():
                continue
            sell_row = matched_sell.row(0, named=True)
            sell_date = sell_row["datetime"][:10]
            pnl = sell_row["pnl_pct"]
            color = "rgba(0,200,0,0.08)" if pnl >= 0 else "rgba(255,0,0,0.08)"
            fig.add_vrect(
                x0=buy_date, x1=sell_date,
                fillcolor=color, layer="below", line_width=0,
                row=1, col=1,
            )

    # Panel 2: 成交量
    colors = ["#d62728" if c >= o else "#2ca02c" for o, c in zip(bar_open, bar_close)]
    fig.add_trace(
        go.Bar(x=bar_dates, y=bar_vol, marker_color=colors, name="成交量",
               showlegend=False, opacity=0.7),
        row=2, col=1,
    )

    # Panel 3: 持仓量
    pos = 0.0
    pos_dates = []
    pos_vals = []
    trade_map: dict[str, float] = {}
    for row in stock_trades.iter_rows(named=True):
        d = row["datetime"][:10]
        if row["direction"] == long_val:
            trade_map[d] = trade_map.get(d, 0) + row["volume"]
        else:
            trade_map[d] = trade_map.get(d, 0) - row["volume"]

    current_pos = 0.0
    for d in bar_dates:
        if d in trade_map:
            current_pos += trade_map[d]
            current_pos = max(0, current_pos)
        pos_dates.append(d)
        pos_vals.append(current_pos)

    fig.add_trace(
        go.Scatter(x=pos_dates, y=pos_vals, mode="lines",
                   fill="tozeroy", fillcolor="rgba(255,165,0,0.15)",
                   line=dict(color="#ff7f0e", width=1.5), name="持仓",
                   showlegend=False),
        row=3, col=1,
    )

    fig.update_layout(
        height=900, width=1200,
        title_text=f"{vt_symbol} 交易详情",
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_rangeslider_visible=False,
    )
    for i in range(1, 4):
        fig.update_xaxes(showgrid=True, gridcolor="LightGray", row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="LightGray", row=i, col=1)
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_yaxes(title_text="持仓量", row=3, col=1)

    return fig
