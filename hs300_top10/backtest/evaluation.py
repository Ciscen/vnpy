"""
hs300_top10/backtest/evaluation.py

回测绩效统计、可视化与报告导出。
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vnpy.alpha.strategy.backtesting import BacktestingEngine


# ══════════════════════════════════════════════════════════
# 终端输出
# ══════════════════════════════════════════════════════════

def print_metrics(stats: dict) -> None:
    """格式化输出回测绩效指标。"""
    print()
    print("=" * 58)
    print("  HS300 Top-10 策略回测绩效")
    print("=" * 58)

    kv = [
        ("回测区间",       f"{stats.get('start_date', '')} ~ {stats.get('end_date', '')}"),
        ("总交易日",       stats.get("total_days", 0)),
        ("盈利交易日",     stats.get("profit_days", 0)),
        ("亏损交易日",     stats.get("loss_days", 0)),
        ("",              ""),
        ("起始资金",       f"{stats.get('capital', 0):>14,.2f}"),
        ("结束资金",       f"{stats.get('end_balance', 0):>14,.2f}"),
        ("总收益率",       f"{stats.get('total_return', 0):>10.2f}%"),
        ("年化收益率",     f"{stats.get('annual_return', 0):>10.2f}%"),
        ("",              ""),
        ("最大回撤",       f"{stats.get('max_drawdown', 0):>14,.2f}"),
        ("百分比最大回撤", f"{stats.get('max_ddpercent', 0):>10.2f}%"),
        ("最长回撤天数",   stats.get("max_drawdown_duration", 0)),
        ("",              ""),
        ("Sharpe Ratio",  f"{stats.get('sharpe_ratio', 0):>10.2f}"),
        ("收益回撤比",     f"{stats.get('return_drawdown_ratio', 0):>10.2f}"),
        ("",              ""),
        ("总盈亏",         f"{stats.get('total_net_pnl', 0):>14,.2f}"),
        ("总手续费",       f"{stats.get('total_commission', 0):>14,.2f}"),
        ("总成交额",       f"{stats.get('total_turnover', 0):>14,.2f}"),
        ("总成交笔数",     stats.get("total_trade_count", 0)),
        ("",              ""),
        ("日均盈亏",       f"{stats.get('daily_net_pnl', 0):>14,.2f}"),
        ("日均成交额",     f"{stats.get('daily_turnover', 0):>14,.2f}"),
    ]

    for label, value in kv:
        if label == "":
            print("-" * 58)
        else:
            print(f"  {label:<16s}  {value}")

    print("=" * 58)


# ══════════════════════════════════════════════════════════
# 交互式图表
# ══════════════════════════════════════════════════════════

def show_charts(engine: BacktestingEngine, benchmark_symbol: str | None = None) -> None:
    """展示权益曲线和绩效图（浏览器弹出）。"""
    try:
        engine.show_chart()
    except Exception as e:
        print(f"  [评估] 权益曲线绘制失败: {e}")

    if benchmark_symbol:
        try:
            engine.show_performance(benchmark_symbol)
        except Exception as e:
            print(f"  [评估] 超额收益图绘制失败: {e}")


# ══════════════════════════════════════════════════════════
# 报告导出（文件持久化）
# ══════════════════════════════════════════════════════════

def export_report(
    engine: BacktestingEngine,
    stats: dict,
    output_dir: str | Path,
    *,
    version_label: str = "",
) -> None:
    """导出完整回测报告到指定目录。

    生成文件
    --------
    output_dir/
    ├── statistics.json          # 绩效统计指标
    ├── daily_pnl.csv            # 逐日盈亏明细
    ├── trades.csv               # 全部成交记录
    ├── equity_curve.html        # 权益曲线图（Plotly 交互式）
    └── daily_pnl_chart.html     # 每日盈亏分布图
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. 统计指标 JSON
    stats_path = out / "statistics.json"
    serializable = {}
    for k, v in stats.items():
        if hasattr(v, "isoformat"):
            serializable[k] = v.isoformat()
        elif hasattr(v, "item"):
            serializable[k] = v.item()
        else:
            serializable[k] = v
    stats_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [报告] 统计指标 -> {stats_path}")

    # 2. 逐日盈亏 CSV
    daily_path = out / "daily_pnl.csv"
    try:
        df: pl.DataFrame = engine.daily_df
        df.write_csv(daily_path)
        print(f"  [报告] 逐日盈亏 ({df.height} 行) -> {daily_path}")
    except Exception as e:
        print(f"  [报告] 逐日盈亏导出失败: {e}")

    # 3. 成交记录 CSV（含触发原因）
    trades_path = out / "trades.csv"
    trade_log_df = None
    try:
        strategy = engine.strategy
        if hasattr(strategy, "trade_log") and strategy.trade_log:
            trade_log_df = pl.DataFrame(strategy.trade_log)
            trade_log_df.write_csv(trades_path)
            print(f"  [报告] 成交记录 ({trade_log_df.height} 笔, 含触发原因) -> {trades_path}")
        else:
            trades = engine.get_all_trades()
            if trades:
                trade_rows = []
                for t in trades:
                    trade_rows.append({
                        "datetime": str(t.datetime),
                        "vt_symbol": t.vt_symbol,
                        "direction": t.direction.value,
                        "price": t.price,
                        "volume": t.volume,
                        "reason": "",
                    })
                trade_log_df = pl.DataFrame(trade_rows)
                trade_log_df.write_csv(trades_path)
                print(f"  [报告] 成交记录 ({len(trades)} 笔) -> {trades_path}")
            else:
                print("  [报告] 无成交记录")
    except Exception as e:
        print(f"  [报告] 成交记录导出失败: {e}")

    # 4. 权益曲线 HTML
    equity_path = out / "equity_curve.html"
    try:
        df = engine.daily_df
        fig = _build_equity_chart(df)
        fig.write_html(str(equity_path), include_plotlyjs="cdn")
        print(f"  [报告] 权益曲线图 -> {equity_path}")
    except Exception as e:
        print(f"  [报告] 权益曲线图导出失败: {e}")

    # 5. 每日盈亏分布 HTML
    pnl_chart_path = out / "daily_pnl_chart.html"
    try:
        df = engine.daily_df
        fig = _build_pnl_chart(df)
        fig.write_html(str(pnl_chart_path), include_plotlyjs="cdn")
        print(f"  [报告] 盈亏分布图 -> {pnl_chart_path}")
    except Exception as e:
        print(f"  [报告] 盈亏分布图导出失败: {e}")

    # 6. 超额收益图 HTML
    excess_path = out / "excess_return.html"
    try:
        fig = _build_excess_return_chart(engine, benchmark_symbol="000300.SSE")
        if fig is not None:
            fig.write_html(str(excess_path), include_plotlyjs="cdn")
            print(f"  [报告] 超额收益图 -> {excess_path}")
    except Exception as e:
        print(f"  [报告] 超额收益图导出失败: {e}")

    # 7. 交易信号图 HTML（资金曲线 + 买卖点标记 + 触发原因）
    trade_chart_path = out / "trade_signals.html"
    try:
        fig = _build_trade_signal_chart(engine, trade_log_df)
        if fig is not None:
            fig.write_html(str(trade_chart_path), include_plotlyjs="cdn")
            print(f"  [报告] 交易信号图 -> {trade_chart_path}")
    except Exception as e:
        print(f"  [报告] 交易信号图导出失败: {e}")

    # 8. 个股交易详情图（top 10 高频交易股票）
    try:
        if trade_log_df is not None and not trade_log_df.is_empty():
            export_stock_details(engine, trade_log_df, output_dir, top_n=10)
    except Exception as e:
        print(f"  [报告] 个股详情图导出失败: {e}")

    # 9. 综合仪表盘 HTML（多 Tab 汇总页面）
    dashboard_path = out / "dashboard.html"
    try:
        html = _build_dashboard_html(engine, stats, trade_log_df, version_label=version_label)
        if html:
            dashboard_path.write_text(html, encoding="utf-8")
            print(f"  [报告] 综合仪表盘 -> {dashboard_path}")
    except Exception as e:
        print(f"  [报告] 综合仪表盘导出失败: {e}")
        import traceback
        traceback.print_exc()


# ══════════════════════════════════════════════════════════
# 图表构建
# ══════════════════════════════════════════════════════════

def _build_equity_chart(df: pl.DataFrame) -> go.Figure:
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
        title_text="HS300 Top-10 回测权益曲线",
        showlegend=False,
        plot_bgcolor="white",
    )
    for i in range(1, 5):
        fig.update_xaxes(showgrid=True, gridcolor="LightGray", row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="LightGray", row=i, col=1)

    return fig


def _build_excess_return_chart(
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
        height=1500, width=1200, title_text="HS300 Top-10 超额收益分析",
        plot_bgcolor="white", paper_bgcolor="white",
    )
    for i in range(1, 6):
        fig.update_xaxes(showgrid=True, gridcolor="LightGray", row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="LightGray", row=i, col=1)

    return fig


def _build_pnl_chart(df: pl.DataFrame) -> go.Figure:
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
        title_text="HS300 Top-10 月度盈亏",
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


def _classify_reason(reason_str: str) -> str:
    """将详细原因字符串归类到大类"""
    if not reason_str:
        return "unknown"
    for key in ["abs_stop_cap", "stop_loss", "trailing_tp", "max_hold",
                "portfolio_stop", "market_filter", "rebalance_out", "signal_buy"]:
        if key in reason_str:
            return key
    return "unknown"


def _detect_direction_values(tlog: pl.DataFrame) -> tuple[str, str]:
    """自动检测 direction 列中买入/卖出的值"""
    vals = set(tlog["direction"].unique().to_list())
    if "Long" in vals:
        return "Long", "Short"
    return "多", "空"


def _build_trade_signal_chart(
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
        pl.col("reason").map_elements(_classify_reason, return_dtype=pl.Utf8).alias("reason_cat"),
    )

    long_val, short_val = _detect_direction_values(tlog)

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
        title_text="HS300 Top-10 交易信号分析",
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


def _build_stock_detail_chart(
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

    long_val, short_val = _detect_direction_values(trade_log_df)
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


def _build_dashboard_html(
    engine: BacktestingEngine,
    stats: dict,
    trade_log_df: pl.DataFrame | None,
    *,
    version_label: str = "",
) -> str | None:
    """构建综合仪表盘：多 Tab 汇总 HTML 页面。

    Tab 页面:
      1. 策略总览: 关键指标 + 权益曲线 + 回撤
      2. 持仓汇总: 所有操作过的股票表格（累计收益、胜率、交易次数等）
      3. 交易明细: 完整交易日志（可排序/搜索）
      4. 个股K线: 逐只股票的 K线 + 买卖标记（下拉选择）
      5. 收益归因: 按股票、按月份、按卖出原因的收益分解
    """
    if trade_log_df is None or trade_log_df.is_empty():
        return None

    import plotly.io as pio

    long_val, short_val = _detect_direction_values(trade_log_df)
    daily_df = engine.daily_df

    import json as _json

    # ── 数据准备 ──
    buys = trade_log_df.filter(pl.col("direction") == long_val).sort("datetime")
    sells = trade_log_df.filter(pl.col("direction") == short_val).sort("datetime")

    # 个股统计
    stock_stats = _compute_stock_stats(buys, sells, long_val, short_val)
    stock_stats_json = _json.dumps(stock_stats, ensure_ascii=False)

    # 交易明细
    trades_json = trade_log_df.sort("datetime").write_json()

    # ── Plotly 图表 ──
    equity_fig = _build_equity_chart(daily_df)
    equity_html = pio.to_html(equity_fig, include_plotlyjs=False, full_html=False, div_id="equity-chart")

    pnl_fig = _build_pnl_chart(daily_df)
    pnl_html = pio.to_html(pnl_fig, include_plotlyjs=False, full_html=False, div_id="monthly-pnl-chart")

    excess_fig = _build_excess_return_chart(engine, benchmark_symbol="000300.SSE")
    excess_html = ""
    if excess_fig is not None:
        excess_html = pio.to_html(excess_fig, include_plotlyjs=False, full_html=False, div_id="excess-chart")

    # 收益归因图表
    attribution_html = _build_attribution_charts(sells, daily_df, long_val)

    # 个股 K 线数据（全部有交易的股票）
    all_traded_symbols = sorted(set(trade_log_df["vt_symbol"].unique().to_list()))
    stock_charts_data = _prepare_stock_charts_data(engine, trade_log_df, all_traded_symbols, long_val, short_val)

    # ── 组装 HTML ──
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HS300 Top-10 策略回测仪表盘</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: #f5f6fa; color: #2c3e50; }}
.header {{ background: linear-gradient(135deg, #2c3e50, #3498db); color: white;
           padding: 24px 32px; }}
.header h1 {{ font-size: 24px; font-weight: 600; }}
.header .subtitle {{ font-size: 14px; opacity: 0.85; margin-top: 4px; }}
.tab-bar {{ display: flex; background: #fff; border-bottom: 2px solid #e0e0e0;
            padding: 0 16px; position: sticky; top: 0; z-index: 100;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
.tab-btn {{ padding: 14px 24px; cursor: pointer; border: none; background: none;
            font-size: 14px; font-weight: 500; color: #666;
            border-bottom: 3px solid transparent; transition: all 0.2s; }}
.tab-btn:hover {{ color: #3498db; background: #f8f9fa; }}
.tab-btn.active {{ color: #3498db; border-bottom-color: #3498db; }}
.tab-content {{ display: none; padding: 24px; max-width: 1400px; margin: 0 auto; }}
.tab-content.active {{ display: block; }}
.card {{ background: #fff; border-radius: 8px; padding: 20px; margin-bottom: 20px;
         box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
.card h3 {{ font-size: 16px; color: #34495e; margin-bottom: 12px; padding-bottom: 8px;
            border-bottom: 1px solid #ecf0f1; }}
.metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                  gap: 16px; margin-bottom: 20px; }}
.metric {{ background: #fff; border-radius: 8px; padding: 16px; text-align: center;
           box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
.metric .value {{ font-size: 28px; font-weight: 700; }}
.metric .label {{ font-size: 12px; color: #95a5a6; margin-top: 4px; }}
.positive {{ color: #27ae60; }}
.negative {{ color: #e74c3c; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{ background: #f8f9fa; padding: 10px 12px; text-align: left; font-weight: 600;
      border-bottom: 2px solid #e0e0e0; cursor: pointer; user-select: none;
      position: sticky; top: 0; }}
th:hover {{ background: #ecf0f1; }}
td {{ padding: 8px 12px; border-bottom: 1px solid #f0f0f0; }}
tr:hover {{ background: #f8f9fa; }}
.table-wrap {{ max-height: 600px; overflow-y: auto; }}
.search-box {{ padding: 8px 12px; border: 1px solid #ddd; border-radius: 4px;
               width: 300px; margin-bottom: 12px; font-size: 14px; }}
.search-box:focus {{ outline: none; border-color: #3498db; }}
select.stock-select {{ padding: 8px 12px; border: 1px solid #ddd; border-radius: 4px;
                       font-size: 14px; min-width: 200px; }}
.pill {{ display: inline-block; padding: 2px 8px; border-radius: 10px;
         font-size: 11px; font-weight: 600; }}
.pill-profit {{ background: #e8f8f0; color: #27ae60; }}
.pill-loss {{ background: #fde8e8; color: #e74c3c; }}
</style>
</head>
<body>
<div class="header">
  <h1>HS300 Top-10 策略回测仪表盘</h1>
  <div class="subtitle">回测区间: {stats.get('start_date','')} ~ {stats.get('end_date','')} | 初始资金: {stats.get('capital',0):,.0f} | 策略版本: {version_label or '未知'}</div>
</div>

<div class="tab-bar">
  <button class="tab-btn active" onclick="switchTab('overview')">策略总览</button>
  <button class="tab-btn" onclick="switchTab('positions')">持仓汇总</button>
  <button class="tab-btn" onclick="switchTab('trades')">交易明细</button>
  <button class="tab-btn" onclick="switchTab('kline')">个股K线</button>
  <button class="tab-btn" onclick="switchTab('attribution')">收益归因</button>
</div>

<!-- ══════════ Tab 1: 策略总览 ══════════ -->
<div id="tab-overview" class="tab-content active">
  <div class="metrics-grid">
    <div class="metric">
      <div class="value {_css_sign(stats.get('total_return',0))}">{stats.get('total_return',0):.2f}%</div>
      <div class="label">总收益率</div>
    </div>
    <div class="metric">
      <div class="value {_css_sign(stats.get('annual_return',0))}">{stats.get('annual_return',0):.2f}%</div>
      <div class="label">年化收益率</div>
    </div>
    <div class="metric">
      <div class="value negative">{stats.get('max_ddpercent',0):.2f}%</div>
      <div class="label">最大回撤</div>
    </div>
    <div class="metric">
      <div class="value">{stats.get('sharpe_ratio',0):.2f}</div>
      <div class="label">Sharpe Ratio</div>
    </div>
    <div class="metric">
      <div class="value">{stats.get('return_drawdown_ratio',0):.2f}</div>
      <div class="label">收益回撤比</div>
    </div>
    <div class="metric">
      <div class="value">{stats.get('total_trade_count',0)}</div>
      <div class="label">总成交笔数</div>
    </div>
    <div class="metric">
      <div class="value">{stats.get('profit_days',0)} / {stats.get('total_days',0)}</div>
      <div class="label">盈利天数 / 总天数</div>
    </div>
    <div class="metric">
      <div class="value">{stats.get('total_commission',0):,.0f}</div>
      <div class="label">总手续费</div>
    </div>
  </div>
  <div class="card">{equity_html}</div>
  {"<div class='card'>" + excess_html + "</div>" if excess_html else ""}
  <div class="card">{pnl_html}</div>
</div>

<!-- ══════════ Tab 2: 持仓汇总 ══════════ -->
<div id="tab-positions" class="tab-content">
  <div class="card">
    <h3>所有操作过的股票汇总（共 <span id="stock-count"></span> 只）</h3>
    <input type="text" class="search-box" id="stock-search" placeholder="搜索股票代码..." oninput="filterStockTable()">
    <div class="table-wrap">
      <table id="stock-table">
        <thead>
          <tr>
            <th onclick="sortTable('stock-table',0)">股票代码</th>
            <th onclick="sortTable('stock-table',1,'num')">买入次数</th>
            <th onclick="sortTable('stock-table',2,'num')">卖出次数</th>
            <th onclick="sortTable('stock-table',3,'num')">累计盈亏(%)</th>
            <th onclick="sortTable('stock-table',4,'num')">平均盈亏(%)</th>
            <th onclick="sortTable('stock-table',5,'num')">胜率(%)</th>
            <th onclick="sortTable('stock-table',6,'num')">最大盈利(%)</th>
            <th onclick="sortTable('stock-table',7,'num')">最大亏损(%)</th>
            <th onclick="sortTable('stock-table',8,'num')">平均持仓天数</th>
            <th onclick="sortTable('stock-table',9)">卖出原因分布</th>
          </tr>
        </thead>
        <tbody id="stock-tbody"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- ══════════ Tab 3: 交易明细 ══════════ -->
<div id="tab-trades" class="tab-content">
  <div class="card">
    <h3>完整交易日志（共 <span id="trade-count"></span> 笔）</h3>
    <input type="text" class="search-box" id="trade-search" placeholder="搜索股票代码、日期或原因..." oninput="filterTradeTable()">
    <div class="table-wrap" style="max-height:700px">
      <table id="trade-table">
        <thead>
          <tr>
            <th onclick="sortTable('trade-table',0)">日期</th>
            <th onclick="sortTable('trade-table',1)">股票代码</th>
            <th onclick="sortTable('trade-table',2)">方向</th>
            <th onclick="sortTable('trade-table',3,'num')">价格</th>
            <th onclick="sortTable('trade-table',4,'num')">数量</th>
            <th onclick="sortTable('trade-table',5)">原因</th>
            <th onclick="sortTable('trade-table',6,'num')">入场价</th>
            <th onclick="sortTable('trade-table',7,'num')">盈亏(%)</th>
            <th onclick="sortTable('trade-table',8,'num')">持仓天数</th>
          </tr>
        </thead>
        <tbody id="trade-tbody"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- ══════════ Tab 4: 个股K线 ══════════ -->
<div id="tab-kline" class="tab-content">
  <div class="card">
    <h3>个股交易详情</h3>
    <select class="stock-select" id="kline-select" onchange="renderStockChart()">
      <option value="">-- 选择股票 --</option>
      {"".join(f'<option value="{s}">{s}</option>' for s in all_traded_symbols)}
    </select>
    <div id="kline-container" style="margin-top:16px"></div>
  </div>
</div>

<!-- ══════════ Tab 5: 收益归因 ══════════ -->
<div id="tab-attribution" class="tab-content">
  {attribution_html}
</div>

<script>
// ── 数据 ──
const stockStats = {stock_stats_json};
const allTrades = {trades_json};
const stockCharts = {stock_charts_data};

// ── Tab 切换 ──
function switchTab(name) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
  if (name === 'kline') {{ setTimeout(() => {{ window.dispatchEvent(new Event('resize')); }}, 100); }}
}}

// ── 持仓汇总表 ──
function renderStockTable() {{
  const tbody = document.getElementById('stock-tbody');
  const rows = stockStats.map(s => {{
    const pnlClass = s.total_pnl_pct >= 0 ? 'pill-profit' : 'pill-loss';
    return `<tr>
      <td><strong>${{s.vt_symbol}}</strong></td>
      <td>${{s.buy_count}}</td>
      <td>${{s.sell_count}}</td>
      <td><span class="pill ${{pnlClass}}">${{s.total_pnl_pct.toFixed(1)}}%</span></td>
      <td>${{s.avg_pnl_pct.toFixed(1)}}%</td>
      <td>${{s.win_rate.toFixed(1)}}%</td>
      <td class="positive">${{s.max_profit.toFixed(1)}}%</td>
      <td class="negative">${{s.max_loss.toFixed(1)}}%</td>
      <td>${{s.avg_hold_days.toFixed(1)}}</td>
      <td style="font-size:11px">${{s.reason_dist}}</td>
    </tr>`;
  }});
  tbody.innerHTML = rows.join('');
  document.getElementById('stock-count').textContent = stockStats.length;
}}

// ── 交易明细表 ──
function renderTradeTable() {{
  const tbody = document.getElementById('trade-tbody');
  const rows = allTrades.map(t => {{
    const dir = t.direction === '{long_val}' ? '买入' : '卖出';
    const dirColor = t.direction === '{long_val}' ? '#27ae60' : '#e74c3c';
    const pnl = t.pnl_pct != null ? t.pnl_pct.toFixed(1) + '%' : '-';
    const pnlClass = (t.pnl_pct || 0) >= 0 ? 'positive' : 'negative';
    return `<tr>
      <td>${{t.datetime.substring(0,10)}}</td>
      <td><strong>${{t.vt_symbol}}</strong></td>
      <td style="color:${{dirColor}};font-weight:600">${{dir}}</td>
      <td>${{t.price.toFixed(2)}}</td>
      <td>${{Math.round(t.volume)}}</td>
      <td style="font-size:12px">${{t.reason || '-'}}</td>
      <td>${{t.entry_price ? t.entry_price.toFixed(2) : '-'}}</td>
      <td class="${{pnlClass}}">${{pnl}}</td>
      <td>${{t.hold_days != null ? t.hold_days : '-'}}</td>
    </tr>`;
  }});
  tbody.innerHTML = rows.join('');
  document.getElementById('trade-count').textContent = allTrades.length;
}}

// ── 搜索过滤 ──
function filterStockTable() {{
  const q = document.getElementById('stock-search').value.toLowerCase();
  document.querySelectorAll('#stock-tbody tr').forEach(tr => {{
    tr.style.display = tr.textContent.toLowerCase().includes(q) ? '' : 'none';
  }});
}}
function filterTradeTable() {{
  const q = document.getElementById('trade-search').value.toLowerCase();
  document.querySelectorAll('#trade-tbody tr').forEach(tr => {{
    tr.style.display = tr.textContent.toLowerCase().includes(q) ? '' : 'none';
  }});
}}

// ── 表格排序 ──
let sortState = {{}};
function sortTable(tableId, colIdx, type) {{
  const key = tableId + '_' + colIdx;
  sortState[key] = !sortState[key];
  const asc = sortState[key];
  const tbody = document.getElementById(tableId).querySelector('tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  rows.sort((a, b) => {{
    let va = a.cells[colIdx].textContent.replace(/[%,元]/g, '').trim();
    let vb = b.cells[colIdx].textContent.replace(/[%,元]/g, '').trim();
    if (type === 'num') {{
      va = parseFloat(va) || 0;
      vb = parseFloat(vb) || 0;
    }}
    if (va < vb) return asc ? -1 : 1;
    if (va > vb) return asc ? 1 : -1;
    return 0;
  }});
  rows.forEach(r => tbody.appendChild(r));
}}

// ── 个股 K 线 ──
function renderStockChart() {{
  const sym = document.getElementById('kline-select').value;
  const container = document.getElementById('kline-container');
  if (!sym || !stockCharts[sym]) {{ container.innerHTML = '<p>请选择一只股票</p>'; return; }}
  const d = stockCharts[sym];
  const traces = [];

  traces.push({{
    type: 'candlestick', x: d.dates, open: d.open, high: d.high, low: d.low, close: d.close,
    name: 'K线', increasing: {{line:{{color:'#e74c3c'}}}}, decreasing: {{line:{{color:'#27ae60'}}}},
    xaxis: 'x', yaxis: 'y'
  }});

  if (d.buy_dates.length) {{
    traces.push({{
      type: 'scatter', mode: 'markers', x: d.buy_dates, y: d.buy_prices,
      name: '买入', text: d.buy_texts,
      marker: {{symbol: 'triangle-up', size: 11, color: '#27ae60', line:{{width:1,color:'#000'}}}},
      hovertemplate: '%{{text}}<br>价格: %{{y:.2f}}<extra></extra>',
      xaxis: 'x', yaxis: 'y'
    }});
  }}
  if (d.sell_dates.length) {{
    traces.push({{
      type: 'scatter', mode: 'markers', x: d.sell_dates, y: d.sell_prices,
      name: '卖出', text: d.sell_texts,
      marker: {{symbol: 'triangle-down', size: 11, color: '#e74c3c', line:{{width:1,color:'#000'}}}},
      hovertemplate: '%{{text}}<br>价格: %{{y:.2f}}<extra></extra>',
      xaxis: 'x', yaxis: 'y'
    }});
  }}

  traces.push({{
    type: 'bar', x: d.dates, y: d.volume,
    marker: {{color: d.vol_colors, opacity: 0.5}},
    name: '成交量', xaxis: 'x2', yaxis: 'y2'
  }});

  const layout = {{
    height: 600, grid: {{rows: 2, columns: 1, pattern: 'independent', roworder: 'top to bottom'}},
    xaxis: {{rangeslider: {{visible: false}}, domain: [0,1]}},
    xaxis2: {{domain: [0,1]}},
    yaxis: {{title: '价格', domain: [0.35, 1]}},
    yaxis2: {{title: '成交量', domain: [0, 0.28]}},
    title: sym + ' 交易详情',
    plot_bgcolor: 'white', paper_bgcolor: 'white',
    showlegend: true, legend: {{x:0, y:1}},
    shapes: d.shapes || []
  }};
  Plotly.newPlot(container, traces, layout, {{responsive: true}});
}}

// ── 初始化 ──
renderStockTable();
renderTradeTable();
</script>
</body>
</html>"""
    return html


def _css_sign(v: float) -> str:
    return "positive" if v >= 0 else "negative"


def _compute_stock_stats(
    buys: pl.DataFrame,
    sells: pl.DataFrame,
    long_val: str,
    short_val: str,
) -> list[dict]:
    """计算每只股票的交易汇总统计"""
    all_symbols = sorted(set(
        buys["vt_symbol"].unique().to_list() + sells["vt_symbol"].unique().to_list()
    ))

    results = []
    for sym in all_symbols:
        sym_buys = buys.filter(pl.col("vt_symbol") == sym)
        sym_sells = sells.filter(pl.col("vt_symbol") == sym)

        buy_count = sym_buys.height
        sell_count = sym_sells.height

        if sell_count > 0 and "pnl_pct" in sym_sells.columns:
            pnls = sym_sells["pnl_pct"].to_list()
            pnls = [p for p in pnls if p is not None]
            total_pnl = sum(pnls) if pnls else 0.0
            avg_pnl = total_pnl / len(pnls) if pnls else 0.0
            wins = sum(1 for p in pnls if p > 0)
            win_rate = (wins / len(pnls) * 100) if pnls else 0.0
            max_profit = max(pnls) if pnls else 0.0
            max_loss = min(pnls) if pnls else 0.0
        else:
            total_pnl = avg_pnl = win_rate = max_profit = max_loss = 0.0

        if sell_count > 0 and "hold_days" in sym_sells.columns:
            hds = sym_sells["hold_days"].to_list()
            hds = [h for h in hds if h is not None]
            avg_hold = sum(hds) / len(hds) if hds else 0.0
        else:
            avg_hold = 0.0

        reason_dist = ""
        if sell_count > 0 and "reason" in sym_sells.columns:
            reasons = sym_sells["reason"].to_list()
            from collections import Counter
            rc = Counter(_classify_reason(r) for r in reasons if r)
            reason_dist = ", ".join(
                f"{REASON_LABELS.get(k, k)}:{v}" for k, v in rc.most_common(3)
            )

        results.append({
            "vt_symbol": sym,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "total_pnl_pct": round(total_pnl, 2),
            "avg_pnl_pct": round(avg_pnl, 2),
            "win_rate": round(win_rate, 1),
            "max_profit": round(max_profit, 2),
            "max_loss": round(max_loss, 2),
            "avg_hold_days": round(avg_hold, 1),
            "reason_dist": reason_dist,
        })

    results.sort(key=lambda x: x["total_pnl_pct"], reverse=True)
    return results


def _build_attribution_charts(
    sells: pl.DataFrame,
    daily_df: pl.DataFrame,
    long_val: str,
) -> str:
    """构建收益归因分析的 HTML 片段"""
    import plotly.io as pio

    html_parts = []

    if sells.is_empty() or "pnl_pct" not in sells.columns:
        return '<div class="card"><h3>无卖出数据</h3></div>'

    # 1. 按股票的累计盈亏 Top/Bottom 10
    stock_pnl = (
        sells
        .group_by("vt_symbol")
        .agg(
            pl.col("pnl_pct").sum().alias("total_pnl"),
            pl.col("pnl_pct").count().alias("count"),
        )
        .sort("total_pnl", descending=True)
    )

    top10 = stock_pnl.head(10)
    bottom10 = stock_pnl.tail(10).sort("total_pnl")
    combined = pl.concat([top10, bottom10.sort("total_pnl", descending=True)])

    fig1 = go.Figure()
    c_symbols = combined["vt_symbol"].to_list()
    c_pnl = combined["total_pnl"].to_list()
    colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in c_pnl]
    fig1.add_trace(go.Bar(
        x=c_symbols, y=c_pnl,
        marker_color=colors, name="累计盈亏",
        text=[f"{v:.1f}%" for v in c_pnl],
        textposition="outside",
    ))
    fig1.update_layout(
        title="Top/Bottom 10 股票累计盈亏", height=400, width=1200,
        plot_bgcolor="white", yaxis_title="累计盈亏 (%)",
    )
    html_parts.append(f'<div class="card">{pio.to_html(fig1, include_plotlyjs=False, full_html=False)}</div>')

    # 2. 按月份收益
    monthly_pnl = (
        sells
        .with_columns(pl.col("datetime").str.slice(0, 7).alias("month"))
        .group_by("month")
        .agg(
            pl.col("pnl_pct").sum().alias("total_pnl"),
            pl.col("pnl_pct").count().alias("count"),
            (pl.col("pnl_pct") > 0).sum().alias("win_count"),
        )
        .sort("month")
    )
    monthly_pnl = monthly_pnl.with_columns(
        (pl.col("win_count") / pl.col("count") * 100).alias("win_rate")
    )

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    m_months = monthly_pnl["month"].to_list()
    m_pnl = monthly_pnl["total_pnl"].to_list()
    m_wr = monthly_pnl["win_rate"].to_list()
    colors2 = ["#27ae60" if v >= 0 else "#e74c3c" for v in m_pnl]
    fig2.add_trace(go.Bar(
        x=m_months, y=m_pnl,
        marker_color=colors2, name="月度累计盈亏(%)",
    ), secondary_y=False)
    fig2.add_trace(go.Scatter(
        x=m_months, y=m_wr,
        mode="lines+markers", name="月度胜率(%)",
        line=dict(color="#3498db", width=2),
    ), secondary_y=True)
    fig2.update_layout(
        title="月度收益与胜率", height=400, width=1200,
        plot_bgcolor="white",
    )
    fig2.update_yaxes(title_text="盈亏 (%)", secondary_y=False)
    fig2.update_yaxes(title_text="胜率 (%)", secondary_y=True)
    html_parts.append(f'<div class="card">{pio.to_html(fig2, include_plotlyjs=False, full_html=False)}</div>')

    # 3. 按卖出原因的盈亏分布
    reason_pnl = (
        sells
        .with_columns(
            pl.col("reason").map_elements(_classify_reason, return_dtype=pl.Utf8).alias("reason_cat"),
        )
        .group_by("reason_cat")
        .agg(
            pl.col("pnl_pct").sum().alias("total_pnl"),
            pl.col("pnl_pct").mean().alias("avg_pnl"),
            pl.col("pnl_pct").count().alias("count"),
            (pl.col("pnl_pct") > 0).sum().alias("win_count"),
        )
        .sort("total_pnl", descending=True)
    )

    fig3 = go.Figure()
    for row in reason_pnl.iter_rows(named=True):
        label = REASON_LABELS.get(row["reason_cat"], row["reason_cat"])
        color = REASON_COLORS.get(row["reason_cat"], "#999")
        fig3.add_trace(go.Bar(
            x=[label], y=[row["total_pnl"]], name=label,
            marker_color=color,
            text=[f'{row["total_pnl"]:.1f}% ({row["count"]}笔, 均{row["avg_pnl"]:.1f}%)'],
            textposition="outside",
        ))
    fig3.update_layout(
        title="按卖出原因的盈亏统计", height=400, width=1200,
        plot_bgcolor="white", yaxis_title="累计盈亏 (%)", showlegend=False,
    )
    html_parts.append(f'<div class="card">{pio.to_html(fig3, include_plotlyjs=False, full_html=False)}</div>')

    # 4. 持仓天数 vs 盈亏散点图
    if "hold_days" in sells.columns:
        s_hold = sells["hold_days"].to_list()
        s_pnl = sells["pnl_pct"].to_list()
        s_sym = sells["vt_symbol"].to_list()
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=s_hold, y=s_pnl,
            mode="markers", name="单笔交易",
            marker=dict(
                color=s_pnl,
                colorscale="RdYlGn", cmin=-10, cmax=10,
                size=6, opacity=0.6, colorbar=dict(title="盈亏%"),
            ),
            text=s_sym,
            hovertemplate="%{text}<br>持仓: %{x}天<br>盈亏: %{y:.1f}%<extra></extra>",
        ))
        fig4.add_hline(y=0, line_dash="dash", line_color="gray")
        fig4.update_layout(
            title="持仓天数 vs 盈亏", height=400, width=1200,
            plot_bgcolor="white", xaxis_title="持仓天数", yaxis_title="盈亏 (%)",
        )
        html_parts.append(f'<div class="card">{pio.to_html(fig4, include_plotlyjs=False, full_html=False)}</div>')

    return "\n".join(html_parts)


def _prepare_stock_charts_data(
    engine: BacktestingEngine,
    trade_log_df: pl.DataFrame,
    symbols: list[str],
    long_val: str,
    short_val: str,
) -> str:
    """为所有交易过的股票准备 K 线 + 交易标记数据（JSON 格式）"""
    import json as _json
    from vnpy.trader.object import BarData as _BarData

    data = {}
    for sym in symbols:
        try:
            bars: list[_BarData] = engine.lab.load_bar_data(
                sym, engine.interval, engine.start, engine.end
            )
            if not bars:
                continue

            dates = [str(b.datetime)[:10] for b in bars]
            opens = [b.open_price for b in bars]
            highs = [b.high_price for b in bars]
            lows = [b.low_price for b in bars]
            closes = [b.close_price for b in bars]
            volumes = [b.volume for b in bars]
            vol_colors = ["#e74c3c" if c >= o else "#27ae60" for o, c in zip(opens, closes)]

            sym_trades = trade_log_df.filter(pl.col("vt_symbol") == sym).sort("datetime")
            sym_buys = sym_trades.filter(pl.col("direction") == long_val)
            sym_sells = sym_trades.filter(pl.col("direction") == short_val)

            buy_dates = [r[:10] for r in sym_buys["datetime"].to_list()]
            buy_prices = sym_buys["price"].to_list()
            buy_texts = [f"买入 vol={v:.0f}" for v in sym_buys["volume"].to_list()]

            sell_dates = [r[:10] for r in sym_sells["datetime"].to_list()]
            sell_prices = sym_sells["price"].to_list()
            sell_texts = []
            for row in sym_sells.iter_rows(named=True):
                pnl = row.get("pnl_pct")
                hd = row.get("hold_days")
                sell_texts.append(
                    f"{row.get('reason','')}<br>盈亏:{pnl:.1f}% 持仓:{hd}天"
                    if pnl is not None else row.get('reason', '')
                )

            shapes = []
            for i, buy_row in enumerate(sym_buys.iter_rows(named=True)):
                bd = buy_row["datetime"][:10]
                matched = sym_sells.filter(pl.col("datetime") > buy_row["datetime"]).head(1)
                if matched.is_empty():
                    continue
                sr = matched.row(0, named=True)
                sd = sr["datetime"][:10]
                pnl_val = sr.get("pnl_pct", 0) or 0
                fc = "rgba(0,200,0,0.07)" if pnl_val >= 0 else "rgba(255,0,0,0.07)"
                shapes.append({
                    "type": "rect", "xref": "x", "yref": "paper",
                    "x0": bd, "x1": sd, "y0": 0, "y1": 1,
                    "fillcolor": fc, "line": {"width": 0}, "layer": "below"
                })

            data[sym] = {
                "dates": dates, "open": opens, "high": highs, "low": lows,
                "close": closes, "volume": volumes, "vol_colors": vol_colors,
                "buy_dates": buy_dates, "buy_prices": buy_prices, "buy_texts": buy_texts,
                "sell_dates": sell_dates, "sell_prices": sell_prices, "sell_texts": sell_texts,
                "shapes": shapes,
            }
        except Exception:
            continue

    return _json.dumps(data, ensure_ascii=False)


def export_stock_details(
    engine: BacktestingEngine,
    trade_log_df: pl.DataFrame,
    output_dir: str,
    top_n: int = 10,
) -> None:
    """导出交易最频繁的 top_n 只个股详情图"""
    out = Path(output_dir) / "stock_details"
    out.mkdir(parents=True, exist_ok=True)

    long_val, _ = _detect_direction_values(trade_log_df)

    freq = (
        trade_log_df
        .filter(pl.col("direction") == long_val)
        .group_by("vt_symbol")
        .agg(pl.len().alias("trade_count"))
        .sort("trade_count", descending=True)
        .head(top_n)
    )

    symbols = freq["vt_symbol"].to_list()
    print(f"  [报告] 生成 {len(symbols)} 只高频交易股票详情图 ...")

    for sym in symbols:
        try:
            fig = _build_stock_detail_chart(engine, trade_log_df, sym)
            if fig is not None:
                path = out / f"{sym.replace('.', '_')}.html"
                fig.write_html(str(path), include_plotlyjs="cdn")
                print(f"    -> {path}")
        except Exception as e:
            print(f"    [!] {sym} 详情图失败: {e}")
