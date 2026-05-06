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

    # 3. 成交记录 CSV
    trades_path = out / "trades.csv"
    try:
        trades = engine.get_all_trades()
        if trades:
            trade_rows = []
            for t in trades:
                trade_rows.append({
                    "datetime": str(t.datetime),
                    "vt_symbol": t.vt_symbol,
                    "direction": t.direction.value,
                    "offset": t.offset.value,
                    "price": t.price,
                    "volume": t.volume,
                    "tradeid": t.tradeid,
                })
            trade_df = pl.DataFrame(trade_rows)
            trade_df.write_csv(trades_path)
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


# ══════════════════════════════════════════════════════════
# 图表构建
# ══════════════════════════════════════════════════════════

def _build_equity_chart(df: pl.DataFrame) -> go.Figure:
    """构建权益曲线 + 回撤四合一图"""
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=["资金曲线 (Balance)", "回撤 (Drawdown)",
                        "每日盈亏 (Daily PnL)", "盈亏分布 (PnL Distribution)"],
        vertical_spacing=0.06,
    )

    fig.add_trace(
        go.Scatter(x=df["date"], y=df["balance"], mode="lines", name="Balance"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["date"], y=df["drawdown"], fill="tozeroy",
                   fillcolor="rgba(255,0,0,0.15)", mode="lines",
                   line=dict(color="red"), name="Drawdown"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Bar(x=df["date"], y=df["net_pnl"], name="Daily PnL"),
        row=3, col=1,
    )
    fig.add_trace(
        go.Histogram(x=df["net_pnl"], nbinsx=80, name="PnL Dist"),
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

    fig.add_trace(go.Scatter(x=perf["date"], y=perf["cumulative_return"],
                             mode="lines", name="策略"), row=1, col=1)
    fig.add_trace(go.Scatter(x=perf["date"],
                             y=perf["cumulative_return"] - perf["cumulative_cost"],
                             mode="lines", name="策略(含成本)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=perf["date"], y=perf["benchmark_return"],
                             mode="lines", name="沪深300"), row=1, col=1)

    fig.add_trace(go.Scatter(x=perf["date"], y=perf["excess_return"],
                             mode="lines", name="Alpha"), row=2, col=1)
    fig.add_trace(go.Scatter(x=perf["date"], y=perf["net_excess_return"],
                             mode="lines", name="Alpha(含成本)"), row=2, col=1)

    fig.add_trace(go.Scatter(x=df["date"],
                             y=df["turnover"] / df["balance"].shift(1),
                             name="换手率"), row=3, col=1)

    fig.add_trace(go.Scatter(x=perf["date"], y=perf["excess_dd"],
                             fill="tozeroy", mode="lines",
                             name="Alpha DD"), row=4, col=1)
    fig.add_trace(go.Scatter(x=perf["date"], y=perf["net_excess_dd"],
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

    colors = ["green" if v >= 0 else "red" for v in monthly["monthly_pnl"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=monthly["month"],
            y=monthly["monthly_pnl"],
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
