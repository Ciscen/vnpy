"""
hs300_top10/backtest/evaluation.py

回测绩效的编排层：统计输出、图表生成、报告导出。

图表构建 → charts.py
仪表盘 HTML → dashboard.py
终端指标 → metrics.py
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from vnpy.alpha.strategy.backtesting import BacktestingEngine

from hs300_top10.backtest.metrics import print_metrics
from hs300_top10.backtest.charts import (
    build_equity_chart,
    build_excess_return_chart,
    build_pnl_chart,
    build_trade_signal_chart,
    build_stock_detail_chart,
)
from hs300_top10.backtest.dashboard import build_dashboard_html


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
    ├── statistics.json
    ├── daily_pnl.csv
    ├── trades.csv
    ├── equity_curve.html
    ├── daily_pnl_chart.html
    ├── excess_return.html
    ├── trade_signals.html
    ├── stock_details/
    └── dashboard.html
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
    _export_chart(build_equity_chart, [engine.daily_df], out / "equity_curve.html", "权益曲线图")

    # 5. 每日盈亏分布 HTML
    _export_chart(build_pnl_chart, [engine.daily_df], out / "daily_pnl_chart.html", "盈亏分布图")

    # 6. 超额收益图 HTML
    _export_chart(
        build_excess_return_chart,
        [engine],
        out / "excess_return.html",
        "超额收益图",
        kwargs={"benchmark_symbol": "000300.SSE"},
    )

    # 7. 交易信号图 HTML
    _export_chart(
        build_trade_signal_chart,
        [engine, trade_log_df],
        out / "trade_signals.html",
        "交易信号图",
    )

    # 8. 个股交易详情图
    try:
        if trade_log_df is not None and not trade_log_df.is_empty():
            export_stock_details(engine, trade_log_df, output_dir, top_n=10)
    except Exception as e:
        print(f"  [报告] 个股详情图导出失败: {e}")

    # 9. 综合仪表盘 HTML
    dashboard_path = out / "dashboard.html"
    try:
        html = build_dashboard_html(engine, stats, trade_log_df, version_label=version_label)
        if html:
            dashboard_path.write_text(html, encoding="utf-8")
            print(f"  [报告] 综合仪表盘 -> {dashboard_path}")
    except Exception as e:
        print(f"  [报告] 综合仪表盘导出失败: {e}")
        import traceback
        traceback.print_exc()


def export_stock_details(
    engine: BacktestingEngine,
    trade_log_df: pl.DataFrame,
    output_dir: str | Path,
    top_n: int = 10,
) -> None:
    """导出高频交易股票的 K 线详情图。"""
    out = Path(output_dir) / "stock_details"
    out.mkdir(parents=True, exist_ok=True)

    sell_counts = (
        trade_log_df.filter(pl.col("direction").is_in(["Short", "空"]))
        .group_by("vt_symbol")
        .agg(pl.len().alias("cnt"))
        .sort("cnt", descending=True)
        .head(top_n)
    )

    print(f"  [报告] 生成 {min(top_n, sell_counts.height)} 只高频交易股票详情图 ...")
    for row in sell_counts.iter_rows(named=True):
        symbol = row["vt_symbol"]
        try:
            fig = build_stock_detail_chart(engine, trade_log_df, symbol)
            if fig is not None:
                fname = symbol.replace(".", "_") + ".html"
                path = out / fname
                fig.write_html(str(path), include_plotlyjs="cdn")
                print(f"    -> {path}")
        except Exception as e:
            print(f"    [跳过] {symbol}: {e}")


def _export_chart(builder, args, path, label, kwargs=None):
    """通用图表导出辅助函数。"""
    try:
        fig = builder(*args, **(kwargs or {}))
        if fig is not None:
            fig.write_html(str(path), include_plotlyjs="cdn")
            print(f"  [报告] {label} -> {path}")
    except Exception as e:
        print(f"  [报告] {label}导出失败: {e}")
