"""
hs300_top10/backtest/metrics.py

回测绩效指标的终端输出与格式化。
"""
from __future__ import annotations


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
