"""
hs300_topk/run_regime_grid2.py

第二轮网格搜索: R6 (DD15%) 底座 + 市场 MA 过滤组合。
每个区间独立训练信号，修复全周期数据复用 bug。
"""
from __future__ import annotations

import logging
import time
from dataclasses import replace
from datetime import datetime

from vnpy.trader.constant import Interval
from vnpy.alpha.strategy import BacktestingEngine

from hs300_topk.data.loader import get_lab, discover_symbols
from hs300_topk.model.rolling_trainer import rolling_train
from hs300_topk.strategy.hs300_topk_strategy import HS300Top10Strategy
from hs300_topk.strategy.config import OPTIMIZED_V14

LAB_PATH = "./lab/hs300"
CAPITAL = 100_000
WEEKLY_LABEL = "dynamic_regime"
LAG_DAYS = 3

PERIODS = {
    "bear": {"data_start": "2016-04-30", "data_end": "2023-12-31",
             "bt_start": "2022-01-01", "bt_end": "2023-12-31"},
    "bull": {"data_start": "2016-04-30", "data_end": "2026-04-30",
             "bt_start": "2024-05-01", "bt_end": "2026-04-30"},
    "full": {"data_start": "2016-04-30", "data_end": "2026-04-30",
             "bt_start": "2022-01-01", "bt_end": "2026-04-30"},
}

GRID: dict[str, dict] = {
    "baseline": dict(),
    "R6_dd15": dict(max_portfolio_drawdown=0.15, drawdown_cooldown_days=10),
    "R6+MA20": dict(max_portfolio_drawdown=0.15, drawdown_cooldown_days=10,
                    use_market_filter=True, market_ma_period=20),
    "R6+MA40": dict(max_portfolio_drawdown=0.15, drawdown_cooldown_days=10,
                    use_market_filter=True, market_ma_period=40),
    "R6+MA60": dict(max_portfolio_drawdown=0.15, drawdown_cooldown_days=10,
                    use_market_filter=True, market_ma_period=60),
    "MA60_only": dict(use_market_filter=True, market_ma_period=60),
    "R6+MA60+cool20": dict(max_portfolio_drawdown=0.15, drawdown_cooldown_days=20,
                           use_market_filter=True, market_ma_period=60),
}


def build_config(overrides: dict):
    return replace(OPTIMIZED_V14, **overrides)


def run_single(signal_df, cfg, period):
    lab = get_lab(LAB_PATH)
    vt_symbols = discover_symbols(LAB_PATH)
    needs_bench = cfg.use_market_filter or cfg.regime_filter or cfg.max_portfolio_drawdown > 0
    if needs_bench and cfg.market_benchmark not in vt_symbols:
        vt_symbols = vt_symbols + [cfg.market_benchmark]
        lab.add_contract_setting(cfg.market_benchmark, 0, 0, 1, 0.01)
    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=datetime.fromisoformat(period["bt_start"]),
        end=datetime.fromisoformat(period["bt_end"]),
        capital=CAPITAL,
    )
    setting = {
        k: v for k, v in cfg.to_dict().items()
        if k not in ("version", "description")
        and not k.startswith("xgb_") and k != "train_years"
    }
    engine.add_strategy(HS300Top10Strategy, setting, signal_df)
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    return engine.calculate_statistics()


def main():
    t0 = time.time()
    signals = {}

    for pname, pconfig in PERIODS.items():
        key = f"{pconfig['data_start']}_{pconfig['data_end']}_{pconfig['bt_start']}_{pconfig['bt_end']}"
        if key in signals:
            continue
        print(f"\n{'='*60}")
        print(f"  训练 {pname}: bt={pconfig['bt_start']}~{pconfig['bt_end']}")
        print(f"{'='*60}")
        sig, _ = rolling_train(
            lab_path=LAB_PATH,
            data_start=pconfig["data_start"],
            data_end=pconfig["data_end"],
            backtest_start=pconfig["bt_start"],
            backtest_end=pconfig["bt_end"],
            weekly_label=WEEKLY_LABEL, lag_days=LAG_DAYS,
        )
        signals[pname] = sig
        print(f"  信号: {sig.shape[0]} 行")

    logging.disable(logging.CRITICAL)
    all_results = {}
    for pname, pconfig in PERIODS.items():
        print(f"\n  回测 {pname} ...")
        all_results[pname] = {}
        for gname, overrides in GRID.items():
            cfg = build_config(overrides)
            all_results[pname][gname] = run_single(signals[pname], cfg, pconfig)
    logging.disable(logging.NOTSET)

    print("\n" + "=" * 120)
    print("  第二轮网格搜索: R6底座 + MA过滤组合")
    print("=" * 120)
    for pname in PERIODS:
        print(f"\n  [{pname}]")
        header = (
            f"  {'方案':<20s} {'Sharpe':>8s} {'年化%':>8s} {'总收益%':>10s} "
            f"{'最大回撤%':>10s} {'收益回撤比':>10s} {'交易笔数':>8s}"
        )
        print(header)
        print("  " + "-" * 96)
        for gname in GRID:
            s = all_results[pname][gname]
            print(
                f"  {gname:<20s} {s.get('sharpe_ratio',0):>8.2f} "
                f"{s.get('annual_return',0):>8.1f} {s.get('total_return',0):>10.1f} "
                f"  {s.get('max_ddpercent',0):>10.1f} "
                f"{s.get('return_drawdown_ratio',0):>10.2f} "
                f"{s.get('total_trade_count',0):>8.0f}"
            )
        print("  " + "-" * 96)

    print(f"\n总耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
