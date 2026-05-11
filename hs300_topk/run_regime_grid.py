"""
hs300_topk/run_regime_grid.py

Regime 参数网格搜索：在三个市场区间对比不同参数组合，寻找最优平衡点。
"""
from __future__ import annotations

import logging
import time
from copy import deepcopy
from dataclasses import replace
from datetime import datetime

import polars as pl

from vnpy.trader.constant import Interval
from vnpy.alpha.strategy import BacktestingEngine

from hs300_topk.data.loader import get_lab, discover_symbols
from hs300_topk.model.rolling_trainer import rolling_train
from hs300_topk.strategy.hs300_topk_strategy import HS300Top10Strategy
from hs300_topk.strategy.config import StrategyConfig, OPTIMIZED_V14

LAB_PATH = "./lab/hs300"
CAPITAL = 100_000
WEEKLY_LABEL = "friday_close"
LAG_DAYS = 3

PERIODS = {
    "bear": {"data_start": "2016-04-30", "data_end": "2023-12-31",
             "bt_start": "2022-01-01", "bt_end": "2023-12-31"},
    "bull": {"data_start": "2016-04-30", "data_end": "2026-04-30",
             "bt_start": "2024-05-01", "bt_end": "2026-04-30"},
    "full": {"data_start": "2016-04-30", "data_end": "2026-04-30",
             "bt_start": "2022-01-01", "bt_end": "2026-04-30"},
}

GRID: dict[str, StrategyConfig] = {
    "baseline": OPTIMIZED_V14,

    "R1_soft": replace(OPTIMIZED_V14,
        version="R1", description="regime only, soft scaling, no breaker",
        regime_filter=True, regime_min_score=0.0,
        max_portfolio_drawdown=0.0,
    ),
    "R2_regime+dd20": replace(OPTIMIZED_V14,
        version="R2", description="regime sqrt + DD 20%",
        regime_filter=True, regime_min_score=0.0,
        max_portfolio_drawdown=0.20, drawdown_cooldown_days=10,
    ),
    "R3_regime+dd15": replace(OPTIMIZED_V14,
        version="R3", description="regime sqrt + DD 15%",
        regime_filter=True, regime_min_score=0.0,
        max_portfolio_drawdown=0.15, drawdown_cooldown_days=10,
    ),
    "R4_cutoff25+dd20": replace(OPTIMIZED_V14,
        version="R4", description="regime cutoff 0.25 + DD 20%",
        regime_filter=True, regime_min_score=0.25,
        max_portfolio_drawdown=0.20, drawdown_cooldown_days=10,
    ),
    "R5_dd_only_20": replace(OPTIMIZED_V14,
        version="R5", description="DD breaker only 20%",
        regime_filter=False,
        max_portfolio_drawdown=0.20, drawdown_cooldown_days=10,
    ),
    "R6_dd_only_15": replace(OPTIMIZED_V14,
        version="R6", description="DD breaker only 15%",
        regime_filter=False,
        max_portfolio_drawdown=0.15, drawdown_cooldown_days=10,
    ),
    "R7_cutoff30+dd15": replace(OPTIMIZED_V14,
        version="R7", description="regime cutoff 0.30 + DD 15%",
        regime_filter=True, regime_min_score=0.30,
        max_portfolio_drawdown=0.15, drawdown_cooldown_days=10,
    ),
}


def run_single(signal_df: pl.DataFrame, cfg: StrategyConfig, period: dict) -> dict:
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


def main() -> None:
    t0 = time.time()
    signals: dict[str, pl.DataFrame] = {}

    seen: dict[str, str] = {}
    for pname, pconfig in PERIODS.items():
        key = f"{pconfig['data_start']}_{pconfig['data_end']}"
        if key in seen:
            print(f"\n  {pname}: 复用 {seen[key]} 的信号")
            signals[pname] = signals[seen[key]]
            continue
        seen[key] = pname
        print(f"\n{'='*60}")
        print(f"  训练 {pname}: {pconfig['data_start']} ~ {pconfig['data_end']}")
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

    all_results: dict[str, dict[str, dict]] = {}
    logging.disable(logging.CRITICAL)
    for pname, pconfig in PERIODS.items():
        print(f"\n  回测 {pname} ...")
        all_results[pname] = {}
        for gname, gcfg in GRID.items():
            stats = run_single(signals[pname], gcfg, pconfig)
            all_results[pname][gname] = stats
    logging.disable(logging.NOTSET)

    print("\n" + "=" * 120)
    print("  Regime 参数网格搜索结果")
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
            sh = s.get("sharpe_ratio", 0)
            ar = s.get("annual_return", 0)
            tr = s.get("total_return", 0)
            md = s.get("max_ddpercent", 0)
            rd = s.get("return_drawdown_ratio", 0)
            tc = s.get("total_trade_count", 0)
            print(
                f"  {gname:<20s} {sh:>8.2f} {ar:>8.1f} {tr:>10.1f} "
                f"  {md:>10.1f} {rd:>10.2f} {tc:>8.0f}"
            )
        print("  " + "-" * 96)

    print(f"\n总耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
