"""
hs300_topk/tune_params.py

在 friday_close 保守标签信号上，分阶段网格扫描策略参数，
找到最优 V1.5 配置。

用法:
    python -m hs300_topk.tune_params
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import polars as pl

from vnpy.trader.constant import Interval

from vnpy.alpha.strategy import BacktestingEngine
from hs300_topk.data.loader import discover_symbols, get_lab
from hs300_topk.pipeline_config import PIPELINE, PipelineConfig
from hs300_topk.strategy.config import StrategyConfig, OPTIMIZED_V14
from hs300_topk.strategy.hs300_topk_strategy import HS300Top10Strategy


REPORT_DIR = Path("hs300_topk") / "output" / "_tune"


def _run_one(
    signal_df: pl.DataFrame,
    pipe: PipelineConfig,
    config: StrategyConfig,
    lab,
    vt_symbols: list[str],
) -> dict:
    """单次回测，返回统计指标字典。"""
    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=datetime.fromisoformat(pipe.backtest_start),
        end=datetime.fromisoformat(pipe.backtest_end),
        capital=pipe.capital,
    )
    setting = {
        k: v for k, v in config.to_dict().items()
        if k not in ("version", "description") and not k.startswith("xgb_")
        and k != "train_years"
    }
    engine.add_strategy(HS300Top10Strategy, setting, signal_df)
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    return engine.calculate_statistics()


def _score(stats: dict) -> float:
    """综合评分：Sharpe * 收益回撤比 的几何均值。"""
    sharpe = max(stats.get("sharpe_ratio", 0), 0)
    rdd = max(stats.get("return_drawdown_ratio", 0), 0)
    return (sharpe * rdd) ** 0.5


def main() -> None:
    pipe = PIPELINE

    cache_path = pipe.signal_cache_weekly_realistic
    if not cache_path.exists():
        print(f"错误: friday_close 信号缓存不存在: {cache_path}")
        print("请先运行: python -m hs300_topk.run_pipeline --weekly-label friday_close --config v1.4")
        sys.exit(1)

    signal_df = pl.read_parquet(cache_path)
    print(f"加载信号: {signal_df.shape[0]} 行, "
          f"{signal_df['datetime'].min()} ~ {signal_df['datetime'].max()}")

    lab = get_lab(pipe.lab_path)
    vt_symbols = discover_symbols(pipe.lab_path)
    print(f"股票数: {len(vt_symbols)}")

    print(f"股票数(信号): {signal_df['vt_symbol'].n_unique()}")

    base = OPTIMIZED_V14

    # ── Stage 1: top_k ──
    print("\n" + "=" * 70)
    print("  Stage 1: 扫描 top_k")
    print("=" * 70)

    stage1_results = []
    for k in [3, 4, 5, 6, 8, 10]:
        cfg = replace(base, version=f"tune_k{k}", top_k=k,
                      dynamic_k_min=max(2, k - 2))
        t0 = time.time()
        stats = _run_one(signal_df, pipe, cfg, lab, vt_symbols)
        elapsed = time.time() - t0
        score = _score(stats)
        stage1_results.append((k, stats, score))
        print(f"  top_k={k:2d}  年化={stats['annual_return']:6.1f}%  "
              f"Sharpe={stats['sharpe_ratio']:.2f}  "
              f"回撤={stats['max_ddpercent']:.1f}%  "
              f"score={score:.3f}  ({elapsed:.1f}s)")

    best_k = max(stage1_results, key=lambda x: x[2])[0]
    print(f"\n  >>> 最优 top_k = {best_k}")

    # ── Stage 2: 止损 + 止盈 ──
    print("\n" + "=" * 70)
    print("  Stage 2: 扫描止损(stop_loss) / 止盈激活(tp_activate)")
    print("=" * 70)

    stage2_results = []
    for sl, tp in itertools.product(
        [0.02, 0.03, 0.04, 0.05, 0.06],
        [0.02, 0.03, 0.04, 0.05, 0.06],
    ):
        cfg = replace(base, version=f"tune_sl{sl}_tp{tp}",
                      top_k=best_k, dynamic_k_min=max(2, best_k - 2),
                      stop_loss_pct=sl, tp_activate_pct=tp)
        stats = _run_one(signal_df, pipe, cfg, lab, vt_symbols)
        score = _score(stats)
        stage2_results.append((sl, tp, stats, score))
        print(f"  SL={sl:.2f} TP={tp:.2f}  年化={stats['annual_return']:6.1f}%  "
              f"Sharpe={stats['sharpe_ratio']:.2f}  "
              f"回撤={stats['max_ddpercent']:.1f}%  score={score:.3f}")

    best_sl, best_tp = max(stage2_results, key=lambda x: x[3])[:2]
    print(f"\n  >>> 最优 SL={best_sl:.2f}, TP={best_tp:.2f}")

    # ── Stage 3: 持仓天数 + trail_pct ──
    print("\n" + "=" * 70)
    print("  Stage 3: 扫描持仓天数(max_hold) / 追踪幅度(tp_trail)")
    print("=" * 70)

    stage3_results = []
    for hold, trail in itertools.product(
        [3, 4, 5, 6, 7],
        [0.01, 0.015, 0.02, 0.025, 0.03],
    ):
        cfg = replace(base, version=f"tune_h{hold}_tr{trail}",
                      top_k=best_k, dynamic_k_min=max(2, best_k - 2),
                      stop_loss_pct=best_sl, tp_activate_pct=best_tp,
                      max_hold_days=hold, tp_trail_pct=trail)
        stats = _run_one(signal_df, pipe, cfg, lab, vt_symbols)
        score = _score(stats)
        stage3_results.append((hold, trail, stats, score))
        print(f"  hold={hold} trail={trail:.3f}  年化={stats['annual_return']:6.1f}%  "
              f"Sharpe={stats['sharpe_ratio']:.2f}  "
              f"回撤={stats['max_ddpercent']:.1f}%  score={score:.3f}")

    best_hold, best_trail = max(stage3_results, key=lambda x: x[3])[:2]
    print(f"\n  >>> 最优 hold={best_hold}, trail={best_trail:.3f}")

    # ── Stage 4: 信号阈值 + 动态K阈值 ──
    print("\n" + "=" * 70)
    print("  Stage 4: 扫描信号阈值(min_signal_prob) / 动态K阈值")
    print("=" * 70)

    stage4_results = []
    for prob, dk_thresh in itertools.product(
        [0.0, 0.05, 0.10, 0.15, 0.20, 0.25],
        [0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
    ):
        cfg = replace(base, version=f"tune_p{prob}_dk{dk_thresh}",
                      top_k=best_k, dynamic_k_min=max(2, best_k - 2),
                      stop_loss_pct=best_sl, tp_activate_pct=best_tp,
                      max_hold_days=best_hold, tp_trail_pct=best_trail,
                      min_signal_prob=prob,
                      dynamic_k_prob_threshold=dk_thresh)
        stats = _run_one(signal_df, pipe, cfg, lab, vt_symbols)
        score = _score(stats)
        stage4_results.append((prob, dk_thresh, stats, score))
        print(f"  prob={prob:.2f} dk={dk_thresh:.2f}  年化={stats['annual_return']:6.1f}%  "
              f"Sharpe={stats['sharpe_ratio']:.2f}  "
              f"回撤={stats['max_ddpercent']:.1f}%  score={score:.3f}")

    best_prob, best_dk = max(stage4_results, key=lambda x: x[3])[:2]
    print(f"\n  >>> 最优 prob={best_prob:.2f}, dk_thresh={best_dk:.2f}")

    # ── Stage 5: 冷却 + 替换比例 ──
    print("\n" + "=" * 70)
    print("  Stage 5: 扫描个股冷却天数 / 换仓比例")
    print("=" * 70)

    stage5_results = []
    for cool, mrr in itertools.product(
        [0, 5, 10, 15, 20],
        [0.5, 0.7, 1.0],
    ):
        cfg = replace(base, version=f"tune_cd{cool}_mr{mrr}",
                      top_k=best_k, dynamic_k_min=max(2, best_k - 2),
                      stop_loss_pct=best_sl, tp_activate_pct=best_tp,
                      max_hold_days=best_hold, tp_trail_pct=best_trail,
                      min_signal_prob=best_prob,
                      dynamic_k_prob_threshold=best_dk,
                      stock_cooldown_days=cool,
                      max_replace_ratio=mrr)
        stats = _run_one(signal_df, pipe, cfg, lab, vt_symbols)
        score = _score(stats)
        stage5_results.append((cool, mrr, stats, score))
        print(f"  cooldown={cool:2d} replace={mrr:.1f}  年化={stats['annual_return']:6.1f}%  "
              f"Sharpe={stats['sharpe_ratio']:.2f}  "
              f"回撤={stats['max_ddpercent']:.1f}%  score={score:.3f}")

    best_cool, best_mrr = max(stage5_results, key=lambda x: x[3])[:2]
    print(f"\n  >>> 最优 cooldown={best_cool}, replace_ratio={best_mrr:.1f}")

    # ── 最终配置 ──
    best_config = replace(
        base,
        version="v1.5",
        description="V1.5: friday_close标签 + 参数优化",
        top_k=best_k,
        dynamic_k_min=max(2, best_k - 2),
        stop_loss_pct=best_sl,
        tp_activate_pct=best_tp,
        max_hold_days=best_hold,
        tp_trail_pct=best_trail,
        min_signal_prob=best_prob,
        dynamic_k_prob_threshold=best_dk,
        stock_cooldown_days=best_cool,
        max_replace_ratio=best_mrr,
    )

    print("\n" + "=" * 70)
    print("  最终配置 V1.5:")
    print("=" * 70)
    for k, v in best_config.to_dict().items():
        if k in ("version", "description") or k.startswith("xgb_") or k == "train_years":
            continue
        base_v = getattr(base, k, None)
        marker = " ← 调整" if v != base_v else ""
        print(f"  {k:30s} = {v}{marker}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    best_config.to_json(REPORT_DIR / "v1.5_config.json")
    print(f"\n  配置已保存: {REPORT_DIR / 'v1.5_config.json'}")

    all_results = {
        "stage1_top_k": [(k, sc) for k, _, sc in stage1_results],
        "stage2_sl_tp": [(sl, tp, sc) for sl, tp, _, sc in stage2_results],
        "stage3_hold_trail": [(h, t, sc) for h, t, _, sc in stage3_results],
        "stage4_prob_dk": [(p, d, sc) for p, d, _, sc in stage4_results],
        "stage5_cool_replace": [(c, m, sc) for c, m, _, sc in stage5_results],
        "best_params": {
            "top_k": best_k,
            "stop_loss_pct": best_sl,
            "tp_activate_pct": best_tp,
            "max_hold_days": best_hold,
            "tp_trail_pct": best_trail,
            "min_signal_prob": best_prob,
            "dynamic_k_prob_threshold": best_dk,
            "stock_cooldown_days": best_cool,
            "max_replace_ratio": best_mrr,
        },
    }
    (REPORT_DIR / "tune_results.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  扫描结果: {REPORT_DIR / 'tune_results.json'}")

    # ── 最终验证回测 ──
    print("\n" + "=" * 70)
    print("  最终回测 V1.5 (全样本)")
    print("=" * 70)
    final_stats = _run_one(signal_df, pipe, best_config, lab, vt_symbols)
    print(f"  年化收益: {final_stats['annual_return']:.1f}%")
    print(f"  Sharpe:   {final_stats['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {final_stats['max_ddpercent']:.1f}%")
    print(f"  收益回撤比: {final_stats['return_drawdown_ratio']:.2f}")

    stats_path = REPORT_DIR / "v1.5_statistics.json"

    def _ser(obj):
        if hasattr(obj, "item"):
            return obj.item()
        return str(obj)

    stats_path.write_text(
        json.dumps(final_stats, indent=2, default=_ser),
        encoding="utf-8",
    )
    print(f"  绩效文件: {stats_path}")


if __name__ == "__main__":
    main()
