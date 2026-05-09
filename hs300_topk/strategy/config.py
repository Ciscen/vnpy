"""
hs300_topk/strategy/config.py

策略配置体系 — 将选股、风控、执行三个维度的参数集中管理。

设计原则:
  - 每个版本一个不可变预设（dataclass frozen=False 仅为序列化方便）
  - 参数通过 to_dict() 注入策略类的 setting，与 AlphaStrategy 的 setattr 机制对接
  - 回测/实盘共用同一套参数定义，避免研究/生产偏差

版本演进路线:
  V1.0  基线（固定止损止盈）
  V1.1  调仓平滑 + ATR 自适应止损 + 动态 K + 概率加权
  V1.2  集中持仓 top-8
  V1.3  + 个股止损冷却 10 天
  V1.4  集中持仓 top-5 + Lag-3 特征（当前生产版本）
  V1.5  friday_close 保守标签 + 全面参数重优化（已验证不如 V1.4）
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class StrategyConfig:
    """策略全局配置，覆盖选股、风控、执行三个层面。"""

    # ── 版本标识 ──
    version: str = "v1.0"
    description: str = "基线版本"

    # ── 选股参数 ──
    top_k: int = 10
    min_signal_prob: float = 0.0       # 最低信号概率阈值（0=不过滤）
    dynamic_k: bool = False            # 是否根据信号强度动态调整 K
    dynamic_k_min: int = 3             # 动态 K 最小值
    dynamic_k_prob_threshold: float = 0.35  # 概率低于此值时缩减 K
    weight_by_signal: bool = False     # 是否按概率加权分配仓位

    # ── 风控参数 ──
    stop_loss_pct: float = 0.03        # 硬止损幅度
    tp_activate_pct: float = 0.03      # 追踪止盈激活阈值
    tp_trail_pct: float = 0.02         # 追踪止盈回撤退出阈值
    max_hold_days: int = 4             # 最大持仓交易日

    use_atr_stop: bool = False         # 是否使用 ATR 自适应止损
    atr_stop_multiplier: float = 2.0   # ATR 止损倍数
    atr_stop_min: float = 0.02         # ATR 止损下限
    atr_stop_max: float = 0.06         # ATR 止损上限

    use_market_filter: bool = False    # 是否使用市场状态过滤
    market_ma_period: int = 20         # 市场均线周期（日）
    market_benchmark: str = "000300.SSE"  # 基准指数

    # ── 执行参数 ──
    cash_ratio: float = 0.95           # 现金使用比例
    min_volume: int = 100              # 最小交易单位
    price_add: float = 0.002           # 滑点
    close_cost_rate: float = 0.002     # 估算卖出成本

    smooth_rebalance: bool = False     # 调仓平滑：已持仓 & 仍在信号中的不动
    max_replace_ratio: float = 1.0     # 单次最大换仓比例（1.0=全换）

    # ── V1.2 组合级风控 ──
    portfolio_daily_loss_limit: float = 0.0   # 单日最大亏损限制（0=禁用）
    cooldown_days: int = 0                    # 触发后冷却天数
    min_signal_spread: float = 0.0            # top1-topK 概率差距最小值

    # ── V1.3 风控增强 ──
    conditional_hold_extend: bool = False
    hold_extend_min_pnl: float = 0.03
    hold_extend_days: int = 2
    absolute_stop_cap: float = 0.0
    profit_lock_threshold: float = 0.0
    profit_lock_trail_pct: float = 0.015
    # 动量确认入场
    momentum_filter: bool = False
    momentum_lookback: int = 3
    momentum_min_return: float = -0.03
    # 个股冷却
    stock_cooldown_days: int = 0             # 止损后个股冷却天数（0=禁用）
    # 调仓周期
    rebalance_period: int = 1                # 1=每周, 2=双周

    # ── 模型参数 ──
    xgb_n_estimators: int = 500
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_early_stopping: int = 30
    train_years: int = 8

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: str | Path) -> StrategyConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ══════════════════════════════════════════════════════════
# 预设配置 — 每个版本记录核心参数变更点
# ══════════════════════════════════════════════════════════

BASELINE_V10 = StrategyConfig(
    version="v1.0",
    description="基线版本：Alpha158 + XGBoost + 固定止损止盈",
)

OPTIMIZED_V11 = StrategyConfig(
    version="v1.1",
    description="V1.1: 调仓平滑 + ATR自适应止损 + 动态K + 概率加权",
    smooth_rebalance=True,
    max_replace_ratio=0.7,
    use_atr_stop=True,
    atr_stop_multiplier=2.0,
    atr_stop_min=0.02,
    atr_stop_max=0.05,
    use_market_filter=False,
    dynamic_k=True,
    dynamic_k_min=5,
    dynamic_k_prob_threshold=0.30,
    weight_by_signal=True,
    min_signal_prob=0.15,
    stop_loss_pct=0.04,
    tp_activate_pct=0.04,
    tp_trail_pct=0.02,
    max_hold_days=5,
)

OPTIMIZED_V12 = StrategyConfig(
    version="v1.2",
    description="V1.2: V1.1 + 集中持仓(top8) + 更低换手",
    smooth_rebalance=True,
    max_replace_ratio=0.7,
    use_atr_stop=True,
    atr_stop_multiplier=2.0,
    atr_stop_min=0.02,
    atr_stop_max=0.05,
    dynamic_k=True,
    dynamic_k_min=4,
    dynamic_k_prob_threshold=0.30,
    weight_by_signal=True,
    min_signal_prob=0.15,
    stop_loss_pct=0.04,
    tp_activate_pct=0.04,
    tp_trail_pct=0.02,
    max_hold_days=5,
    top_k=8,
)

OPTIMIZED_V13 = StrategyConfig(
    version="v1.3",
    description="V1.3: V1.2 + 个股止损冷却(10天)",
    smooth_rebalance=True,
    max_replace_ratio=0.7,
    use_atr_stop=True,
    atr_stop_multiplier=2.0,
    atr_stop_min=0.02,
    atr_stop_max=0.05,
    dynamic_k=True,
    dynamic_k_min=4,
    dynamic_k_prob_threshold=0.30,
    weight_by_signal=True,
    min_signal_prob=0.15,
    stop_loss_pct=0.04,
    tp_activate_pct=0.04,
    tp_trail_pct=0.02,
    max_hold_days=5,
    top_k=8,
    stock_cooldown_days=10,
)

OPTIMIZED_V14 = StrategyConfig(
    version="v1.4",
    description="V1.4: V1.3 + 集中持仓(top5) + Lag-3 (当前生产版本)",
    smooth_rebalance=True,
    max_replace_ratio=0.7,
    use_atr_stop=True,
    atr_stop_multiplier=2.0,
    atr_stop_min=0.02,
    atr_stop_max=0.05,
    dynamic_k=True,
    dynamic_k_min=3,
    dynamic_k_prob_threshold=0.30,
    weight_by_signal=True,
    min_signal_prob=0.15,
    stop_loss_pct=0.04,
    tp_activate_pct=0.04,
    tp_trail_pct=0.02,
    max_hold_days=5,
    top_k=5,
    stock_cooldown_days=10,
)

OPTIMIZED_V15 = StrategyConfig(
    version="v1.5",
    description="V1.5: friday_close标签 + 参数优化 (top4/SL2%/TP3%/hold5)",
    smooth_rebalance=True,
    max_replace_ratio=0.5,
    use_atr_stop=True,
    atr_stop_multiplier=2.0,
    atr_stop_min=0.02,
    atr_stop_max=0.05,
    dynamic_k=True,
    dynamic_k_min=2,
    dynamic_k_prob_threshold=0.30,
    weight_by_signal=True,
    min_signal_prob=0.0,
    stop_loss_pct=0.02,
    tp_activate_pct=0.03,
    tp_trail_pct=0.015,
    max_hold_days=5,
    top_k=4,
    stock_cooldown_days=10,
)

