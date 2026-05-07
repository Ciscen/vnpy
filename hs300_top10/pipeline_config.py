"""
hs300_top10/pipeline_config.py

流水线级别全局配置（日期、路径、资金等），单一来源。

所有需要这些常量的模块统一从此处导入，避免多处重复定义。

支持两种模式:
  - 固定日期（回测研究）: PipelineConfig(data_end="2026-04-30")
  - 动态日期（生产运行）: PipelineConfig(data_end="auto").resolve()
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple


class ResolvedDates(NamedTuple):
    """resolve() 返回的确定性日期集合。"""
    data_start: str
    data_end: str
    backtest_start: str
    backtest_end: str


@dataclass(frozen=True)
class PipelineConfig:
    """流水线运行参数（不涉及策略逻辑）。

    日期字段值为 ``"auto"`` 时，由 :meth:`resolve` 根据当前日期动态计算。
    """

    lab_path: str = "./lab/hs300"
    data_start: str = "2016-04-30"
    data_end: str = "2026-04-30"
    backtest_start: str = "2024-05-01"
    backtest_end: str = "2026-04-30"
    capital: int = 100_000
    benchmark: str = "000300.SSE"
    train_years: int = 8

    @property
    def signal_cache(self) -> Path:
        return Path(self.lab_path) / "signal" / "hs300_top10.parquet"

    @property
    def signal_cache_daily(self) -> Path:
        return Path(self.lab_path) / "signal" / "hs300_top10_daily.parquet"

    def resolve(self, ref_date: date | None = None) -> ResolvedDates:
        """将 ``"auto"`` 占位符解析为实际日期字符串。

        Parameters
        ----------
        ref_date : date | None
            参考日期，默认 ``date.today()``。

        Returns
        -------
        ResolvedDates
            (data_start, data_end, backtest_start, backtest_end)
        """
        today = ref_date or date.today()

        d_end = today.isoformat() if self.data_end == "auto" else self.data_end
        d_start = self.data_start

        if self.backtest_end == "auto":
            bt_end = today.isoformat()
        else:
            bt_end = self.backtest_end

        if self.backtest_start == "auto":
            bt_start_date = today - timedelta(days=2 * 365)
            bt_start = bt_start_date.isoformat()
        else:
            bt_start = self.backtest_start

        return ResolvedDates(
            data_start=d_start,
            data_end=d_end,
            backtest_start=bt_start,
            backtest_end=bt_end,
        )


# 默认实例 — 固定日期，回测用
PIPELINE = PipelineConfig()

# 生产实例 — 动态日期，run_live 使用
PIPELINE_LIVE = PipelineConfig(
    data_end="auto",
    backtest_start="auto",
    backtest_end="auto",
)
