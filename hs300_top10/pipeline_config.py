"""
hs300_top10/pipeline_config.py

流水线级别全局配置（日期、路径、资金等），单一来源。

所有需要这些常量的模块统一从此处导入，避免多处重复定义。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """流水线运行参数（不涉及策略逻辑）。"""

    lab_path: str = "./lab/hs300"
    data_start: str = "2016-04-30"
    data_end: str = "2026-04-30"
    backtest_start: str = "2024-05-01"
    backtest_end: str = "2026-04-30"
    capital: int = 100_000
    benchmark: str = "000300.SSE"

    @property
    def signal_cache(self) -> Path:
        return Path(self.lab_path) / "signal" / "hs300_top10.parquet"

    @property
    def signal_cache_daily(self) -> Path:
        return Path(self.lab_path) / "signal" / "hs300_top10_daily.parquet"


PIPELINE = PipelineConfig()
