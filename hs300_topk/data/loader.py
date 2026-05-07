"""
hs300_topk/data/loader.py

封装 AlphaLab 数据加载接口，提供统一的日线数据读取入口。
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from vnpy.trader.constant import Interval
from vnpy.alpha import AlphaLab

from hs300_topk.pipeline_config import PIPELINE

DEFAULT_LAB_PATH = PIPELINE.lab_path


def get_lab(lab_path: str = DEFAULT_LAB_PATH) -> AlphaLab:
    """获取 AlphaLab 实例"""
    return AlphaLab(lab_path)


INDEX_SYMBOLS: set[str] = {"000300.SSE", "000016.SSE", "000905.SSE", "000852.SSE"}


def discover_symbols(
    lab_path: str = DEFAULT_LAB_PATH,
    exclude_index: bool = True,
) -> list[str]:
    """扫描 AlphaLab daily 目录，返回所有可用 vt_symbol 列表。

    Parameters
    ----------
    exclude_index : bool
        是否排除指数代码（如 000300.SSE），默认 True。
    """
    daily_path = Path(lab_path) / "daily"
    symbols = sorted(f.stem for f in daily_path.glob("*.parquet"))
    if exclude_index:
        symbols = [s for s in symbols if s not in INDEX_SYMBOLS]
    return symbols


def load_bar_df(
    lab: AlphaLab,
    vt_symbols: list[str],
    start: str,
    end: str,
    extended_days: int = 100,
) -> pl.DataFrame:
    """加载日线数据为 Polars DataFrame。

    Parameters
    ----------
    lab : AlphaLab
        AlphaLab 实例
    vt_symbols : list[str]
        股票代码列表，格式 "600519.SSE"
    start : str
        起始日期，格式 "YYYY-MM-DD"
    end : str
        结束日期，格式 "YYYY-MM-DD"
    extended_days : int
        向前多读的天数，用于因子回溯窗口

    Returns
    -------
    pl.DataFrame
        列: datetime, vt_symbol, open, high, low, close, volume, turnover, ...
    """
    df = lab.load_bar_df(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=start,
        end=end,
        extended_days=extended_days,
    )
    if df is None:
        raise RuntimeError(
            f"AlphaLab.load_bar_df 返回 None，请检查 lab 目录和股票列表 "
            f"(symbols={len(vt_symbols)}, range={start}~{end})"
        )
    return df
