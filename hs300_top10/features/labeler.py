"""
hs300_top10/features/labeler.py

生成周度二分类标签：
  - 基准日：每周一（特征截止日）
  - 入场价：周二开盘价
  - 标签=1 ：周二开盘到周五收盘期间，最高价 >= 周二开盘价 * (1 + RISE_THRESH)
  - 标签=0 ：否则

使用 Polars 向量化实现，避免逐行遍历。
"""
from __future__ import annotations

import polars as pl

RISE_THRESH: float = 0.05
WEEK_HORIZON: int = 4  # 周二到周五共 4 个交易日


def generate_weekly_labels(
    df: pl.DataFrame,
    rise_thresh: float = RISE_THRESH,
) -> pl.DataFrame:
    """根据日线数据生成周度二分类标签。

    Parameters
    ----------
    df : pl.DataFrame
        日线数据，需包含 datetime, vt_symbol, open, high, close 列。
    rise_thresh : float
        上涨阈值，默认 0.05 (5%)。

    Returns
    -------
    pl.DataFrame
        (datetime, vt_symbol, label) — datetime 为周一日期。
    """
    work_df = df.select(["datetime", "vt_symbol", "open", "high", "close"]).sort(
        ["vt_symbol", "datetime"]
    )

    work_df = work_df.with_columns(pl.col("datetime").dt.weekday().alias("weekday"))

    all_labels: list[pl.DataFrame] = []

    for symbol, grp in work_df.group_by("vt_symbol"):
        grp = grp.sort("datetime")

        sym_name = symbol[0] if isinstance(symbol, tuple) else symbol

        mondays = grp.filter(pl.col("weekday") == 1)
        if mondays.is_empty():
            continue

        dates = grp["datetime"]
        highs = grp["high"]
        opens = grp["open"]

        date_list = dates.to_list()

        labels: list[dict] = []
        for monday_dt in mondays["datetime"]:
            # 找到周一之后的交易日（周二~周五）
            mask = dates > monday_dt
            future_indices = [i for i, v in enumerate(mask) if v]

            if not future_indices:
                continue

            horizon_end = min(len(future_indices), WEEK_HORIZON)
            horizon_idx = future_indices[:horizon_end]

            tuesday_open = opens[horizon_idx[0]]
            if tuesday_open is None or tuesday_open <= 0:
                continue

            max_high = max(highs[i] for i in horizon_idx)
            label = 1 if max_high >= tuesday_open * (1 + rise_thresh) else 0

            labels.append({
                "datetime": monday_dt,
                "vt_symbol": sym_name,
                "label": label,
            })

        if labels:
            all_labels.append(pl.DataFrame(labels))

    if not all_labels:
        return pl.DataFrame(
            schema={"datetime": pl.Datetime, "vt_symbol": pl.Utf8, "label": pl.Int64}
        )

    result = pl.concat(all_labels).sort(["datetime", "vt_symbol"])
    result = result.with_columns(pl.col("label").cast(pl.Float64))
    return result
