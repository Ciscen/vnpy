"""
hs300_topk/features/engineer.py

特征工程 — 复用 Qlib Alpha158 的 158 个量价因子。

Alpha158 因子包括: 动量类(ROC/RSI)、波动率类(STD/ATR)、成交量类(VWAP/OBV)、
均线类(MA/EMA)、以及交叉类特征。全部基于历史价格计算，无未来信息泄漏。

标签由外部 labeler.py 生成后替换，此处仅设置 Alpha158 的占位标签
以满足 prepare_data 的列计算流程。
"""
from __future__ import annotations

import polars as pl

from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158


class HS300Top10Dataset(Alpha158):
    """Alpha158 基础上的周度选股数据集。

    复用 Alpha158 的 158 个因子特征，但标签由外部 labeler 提供，
    此处仅设置一个占位标签以满足 prepare_data 流程。
    """

    def __init__(
        self,
        df: pl.DataFrame,
        train_period: tuple[str, str],
        valid_period: tuple[str, str],
        test_period: tuple[str, str],
    ) -> None:
        super().__init__(
            df=df,
            train_period=train_period,
            valid_period=valid_period,
            test_period=test_period,
        )
        # Alpha158.__init__ 已设置 label = "ts_delay(close, -3) / ts_delay(close, -1) - 1"
        # 这里不覆盖，让 prepare_data 正常计算。
        # 真正的二分类标签将在后续通过 replace_labels() 替换。

