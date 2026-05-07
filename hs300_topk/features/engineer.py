"""
hs300_topk/features/engineer.py

继承 Alpha158，复用 158 个因子，使用占位标签。
实际标签由 labeler.py 在 prepare_data 后替换。
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

    def replace_labels(self, labels_df: pl.DataFrame) -> None:
        """用外部计算的周度二分类标签替换 raw_df/learn_df/infer_df 中的 label 列。

        Parameters
        ----------
        labels_df : pl.DataFrame
            必须包含 (datetime, vt_symbol, label) 三列，
            datetime 对应周一日期。
        """
        for attr in ("raw_df", "learn_df", "infer_df"):
            df: pl.DataFrame = getattr(self, attr)

            df = df.drop("label").join(
                labels_df.select(["datetime", "vt_symbol", "label"]),
                on=["datetime", "vt_symbol"],
                how="left",
            )
            setattr(self, attr, df)
