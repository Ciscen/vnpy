"""
hs300_top10/model/predictor.py

从训练好的 XGBoost 模型生成信号 DataFrame，
输出格式兼容 vnpy BacktestingEngine.add_strategy() 的 signal_df 参数。
"""
from __future__ import annotations

import numpy as np
import polars as pl

from hs300_top10.model.trainer import XgbClassifierModel


def generate_signals(
    model: XgbClassifierModel,
    features_df: pl.DataFrame,
) -> pl.DataFrame:
    """用模型预测并生成信号表。

    Parameters
    ----------
    model : XgbClassifierModel
        已训练的模型。
    features_df : pl.DataFrame
        特征 DataFrame，包含 datetime, vt_symbol, feature_1 … feature_N 列。
        通常只保留周一行（信号基准日）。

    Returns
    -------
    pl.DataFrame
        (datetime, vt_symbol, signal) — signal 为正类概率。
    """
    features_df = features_df.sort(["datetime", "vt_symbol"])

    probas: np.ndarray = model.predict_from_df(features_df)

    signal_df = features_df.select(["datetime", "vt_symbol"]).with_columns(
        pl.Series("signal", probas)
    )
    return signal_df
