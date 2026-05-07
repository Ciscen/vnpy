"""
hs300_topk/model/trainer.py

XGBoost 二分类模型，实现 AlphaModel 接口。
用于预测股票在未来一周内上涨 >=5% 的概率。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb

from vnpy.alpha.dataset import AlphaDataset, Segment
from vnpy.alpha.model import AlphaModel


class XgbClassifierModel(AlphaModel):
    """XGBoost 二分类模型。

    使用 binary:logistic 目标函数，predict() 返回正类概率
    （即股票未来一周涨幅 >=5% 的概率），用于排序选股。
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 30,
        eval_metric: str = "logloss",
        seed: int = 42,
        verbose_eval: int = 50,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.seed = seed
        self.verbose_eval = verbose_eval

        self.model: xgb.XGBClassifier | None = None
        self.feature_names: list[str] = []

    def _extract_xy(
        self, df: pl.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """从 Polars DataFrame 提取特征矩阵和标签向量。

        跳过前 2 列 (datetime, vt_symbol) 和最后 1 列 (label)。
        """
        df = df.sort(["datetime", "vt_symbol"])
        feature_cols = df.columns[2:-1]
        X = df.select(feature_cols).to_numpy()
        y = np.array(df["label"])
        return X, y, feature_cols

    def fit(self, dataset: AlphaDataset) -> None:
        """训练模型。

        使用 TRAIN 段作为训练集，VALID 段作为验证集 (early stopping)。
        """
        train_df = dataset.fetch_learn(Segment.TRAIN)
        valid_df = dataset.fetch_learn(Segment.VALID)

        X_train, y_train, feature_cols = self._extract_xy(train_df)
        X_valid, y_valid, _ = self._extract_xy(valid_df)

        self.feature_names = feature_cols

        # 将 NaN 标签行去除（标签缺失意味着无法构建监督样本）
        train_mask = ~np.isnan(y_train)
        valid_mask = ~np.isnan(y_valid)
        X_train, y_train = X_train[train_mask], y_train[train_mask]
        X_valid, y_valid = X_valid[valid_mask], y_valid[valid_mask]

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            eval_metric=self.eval_metric,
            random_state=self.seed,
            use_label_encoder=False,
            early_stopping_rounds=self.early_stopping_rounds,
            verbosity=1,
        )

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            verbose=self.verbose_eval,
        )

    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        """预测正类概率。

        Returns
        -------
        np.ndarray
            每行对应一只股票某日的上涨概率，值域 [0, 1]。
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()")

        df = dataset.fetch_infer(segment)
        df = df.sort(["datetime", "vt_symbol"])
        X = df.select(df.columns[2:-1]).to_numpy()

        proba = self.model.predict_proba(X)[:, 1]
        return proba

    def predict_from_df(self, features_df: pl.DataFrame) -> np.ndarray:
        """直接从特征 DataFrame 预测，用于滚动训练场景。

        Parameters
        ----------
        features_df : pl.DataFrame
            包含 datetime, vt_symbol 和特征列的 DataFrame（无需 label 列）。
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()")

        feature_cols = [c for c in features_df.columns if c not in ("datetime", "vt_symbol", "label")]
        X = features_df.select(feature_cols).to_numpy()
        return self.model.predict_proba(X)[:, 1]

    def detail(self) -> Any:
        """输出特征重要性信息"""
        if self.model is None:
            return None
        importance = self.model.feature_importances_
        pairs = sorted(zip(self.feature_names, importance), key=lambda x: -x[1])
        return pairs[:30]
