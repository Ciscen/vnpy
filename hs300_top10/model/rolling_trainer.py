"""
hs300_top10/model/rolling_trainer.py

月度滚动（walk-forward）训练流水线，支持周频和日频两种模式：

周频模式 (rolling_train):
  1. 加载全量日线 → Alpha158 因子
  2. 生成周度二分类标签
  3. 按月切分：用截止当月的历史数据训练，下月周一行做预测
  4. 拼接所有月度信号

日频模式 (rolling_train_daily):
  1. 加载全量日线 → Alpha158 因子
  2. 生成日频二分类标签（3日/2%）
  3. 按月切分：用全量日线训练，下月所有交易日做预测
  4. 拼接所有月度信号（每日每股一行）

用法::

    python -m hs300_top10.model.rolling_trainer
"""
from __future__ import annotations

import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import polars as pl

from vnpy.trader.constant import Interval
from vnpy.alpha import AlphaLab
from vnpy.alpha.dataset import process_drop_na

from hs300_top10.data.loader import discover_symbols, get_lab, load_bar_df
from hs300_top10.features.engineer import HS300Top10Dataset
from hs300_top10.features.labeler import generate_weekly_labels, generate_daily_labels
from hs300_top10.model.trainer import XgbClassifierModel
from hs300_top10.model.predictor import generate_signals

# ──────────────────────────────────────────────────
# 默认配置
# ──────────────────────────────────────────────────
DEFAULT_LAB_PATH = "./lab/hs300"
DATA_START = "2016-04-30"
DATA_END = "2026-04-30"

TRAIN_YEARS = 8
BACKTEST_START = "2024-05-01"
BACKTEST_END = "2026-04-30"


def _month_range(start: str, end: str) -> list[tuple[str, str]]:
    """生成从 start 到 end 的逐月区间列表。

    Returns
    -------
    list of (month_start, month_end)
        格式: [("2024-01-01", "2024-01-31"), ("2024-02-01", "2024-02-29"), ...]
    """
    from datetime import date, timedelta
    import calendar

    s = date.fromisoformat(start)
    e = date.fromisoformat(end)

    ranges: list[tuple[str, str]] = []
    cur = s.replace(day=1)

    while cur <= e:
        last_day = calendar.monthrange(cur.year, cur.month)[1]
        month_end = cur.replace(day=last_day)
        if month_end > e:
            month_end = e
        ranges.append((cur.isoformat(), month_end.isoformat()))
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)

    return ranges


def rolling_train(
    lab_path: str = DEFAULT_LAB_PATH,
    data_start: str = DATA_START,
    data_end: str = DATA_END,
    backtest_start: str = BACKTEST_START,
    backtest_end: str = BACKTEST_END,
    train_years: int = TRAIN_YEARS,
    max_workers: int = 4,
) -> tuple[pl.DataFrame, list[str]]:
    """执行月度滚动训练并返回完整信号表。

    流程
    ----
    1. 加载日线数据 → 构建 Alpha158 全量特征（一次性计算）
    2. 生成周度标签
    3. 将特征仅保留周一行，与标签合并
    4. 逐月滚动：
       - 训练集 = 截止上月底的所有历史周一数据（最多 train_years 年）
       - 验证集 = 训练集最后 20% 时间段
       - 预测集 = 当月的周一行
       - 训练 XGBoost → 预测 → 记录信号
    5. 拼接所有月份信号

    Returns
    -------
    signal_df : pl.DataFrame
        (datetime, vt_symbol, signal) — 覆盖 backtest 区间的完整信号
    vt_symbols : list[str]
        参与回测的股票列表
    """
    print("=" * 60)
    print("  HS300 Top-10 滚动训练")
    print("=" * 60)

    lab = get_lab(lab_path)
    vt_symbols = discover_symbols(lab_path)
    print(f"\n可用股票数: {len(vt_symbols)}")

    # ── Step 1: 加载日线数据 ──
    print("\n[Step 1/4] 加载日线数据 ...")
    bar_df = load_bar_df(lab, vt_symbols, data_start, data_end, extended_days=100)
    print(f"  原始数据: {bar_df.shape[0]} 行 x {bar_df.shape[1]} 列")

    # ── Step 2: 构建 Alpha158 特征（一次性计算） ──
    print("\n[Step 2/4] 计算 Alpha158 因子 (多进程, 可能需要几分钟) ...")

    # 设置一个覆盖整个区间的 period（后续手动切分）
    dataset = HS300Top10Dataset(
        df=bar_df,
        train_period=(data_start, backtest_end),
        valid_period=(data_start, backtest_end),
        test_period=(data_start, backtest_end),
    )

    # 加载成分股筛选器
    index_symbol = "HS300.SSE"
    try:
        filters = lab.load_component_filters(index_symbol, data_start, data_end)
    except Exception:
        filters = None
        print("  [警告] 未找到成分股索引，跳过筛选")

    dataset.prepare_data(filters, max_workers=max_workers)
    print(f"  特征矩阵: {dataset.raw_df.shape[0]} 行 x {dataset.raw_df.shape[1]} 列")

    # ── Step 3: 生成周度标签 & 合并 ──
    print("\n[Step 3/4] 生成周度标签 ...")
    labels_df = generate_weekly_labels(bar_df)
    print(f"  标签总数: {labels_df.shape[0]} (正例率: {labels_df['label'].mean():.2%})")

    # 获取全量特征 DataFrame，只保留周一行
    full_features = dataset.raw_df.with_columns(
        pl.col("datetime").dt.weekday().alias("_weekday")
    )
    monday_features = full_features.filter(pl.col("_weekday") == 1).drop("_weekday")

    # 合并标签
    monday_with_labels = monday_features.drop("label").join(
        labels_df, on=["datetime", "vt_symbol"], how="left"
    )

    print(f"  周一特征行数: {monday_with_labels.shape[0]}")

    # ── Step 4: 逐月滚动训练 ──
    print(f"\n[Step 4/4] 逐月滚动训练 ({backtest_start} ~ {backtest_end}) ...")
    months = _month_range(backtest_start, backtest_end)

    all_signals: list[pl.DataFrame] = []
    feature_cols = [
        c for c in monday_with_labels.columns
        if c not in ("datetime", "vt_symbol", "label")
    ]

    from datetime import date, timedelta

    # 标签间隔：周度标签使用 Monday 后 WEEK_HORIZON(4) 个交易日的数据，
    # 为防止最后一个 Monday 的标签窥探预测月价格，留出 7 日历天间隔
    weekly_label_gap_days = 7

    for i, (m_start, m_end) in enumerate(months):
        print(f"\n  [{i+1}/{len(months)}] 预测月份: {m_start[:7]}")

        train_cutoff = (
            date.fromisoformat(m_start) - timedelta(days=1 + weekly_label_gap_days)
        ).isoformat()
        train_start_limit = date.fromisoformat(m_start) - timedelta(days=1 + train_years * 365)
        train_start_str = train_start_limit.isoformat()

        train_pool = monday_with_labels.filter(
            (pl.col("datetime") >= pl.lit(datetime.fromisoformat(train_start_str)))
            & (pl.col("datetime") <= pl.lit(datetime.fromisoformat(train_cutoff)))
        ).drop_nulls(subset=["label"])

        if train_pool.height < 100:
            print(f"    训练样本不足 ({train_pool.height})，跳过")
            continue

        # 验证集：训练集后 20%
        n_train = train_pool.height
        split_idx = int(n_train * 0.8)
        train_sorted = train_pool.sort("datetime")
        split_date = train_sorted["datetime"][split_idx]

        train_data = train_sorted.slice(0, split_idx)
        valid_data = train_sorted.slice(split_idx, n_train - split_idx)

        # 预测集：当月的周一行
        predict_pool = monday_with_labels.filter(
            (pl.col("datetime") >= pl.lit(datetime.fromisoformat(m_start)))
            & (pl.col("datetime") <= pl.lit(datetime.fromisoformat(m_end)))
        )

        if predict_pool.is_empty():
            print(f"    当月无周一交易日，跳过")
            continue

        # 直接用 numpy 训练 XGBoost（不走 AlphaDataset 接口，因为数据已准备好）
        import numpy as np

        def _to_xy(df: pl.DataFrame):
            X = df.select(feature_cols).to_numpy()
            y = np.array(df["label"])
            mask = ~np.isnan(y)
            return X[mask], y[mask]

        X_train, y_train = _to_xy(train_data)
        X_valid, y_valid = _to_xy(valid_data)

        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False,
            early_stopping_rounds=30,
            verbosity=0,
        )
        clf.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )

        # 预测
        X_pred = predict_pool.select(feature_cols).to_numpy()
        probas = clf.predict_proba(X_pred)[:, 1]

        month_signal = predict_pool.select(["datetime", "vt_symbol"]).with_columns(
            pl.Series("signal", probas)
        )
        all_signals.append(month_signal)

        print(f"    训练: {X_train.shape[0]} 样本, 验证: {X_valid.shape[0]}, "
              f"预测: {month_signal.height} 行, "
              f"best_iter: {clf.best_iteration}")

    if not all_signals:
        raise RuntimeError("未生成任何信号，请检查数据和日期范围")

    signal_df = pl.concat(all_signals).sort(["datetime", "vt_symbol"])
    print(f"\n信号生成完成: {signal_df.shape[0]} 行")
    print(f"日期范围: {signal_df['datetime'].min()} ~ {signal_df['datetime'].max()}")

    return signal_df, vt_symbols


def rolling_train_daily(
    lab_path: str = DEFAULT_LAB_PATH,
    data_start: str = DATA_START,
    data_end: str = DATA_END,
    backtest_start: str = BACKTEST_START,
    backtest_end: str = BACKTEST_END,
    train_years: int = TRAIN_YEARS,
    max_workers: int = 4,
    rise_thresh: float = 0.02,
    horizon: int = 3,
) -> tuple[pl.DataFrame, list[str]]:
    """执行月度滚动训练（日频模式）并返回完整信号表。

    与 rolling_train 的区别：
    - 使用日频标签（未来 horizon 天涨 rise_thresh）替代周度标签
    - 训练和预测使用全量日线，不过滤 weekday
    - 输出 signal_df 每日每股一行

    Returns
    -------
    signal_df : pl.DataFrame
        (datetime, vt_symbol, signal) — 覆盖 backtest 区间的完整日频信号
    vt_symbols : list[str]
        参与回测的股票列表
    """
    print("=" * 60)
    print("  HS300 Top-10 日频滚动训练")
    print(f"  标签: 未来{horizon}日涨{rise_thresh*100:.0f}%")
    print("=" * 60)

    lab = get_lab(lab_path)
    vt_symbols = discover_symbols(lab_path)
    print(f"\n可用股票数: {len(vt_symbols)}")

    # ── Step 1: 加载日线数据 ──
    print("\n[Step 1/4] 加载日线数据 ...")
    bar_df = load_bar_df(lab, vt_symbols, data_start, data_end, extended_days=100)
    print(f"  原始数据: {bar_df.shape[0]} 行 x {bar_df.shape[1]} 列")

    # ── Step 2: 构建 Alpha158 特征 ──
    print("\n[Step 2/4] 计算 Alpha158 因子 (多进程, 可能需要几分钟) ...")
    dataset = HS300Top10Dataset(
        df=bar_df,
        train_period=(data_start, backtest_end),
        valid_period=(data_start, backtest_end),
        test_period=(data_start, backtest_end),
    )

    index_symbol = "HS300.SSE"
    try:
        filters = lab.load_component_filters(index_symbol, data_start, data_end)
    except Exception:
        filters = None
        print("  [警告] 未找到成分股索引，跳过筛选")

    dataset.prepare_data(filters, max_workers=max_workers)
    print(f"  特征矩阵: {dataset.raw_df.shape[0]} 行 x {dataset.raw_df.shape[1]} 列")

    # ── Step 3: 生成日频标签 & 合并 ──
    print(f"\n[Step 3/4] 生成日频标签 (horizon={horizon}, thresh={rise_thresh}) ...")
    labels_df = generate_daily_labels(bar_df, rise_thresh=rise_thresh, horizon=horizon)
    print(f"  标签总数: {labels_df.shape[0]} (正例率: {labels_df['label'].mean():.2%})")

    full_features = dataset.raw_df
    daily_with_labels = full_features.drop("label").join(
        labels_df, on=["datetime", "vt_symbol"], how="left"
    )
    print(f"  日频特征行数: {daily_with_labels.shape[0]}")

    # ── Step 4: 逐月滚动训练 ──
    print(f"\n[Step 4/4] 逐月滚动训练 ({backtest_start} ~ {backtest_end}) ...")
    months = _month_range(backtest_start, backtest_end)

    all_signals: list[pl.DataFrame] = []
    feature_cols = [
        c for c in daily_with_labels.columns
        if c not in ("datetime", "vt_symbol", "label")
    ]

    import numpy as np
    import xgboost as xgb

    from datetime import date, timedelta

    # 标签间隔：horizon 天标签使用了 T+1..T+horizon 的价格，
    # 为防止训练标签窥探预测月数据，需留出 gap
    label_gap_days = horizon * 2 + 1

    for i, (m_start, m_end) in enumerate(months):
        print(f"\n  [{i+1}/{len(months)}] 预测月份: {m_start[:7]}")

        train_cutoff = (
            date.fromisoformat(m_start) - timedelta(days=1 + label_gap_days)
        ).isoformat()
        train_start_limit = date.fromisoformat(m_start) - timedelta(days=1 + train_years * 365)
        train_start_str = train_start_limit.isoformat()

        train_pool = daily_with_labels.filter(
            (pl.col("datetime") >= pl.lit(datetime.fromisoformat(train_start_str)))
            & (pl.col("datetime") <= pl.lit(datetime.fromisoformat(train_cutoff)))
        ).drop_nulls(subset=["label"])

        if train_pool.height < 500:
            print(f"    训练样本不足 ({train_pool.height})，跳过")
            continue

        n_train = train_pool.height
        split_idx = int(n_train * 0.8)
        train_sorted = train_pool.sort("datetime")

        train_data = train_sorted.slice(0, split_idx)
        valid_data = train_sorted.slice(split_idx, n_train - split_idx)

        predict_pool = daily_with_labels.filter(
            (pl.col("datetime") >= pl.lit(datetime.fromisoformat(m_start)))
            & (pl.col("datetime") <= pl.lit(datetime.fromisoformat(m_end)))
        )

        if predict_pool.is_empty():
            print(f"    当月无交易日，跳过")
            continue

        def _to_xy(df: pl.DataFrame):
            X = df.select(feature_cols).to_numpy()
            y = np.array(df["label"])
            mask = ~np.isnan(y)
            return X[mask], y[mask]

        X_train, y_train = _to_xy(train_data)
        X_valid, y_valid = _to_xy(valid_data)

        clf = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False,
            early_stopping_rounds=30,
            verbosity=0,
        )
        clf.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )

        X_pred = predict_pool.select(feature_cols).to_numpy()
        probas = clf.predict_proba(X_pred)[:, 1]

        month_signal = predict_pool.select(["datetime", "vt_symbol"]).with_columns(
            pl.Series("signal", probas)
        )
        all_signals.append(month_signal)

        print(f"    训练: {X_train.shape[0]} 样本, 验证: {X_valid.shape[0]}, "
              f"预测: {month_signal.height} 行, "
              f"best_iter: {clf.best_iteration}")

    if not all_signals:
        raise RuntimeError("未生成任何信号，请检查数据和日期范围")

    signal_df = pl.concat(all_signals).sort(["datetime", "vt_symbol"])
    print(f"\n日频信号生成完成: {signal_df.shape[0]} 行")
    print(f"日期范围: {signal_df['datetime'].min()} ~ {signal_df['datetime'].max()}")

    return signal_df, vt_symbols


if __name__ == "__main__":
    signal_df, vt_symbols = rolling_train()
    print(f"\n最终信号表: {signal_df.shape}")
    print(signal_df.head(20))
