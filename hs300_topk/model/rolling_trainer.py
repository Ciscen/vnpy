"""
hs300_topk/model/rolling_trainer.py

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

    python -m hs300_topk.model.rolling_trainer
"""
from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import xgboost as xgb

from hs300_topk.data.loader import discover_symbols, get_lab, load_bar_df
from hs300_topk.features.engineer import HS300Top10Dataset
from hs300_topk.features.labeler import generate_weekly_labels, generate_daily_labels
from hs300_topk.pipeline_config import PIPELINE

# ──────────────────────────────────────────────────
# 默认配置 — 来自 pipeline_config 统一管理
# ──────────────────────────────────────────────────
DEFAULT_LAB_PATH = PIPELINE.lab_path
DATA_START = PIPELINE.data_start
DATA_END = PIPELINE.data_end
TRAIN_YEARS = 8
BACKTEST_START = PIPELINE.backtest_start
BACKTEST_END = PIPELINE.backtest_end


def _month_range(start: str, end: str) -> list[tuple[str, str]]:
    """生成从 start 到 end 的逐月区间列表。

    Returns
    -------
    list of (month_start, month_end)
        格式: [("2024-01-01", "2024-01-31"), ("2024-02-01", "2024-02-29"), ...]
    """
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


# ──────────────────────────────────────────────────
# 公共数据准备
# ──────────────────────────────────────────────────

def _load_features(
    lab_path: str,
    data_start: str,
    data_end: str,
    backtest_end: str,
    max_workers: int,
) -> tuple[pl.DataFrame, list[str]]:
    """加载日线并构建 Alpha158 特征矩阵（Step 1 & 2 公共逻辑）。"""
    lab = get_lab(lab_path)
    vt_symbols = discover_symbols(lab_path)
    print(f"\n可用股票数: {len(vt_symbols)}")

    print("\n[Step 1/4] 加载日线数据 ...")
    bar_df = load_bar_df(lab, vt_symbols, data_start, data_end, extended_days=100)
    print(f"  原始数据: {bar_df.shape[0]} 行 x {bar_df.shape[1]} 列")

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

    return bar_df, dataset.raw_df, vt_symbols


# ──────────────────────────────────────────────────
# 公共月度滚动训练循环
# ──────────────────────────────────────────────────

def _rolling_loop(
    feature_df: pl.DataFrame,
    backtest_start: str,
    backtest_end: str,
    train_years: int,
    label_gap_days: int,
    min_train_samples: int,
) -> pl.DataFrame:
    """按月滚动训练 XGBoost 并拼接信号（Step 4 公共逻辑）。

    Parameters
    ----------
    feature_df : pl.DataFrame
        含 datetime, vt_symbol, label 及特征列的完整数据。
    label_gap_days : int
        训练截止日与预测月之间的安全间隔天数。
    min_train_samples : int
        最少训练样本数，不足则跳过该月。
    """
    months = _month_range(backtest_start, backtest_end)
    feature_cols = [
        c for c in feature_df.columns
        if c not in ("datetime", "vt_symbol", "label")
    ]

    def _to_xy(df: pl.DataFrame):
        X = df.select(feature_cols).to_numpy()
        y = np.array(df["label"])
        mask = ~np.isnan(y)
        return X[mask], y[mask]

    all_signals: list[pl.DataFrame] = []

    for i, (m_start, m_end) in enumerate(months):
        print(f"\n  [{i+1}/{len(months)}] 预测月份: {m_start[:7]}")

        train_cutoff = (
            date.fromisoformat(m_start) - timedelta(days=1 + label_gap_days)
        ).isoformat()
        train_start_limit = date.fromisoformat(m_start) - timedelta(days=1 + train_years * 365)
        train_start_str = train_start_limit.isoformat()

        train_pool = feature_df.filter(
            (pl.col("datetime") >= pl.lit(datetime.fromisoformat(train_start_str)))
            & (pl.col("datetime") <= pl.lit(datetime.fromisoformat(train_cutoff)))
        ).drop_nulls(subset=["label"])

        if train_pool.height < min_train_samples:
            print(f"    训练样本不足 ({train_pool.height} < {min_train_samples})，跳过")
            continue

        n_train = train_pool.height
        split_idx = int(n_train * 0.8)
        train_sorted = train_pool.sort("datetime")

        train_data = train_sorted.slice(0, split_idx)
        valid_data = train_sorted.slice(split_idx, n_train - split_idx)

        predict_pool = feature_df.filter(
            (pl.col("datetime") >= pl.lit(datetime.fromisoformat(m_start)))
            & (pl.col("datetime") <= pl.lit(datetime.fromisoformat(m_end)))
        )

        if predict_pool.is_empty():
            print("    当月无交易日，跳过")
            continue

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
    return signal_df


# ══════════════════════════════════════════════════
# 公共 API
# ══════════════════════════════════════════════════

def rolling_train(
    lab_path: str = DEFAULT_LAB_PATH,
    data_start: str = DATA_START,
    data_end: str = DATA_END,
    backtest_start: str = BACKTEST_START,
    backtest_end: str = BACKTEST_END,
    train_years: int = TRAIN_YEARS,
    max_workers: int = 4,
) -> tuple[pl.DataFrame, list[str]]:
    """执行月度滚动训练（周频模式）并返回完整信号表。

    Returns
    -------
    signal_df : pl.DataFrame
        (datetime, vt_symbol, signal) — 覆盖 backtest 区间的完整信号
    vt_symbols : list[str]
        参与回测的股票列表
    """
    print("=" * 60)
    print("  HS300 Top-K 滚动训练")
    print("=" * 60)

    bar_df, raw_features, vt_symbols = _load_features(
        lab_path, data_start, data_end, backtest_end, max_workers,
    )

    # ── Step 3: 生成周度标签 & 合并 ──
    print("\n[Step 3/4] 生成周度标签 ...")
    labels_df = generate_weekly_labels(bar_df)
    print(f"  标签总数: {labels_df.shape[0]} (正例率: {labels_df['label'].mean():.2%})")

    monday_features = raw_features.with_columns(
        pl.col("datetime").dt.weekday().alias("_weekday")
    ).filter(pl.col("_weekday") == 1).drop("_weekday")

    monday_with_labels = monday_features.drop("label").join(
        labels_df, on=["datetime", "vt_symbol"], how="left"
    )
    print(f"  周一特征行数: {monday_with_labels.shape[0]}")

    # ── Step 4: 逐月滚动训练 ──
    print(f"\n[Step 4/4] 逐月滚动训练 ({backtest_start} ~ {backtest_end}) ...")
    signal_df = _rolling_loop(
        feature_df=monday_with_labels,
        backtest_start=backtest_start,
        backtest_end=backtest_end,
        train_years=train_years,
        label_gap_days=7,
        min_train_samples=100,
    )

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
    print("  HS300 Top-K 日频滚动训练")
    print(f"  标签: 未来{horizon}日涨{rise_thresh*100:.0f}%")
    print("=" * 60)

    bar_df, raw_features, vt_symbols = _load_features(
        lab_path, data_start, data_end, backtest_end, max_workers,
    )

    # ── Step 3: 生成日频标签 & 合并 ──
    print(f"\n[Step 3/4] 生成日频标签 (horizon={horizon}, thresh={rise_thresh}) ...")
    labels_df = generate_daily_labels(bar_df, rise_thresh=rise_thresh, horizon=horizon)
    print(f"  标签总数: {labels_df.shape[0]} (正例率: {labels_df['label'].mean():.2%})")

    daily_with_labels = raw_features.drop("label").join(
        labels_df, on=["datetime", "vt_symbol"], how="left"
    )
    print(f"  日频特征行数: {daily_with_labels.shape[0]}")

    # ── Step 4: 逐月滚动训练 ──
    print(f"\n[Step 4/4] 逐月滚动训练 ({backtest_start} ~ {backtest_end}) ...")
    label_gap_days = horizon * 2 + 1
    signal_df = _rolling_loop(
        feature_df=daily_with_labels,
        backtest_start=backtest_start,
        backtest_end=backtest_end,
        train_years=train_years,
        label_gap_days=label_gap_days,
        min_train_samples=500,
    )

    print(f"\n日频信号生成完成: {signal_df.shape[0]} 行")
    print(f"日期范围: {signal_df['datetime'].min()} ~ {signal_df['datetime'].max()}")
    return signal_df, vt_symbols


def predict_live(
    target_date: date,
    lab_path: str = DEFAULT_LAB_PATH,
    data_start: str = DATA_START,
    train_years: int = TRAIN_YEARS,
    max_workers: int = 4,
) -> pl.DataFrame:
    """为单个目标日期（周一）生成所有股票的信号概率。

    流程：
    1. 加载日线 → Alpha158 特征（data_start ~ target_date）
    2. 生成周度标签
    3. 用 target_date 之前的历史数据训练一次 XGBoost
    4. 对 target_date 当天所有股票做预测

    Parameters
    ----------
    target_date : date
        预测目标日（应为周一）。
    lab_path : str
        AlphaLab 数据路径。
    data_start : str
        历史数据起始日期。
    train_years : int
        训练窗口年限。
    max_workers : int
        特征计算并行度。

    Returns
    -------
    pl.DataFrame
        列: vt_symbol, signal — 每只股票的信号概率
    """
    data_end = target_date.isoformat()

    print("=" * 60)
    print(f"  HS300 Top-K 实时信号生成 ({target_date})")
    print("=" * 60)

    bar_df, raw_features, vt_symbols = _load_features(
        lab_path, data_start, data_end, data_end, max_workers,
    )

    print("\n[Step 3] 生成周度标签 ...")
    labels_df = generate_weekly_labels(bar_df)

    monday_features = raw_features.with_columns(
        pl.col("datetime").dt.weekday().alias("_weekday")
    ).filter(pl.col("_weekday") == 1).drop("_weekday")

    monday_with_labels = monday_features.drop("label").join(
        labels_df, on=["datetime", "vt_symbol"], how="left"
    )

    feature_cols = [
        c for c in monday_with_labels.columns
        if c not in ("datetime", "vt_symbol", "label")
    ]

    label_gap_days = 7
    train_cutoff = (target_date - timedelta(days=1 + label_gap_days)).isoformat()
    train_start_limit = (target_date - timedelta(days=1 + train_years * 365)).isoformat()

    train_pool = monday_with_labels.filter(
        (pl.col("datetime") >= pl.lit(datetime.fromisoformat(train_start_limit)))
        & (pl.col("datetime") <= pl.lit(datetime.fromisoformat(train_cutoff)))
    ).drop_nulls(subset=["label"])

    print(f"\n[Step 4] 训练 XGBoost (样本: {train_pool.height}) ...")
    if train_pool.height < 100:
        raise RuntimeError(f"训练样本不足: {train_pool.height} < 100")

    n_train = train_pool.height
    split_idx = int(n_train * 0.8)
    train_sorted = train_pool.sort("datetime")
    train_data = train_sorted.slice(0, split_idx)
    valid_data = train_sorted.slice(split_idx, n_train - split_idx)

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
    clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    print(f"  训练: {X_train.shape[0]}, 验证: {X_valid.shape[0]}, "
          f"best_iter: {clf.best_iteration}")

    target_dt = datetime(target_date.year, target_date.month, target_date.day)
    predict_pool = monday_with_labels.filter(
        pl.col("datetime") == pl.lit(target_dt)
    )

    if predict_pool.is_empty():
        # 数据截至日可能早于 target_date（例如周末/节假日/数据延迟），
        # 向前搜索最近可用的周一数据，最多回溯 14 天
        nearby = monday_with_labels.filter(
            pl.col("datetime") >= pl.lit(target_dt - timedelta(days=14))
        ).filter(
            pl.col("datetime") <= pl.lit(target_dt + timedelta(days=3))
        )
        if not nearby.is_empty():
            actual_date = nearby["datetime"].max()
            predict_pool = monday_with_labels.filter(
                pl.col("datetime") == pl.lit(actual_date)
            )
            print(f"  [注意] 目标日期 {target_date} 无数据，使用最近日期 {actual_date.date()}")
        else:
            raise RuntimeError(f"目标日期 {target_date} 附近 14 天内无可用数据")

    X_pred = predict_pool.select(feature_cols).to_numpy()
    probas = clf.predict_proba(X_pred)[:, 1]

    result = predict_pool.select(["vt_symbol"]).with_columns(
        pl.Series("signal", probas)
    ).sort("signal", descending=True)

    print(f"\n信号生成完成: {result.height} 只股票")
    print(f"Top-5: {result.head(5).to_dicts()}")
    return result


if __name__ == "__main__":
    signal_df, vt_symbols = rolling_train()
    print(f"\n最终信号表: {signal_df.shape}")
    print(signal_df.head(20))
