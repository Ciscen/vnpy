"""
hs300_topk/features/labeler.py

标签生成模块（周度）：

周度标签 (generate_weekly_labels):
  - 基准日：每周一（特征截止日）
  - 入场价：周二开盘价
  - 标签=1 ：周二开盘到周五收盘期间，最高价 >= 周二开盘价 * (1 + RISE_THRESH)

周度保守标签 (generate_weekly_labels_realistic, 对照):
  - 同上基准日与周二开盘价
  - 标签=1 ：本周最后一个交易日收盘价 >= 周二开盘价 * (1 + REALISTIC_CLOSE_THRESH)
    （默认 +3%，持有到周期末的可实现收益 proxy）
"""
from __future__ import annotations

import polars as pl

RISE_THRESH: float = 0.05
WEEK_HORIZON: int = 4  # 周二到周五共 4 个交易日
REALISTIC_CLOSE_THRESH: float = 0.03  # 方案 B：周五（或周内最后一日）收盘相对周二开盘


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


def generate_weekly_labels_realistic(
    df: pl.DataFrame,
    close_rise_thresh: float = REALISTIC_CLOSE_THRESH,
) -> pl.DataFrame:
    """周内「持有到期」风格的保守二分类标签（对照组，不替换 optimistic 标签）。

    对每周一：取周二开盘价与随后最多 ``WEEK_HORIZON`` 个交易日；
    若周内最后一日收盘价 >= 周二开盘价 * (1 + close_rise_thresh)，则 label=1。

    Parameters
    ----------
    df : pl.DataFrame
        日线数据，需包含 datetime, vt_symbol, open, close 列。
    close_rise_thresh : float
        默认 0.03（+3%）。

    Returns
    -------
    pl.DataFrame
        (datetime, vt_symbol, label) — datetime 为周一日期。
    """
    work_df = df.select(["datetime", "vt_symbol", "open", "close"]).sort(
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
        opens = grp["open"]
        closes = grp["close"]

        labels: list[dict] = []
        for monday_dt in mondays["datetime"]:
            mask = dates > monday_dt
            future_indices = [i for i, v in enumerate(mask) if v]

            if not future_indices:
                continue

            horizon_end = min(len(future_indices), WEEK_HORIZON)
            horizon_idx = future_indices[:horizon_end]

            tuesday_open = opens[horizon_idx[0]]
            if tuesday_open is None or tuesday_open <= 0:
                continue

            last_close = closes[horizon_idx[-1]]
            if last_close is None or last_close <= 0:
                continue

            label = 1 if last_close >= tuesday_open * (1 + close_rise_thresh) else 0
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

def generate_weekly_labels_excess_return(
    df: pl.DataFrame,
    benchmark_symbol: str = "000300.SSE",
    excess_thresh: float = 0.0,
) -> pl.DataFrame:
    """计算周度超额收益标签（V1.5 策略核心）。
    
    标签 = 1 if (个股持有期收益 - 基准持有期收益) > excess_thresh else 0
    """
    work_df = df.select(["datetime", "vt_symbol", "open", "close"]).sort(
        ["vt_symbol", "datetime"]
    )
    work_df = work_df.with_columns(pl.col("datetime").dt.weekday().alias("weekday"))
    
    # 1. 先计算所有 symbol 的周度收益
    all_rets: list[dict] = []
    
    for symbol, grp in work_df.group_by("vt_symbol"):
        grp = grp.sort("datetime")
        sym_name = symbol[0] if isinstance(symbol, tuple) else symbol
        
        mondays = grp.filter(pl.col("weekday") == 1)
        if mondays.is_empty():
            continue
            
        dates = grp["datetime"]
        opens = grp["open"]
        closes = grp["close"]
        
        for monday_dt in mondays["datetime"]:
            mask = dates > monday_dt
            future_indices = [i for i, v in enumerate(mask) if v]
            if not future_indices:
                continue
                
            horizon_end = min(len(future_indices), WEEK_HORIZON)
            horizon_idx = future_indices[:horizon_end]
            
            tuesday_open = opens[horizon_idx[0]]
            if tuesday_open is None or tuesday_open <= 0:
                continue
                
            last_close = closes[horizon_idx[-1]]
            if last_close is None or last_close <= 0:
                continue
                
            ret = (last_close / tuesday_open) - 1.0
            
            all_rets.append({
                "datetime": monday_dt,
                "vt_symbol": sym_name,
                "ret": ret,
            })
            
    if not all_rets:
        return pl.DataFrame(
            schema={"datetime": pl.Datetime, "vt_symbol": pl.Utf8, "label": pl.Float64}
        )
        
    ret_df = pl.DataFrame(all_rets)
    
    # 2. 提取基准收益
    benchmark_df = ret_df.filter(pl.col("vt_symbol") == benchmark_symbol).select(
        ["datetime", pl.col("ret").alias("bench_ret")]
    )
    
    # 如果不存在基准数据，降级为绝对收益
    if benchmark_df.is_empty():
        print(f"  [警告] 未找到基准 {benchmark_symbol} 数据，降级为绝对收益计算超额标签。")
        ret_df = ret_df.with_columns(pl.lit(0.0).alias("bench_ret"))
    else:
        ret_df = ret_df.join(benchmark_df, on="datetime", how="left")
        # 对于缺失基准收益的周，假设基准收益为 0
        ret_df = ret_df.with_columns(pl.col("bench_ret").fill_null(0.0))
        
    # 3. 计算超额收益并打标
    # 个股需要排除基准自身（可选，但通常保留也没关系，它总是0超额）
    ret_df = ret_df.filter(pl.col("vt_symbol") != benchmark_symbol)
    
    ret_df = ret_df.with_columns(
        ((pl.col("ret") - pl.col("bench_ret")) > excess_thresh).cast(pl.Float64).alias("label")
    )
    
    result = ret_df.select(["datetime", "vt_symbol", "label"]).sort(["datetime", "vt_symbol"])
    return result

def generate_weekly_labels_dynamic(
    df: pl.DataFrame,
    benchmark_symbol: str = "000300.SSE",
    ma_period: int = 40,
    rise_thresh: float = RISE_THRESH,
    excess_thresh: float = 0.0,
) -> pl.DataFrame:
    """根据大盘 Regime 动态切换打标逻辑。
    
    牛市 (基准收盘价 > 基准 MA): 倾向高弹性，用 high_touch
    熊市 (基准收盘价 <= 基准 MA): 倾向防御，用 excess_return
    """
    work_df = df.select(["datetime", "vt_symbol", "open", "high", "close"]).sort(
        ["vt_symbol", "datetime"]
    )
    work_df = work_df.with_columns(pl.col("datetime").dt.weekday().alias("weekday"))
    
    # 1. 计算 Benchmark MA
    bench_df = work_df.filter(pl.col("vt_symbol") == benchmark_symbol).sort("datetime")
    bench_df = bench_df.with_columns([
        pl.col("close").rolling_mean(window_size=ma_period).alias("ma"),
    ])
    # 将 MA 前移一天，因为决策是在周一，我们用上周五/最近交易日的 regime
    bench_df = bench_df.with_columns([
        pl.col("close").shift(1).alias("prev_close"),
        pl.col("ma").shift(1).alias("prev_ma"),
    ])
    # Regime: True = Bull, False = Bear
    bench_df = bench_df.with_columns(
        (pl.col("prev_close") > pl.col("prev_ma")).alias("is_bull")
    )
    regime_map = dict(zip(bench_df["datetime"].to_list(), bench_df["is_bull"].to_list()))
    
    # 2. 计算收益和涨幅
    all_rets: list[dict] = []
    for symbol, grp in work_df.group_by("vt_symbol"):
        grp = grp.sort("datetime")
        sym_name = symbol[0] if isinstance(symbol, tuple) else symbol
        
        mondays = grp.filter(pl.col("weekday") == 1)
        if mondays.is_empty():
            continue
            
        dates = grp["datetime"]
        opens = grp["open"]
        closes = grp["close"]
        highs = grp["high"]
        
        for monday_dt in mondays["datetime"]:
            mask = dates > monday_dt
            future_indices = [i for i, v in enumerate(mask) if v]
            if not future_indices:
                continue
                
            horizon_end = min(len(future_indices), WEEK_HORIZON)
            horizon_idx = future_indices[:horizon_end]
            
            tuesday_open = opens[horizon_idx[0]]
            if tuesday_open is None or tuesday_open <= 0:
                continue
                
            last_close = closes[horizon_idx[-1]]
            max_high = max(highs[i] for i in horizon_idx)
            
            if last_close is None or last_close <= 0:
                continue
                
            ret = (last_close / tuesday_open) - 1.0
            high_touch_hit = max_high >= tuesday_open * (1 + rise_thresh)
            
            all_rets.append({
                "datetime": monday_dt,
                "vt_symbol": sym_name,
                "ret": ret,
                "high_touch_hit": high_touch_hit,
            })
            
    if not all_rets:
        return pl.DataFrame(
            schema={"datetime": pl.Datetime, "vt_symbol": pl.Utf8, "label": pl.Float64}
        )
        
    ret_df = pl.DataFrame(all_rets)
    
    # 3. 提取基准收益
    benchmark_rets = ret_df.filter(pl.col("vt_symbol") == benchmark_symbol).select(
        ["datetime", pl.col("ret").alias("bench_ret")]
    )
    if benchmark_rets.is_empty():
        ret_df = ret_df.with_columns(pl.lit(0.0).alias("bench_ret"))
    else:
        ret_df = ret_df.join(benchmark_rets, on="datetime", how="left").with_columns(pl.col("bench_ret").fill_null(0.0))
        
    # 4. 结合 Regime 计算最终标签
    ret_df = ret_df.filter(pl.col("vt_symbol") != benchmark_symbol)
    
    # 获取每个 monday_dt 的 regime
    ret_df = ret_df.with_columns([
        pl.col("datetime").map_elements(lambda dt: regime_map.get(dt, True), return_dtype=pl.Boolean).alias("is_bull")
    ])
    
    # Bull -> high_touch_hit
    # Bear -> ret - bench_ret > excess_thresh
    ret_df = ret_df.with_columns(
        pl.when(pl.col("is_bull"))
        .then(pl.col("high_touch_hit"))
        .otherwise((pl.col("ret") - pl.col("bench_ret")) > excess_thresh)
        .cast(pl.Float64)
        .alias("label")
    )
    
    result = ret_df.select(["datetime", "vt_symbol", "label"]).sort(["datetime", "vt_symbol"])
    return result
