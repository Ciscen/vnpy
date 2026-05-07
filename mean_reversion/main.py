import os
import json
import numpy as np
import pandas as pd
import akshare as ak
import plotly.graph_objects as go
from datetime import datetime
from itertools import product
from dashboard import build_dashboard

# 创建输出目录
output_dir = os.path.join(os.path.dirname(__file__), "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def fetch_data(symbol="003031", period="daily", start_date="20140101", end_date=None):
    """获取股票数据，默认近10年"""
    print(f"正在获取 {symbol} 的数据...")
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
        
    ak_symbol = f"sh{symbol}" if symbol.startswith(("6", "5")) else f"sz{symbol}"
    
    # 增加重试机制，以防网络波动
    max_retries = 3
    for attempt in range(max_retries):
        try:
            df = ak.stock_zh_a_daily(symbol=ak_symbol, start_date=start_date, end_date=end_date, adjust="qfq")
            if df is not None and not df.empty:
                break
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep((attempt + 1) * 2)
            else:
                raise e
    
    # stock_zh_a_daily 返回的是英文列名
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def run_backtest(df, ma_period, buy_threshold, sell_threshold, initial_cash=100000.0, fee_rate=0.001, slippage_rate=0.001):
    """
    执行回测 (包含趋势过滤、涨跌停限制、时间止损与基于成本的动态止损)
    """
    df = df.copy()
    df['MA'] = df['close'].rolling(window=ma_period).mean()
    df['Bias'] = (df['close'] - df['MA']) / df['MA']
    
    # 增加60日长期均线用作趋势过滤
    df['MA60'] = df['close'].rolling(window=60).mean()
    # 增加昨收价用于计算涨跌停
    df['pre_close'] = df['close'].shift(1)
    
    cash = initial_cash
    shares = 0
    equities = np.zeros(len(df))
    
    trades = []
    trade_logs = []
    entry_price = 0
    entry_date = None
    
    closes = df['close'].values
    opens = df['open'].values
    biases = df['Bias'].values
    ma60s = df['MA60'].values
    pre_closes = df['pre_close'].values
    dates = df['date'].values
    
    for i in range(len(df)):
        if i == 0 or np.isnan(biases[i-1]) or np.isnan(ma60s[i-1]):
            equities[i] = cash + shares * closes[i]
            continue
            
        yesterday_bias = biases[i-1]
        yesterday_ma60 = ma60s[i-1]
        yesterday_close = closes[i-1]
        pre_close = pre_closes[i]
        today_open = opens[i]
        current_date = pd.to_datetime(dates[i])
        
        # 涨跌停限制计算 (10% 限制的简单近似)
        limit_up = round(pre_close * 1.1, 2) if not np.isnan(pre_close) else float('inf')
        limit_down = round(pre_close * 0.9, 2) if not np.isnan(pre_close) else 0
        
        # 判断能否交易（停牌、一字涨跌停等无法交易）
        can_trade = today_open > 0 and today_open < limit_up and today_open > limit_down
        
        if can_trade:
            if shares == 0:
                # 买入逻辑
                # 过滤条件: 收盘价需在60日均线之上，避免主跌浪接飞刀
                if yesterday_bias < buy_threshold and yesterday_close > yesterday_ma60:
                    cost_price = today_open * (1 + slippage_rate)
                    
                    # 全仓买入，使用当前所有可用现金
                    trade_capital = cash
                    
                    # 可买股数，必须是100的整数倍
                    max_shares = int(trade_capital / (cost_price * (1 + fee_rate))) // 100 * 100
                    if max_shares > 0:
                        trade_amount = max_shares * cost_price
                        fee = trade_amount * fee_rate
                        cash -= (trade_amount + fee)
                        shares += max_shares
                        entry_price = cost_price
                        entry_date = current_date
                        
                        trade_logs.append({
                            'datetime': entry_date.strftime('%Y-%m-%d'),
                            'direction': 'Long',
                            'price': cost_price,
                            'volume': max_shares,
                            'reason': 'Bias Buy',
                            'entry_price': cost_price,
                            'pnl_pct': 0,
                            'hold_days': 0
                        })
            else:
                # 卖出逻辑
                hold_days = (current_date - entry_date).days
                
                # 触发条件: 
                # 1. 均值回归或止盈: yesterday_bias > sell_threshold
                # 2. 价格止损: current open drops 8% below entry price
                # 3. 时间止损: 持有超过 15 天且没有有效利润
                stop_loss_triggered = today_open < entry_price * 0.92
                time_stop_triggered = hold_days > 15
                take_profit_triggered = yesterday_bias > sell_threshold
                
                if stop_loss_triggered or time_stop_triggered or take_profit_triggered:
                    sell_price = today_open * (1 - slippage_rate)
                    trade_amount = shares * sell_price
                    fee = trade_amount * fee_rate
                    cash += (trade_amount - fee)
                    
                    ret = (sell_price * (1 - fee_rate)) / (entry_price * (1 + fee_rate)) - 1
                    trades.append(ret)
                    
                    if stop_loss_triggered:
                        reason = "Stop Loss (-8%)"
                    elif take_profit_triggered:
                        reason = "Take Profit"
                    else:
                        reason = "Time Stop (15 Days)"
                        
                    trade_logs.append({
                        'datetime': current_date.strftime('%Y-%m-%d'),
                        'direction': 'Short',
                        'price': sell_price,
                        'volume': shares,
                        'reason': reason,
                        'entry_price': entry_price,
                        'pnl_pct': ret * 100,
                        'hold_days': hold_days
                    })
                    shares = 0
                    
        equities[i] = cash + shares * closes[i]
        
    df['equity'] = equities
    df['daily_return'] = df['equity'].pct_change().fillna(0)
    df['net_pnl'] = df['equity'].diff().fillna(0)
    
    days = len(df)
    if days == 0: return None
    
    total_return = df['equity'].iloc[-1] / initial_cash - 1
    annual_return = (1 + total_return) ** (252 / days) - 1 if (1 + total_return) > 0 else -1
    
    cummax = df['equity'].cummax()
    df['drawdown'] = (df['equity'] - cummax) / cummax
    max_dd = abs(df['drawdown'].min())
    
    mean_ret = df['daily_return'].mean()
    std_ret = df['daily_return'].std()
    sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 1e-6 else 0
    
    win_rate = sum(1 for t in trades if t > 0) / len(trades) if trades else 0
    
    return {
        'ma_period': ma_period,
        'buy_threshold': buy_threshold,
        'sell_threshold': sell_threshold,
        'total_return': total_return,
        'annual_return': annual_return,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'trades_count': len(trades),
        'df': df,
        'trade_logs': pd.DataFrame(trade_logs) if trade_logs else pd.DataFrame(columns=['datetime', 'direction', 'price', 'volume', 'reason', 'entry_price', 'pnl_pct', 'hold_days'])
    }

def optimize_parameters(train_df):
    """网格搜索最优参数"""
    ma_periods = [20, 30, 40, 60]
    buy_thresholds = [-0.02, -0.03, -0.04, -0.05, -0.06, -0.08]
    sell_thresholds = [0.02, 0.03, 0.04, 0.05]
    
    print(f"开始在训练集(长度: {len(train_df)}天)上进行网格搜索...")
    
    results = []
    total = len(ma_periods) * len(buy_thresholds) * len(sell_thresholds)
    count = 0
    
    for ma, bt, st in product(ma_periods, buy_thresholds, sell_thresholds):
        count += 1
        res = run_backtest(train_df, ma, bt, st)
        if res:
            results.append(res)
        if count % 20 == 0:
            print(f"进度: {count}/{total}")
            
    # 按夏普比率降序，且最大回撤 < 20%
    valid_results = [r for r in results if r['max_dd'] < 0.2]
    if not valid_results:
        print("没有满足回撤要求的参数，放宽要求，直接按夏普比率排序。")
        valid_results = results
        
    valid_results.sort(key=lambda x: x['sharpe'], reverse=True)
    best_res = valid_results[0]
    
    print("-------------------------------------------------")
    print(f"寻找到最优参数:")
    print(f"MA周期: {best_res['ma_period']}")
    print(f"买入阈值: {best_res['buy_threshold']}")
    print(f"卖出阈值: {best_res['sell_threshold']}")
    print(f"训练集 夏普比率: {best_res['sharpe']:.2f}, 最大回撤: {best_res['max_dd']*100:.2f}%, 年化收益: {best_res['annual_return']*100:.2f}%")
    print("-------------------------------------------------")
    
    return best_res['ma_period'], best_res['buy_threshold'], best_res['sell_threshold']

def plot_equity_curve(full_res, train_end_date):
    """绘制权益曲线并标注训练集和测试集"""
    df = full_res['df']
    
    fig = go.Figure()
    
    # 添加资金曲线
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['equity'],
        mode='lines',
        name='资金曲线',
        line=dict(color='blue', width=2)
    ))
    
    # 标注基准线（初始资金）
    fig.add_hline(y=100000, line_dash="dash", line_color="gray")
    
    # 添加训练/测试分割线
    fig.add_vline(x=train_end_date, line_width=2, line_dash="dash", line_color="red")
    
    # 标注区域
    fig.add_annotation(
        x=df['date'].iloc[len(df)//2 - len(df)//4],
        y=df['equity'].max() * 0.95,
        text="训练集 (In-Sample)",
        showarrow=False,
        font=dict(size=14, color="black")
    )
    
    fig.add_annotation(
        x=df['date'].iloc[-1 * (len(df) - len(df)//2) + len(df)//8],
        y=df['equity'].max() * 0.95,
        text="测试集 (Out-of-Sample)",
        showarrow=False,
        font=dict(size=14, color="black")
    )

    fig.update_layout(
        title="均值回归策略回测资金曲线 (支持交互)",
        xaxis_title="日期",
        yaxis_title="资金 (元)",
        template="plotly_white",
        hovermode="x unified",
        height=600,
        width=1000
    )
    
    html_path = os.path.join(output_dir, "equity_curve.html")
    fig.write_html(html_path)
    print(f"资金曲线图已保存至: {html_path}")

def main():
    symbol = "601318" # 中国平安
    
    # 获取近10年的数据
    start_date = (datetime.now().replace(year=datetime.now().year - 10)).strftime("%Y%m%d")
    df = fetch_data(symbol=symbol, start_date=start_date)
    
    print(f"获取到数据总量: {len(df)} 行。起始日期: {df['date'].iloc[0].date()}, 结束日期: {df['date'].iloc[-1].date()}")
    
    # 划分训练集和测试集 (80% / 20%)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    train_end_date = train_df['date'].iloc[-1]
    
    print(f"训练集区间: {train_df['date'].iloc[0].date()} 至 {train_end_date.date()}")
    print(f"测试集区间: {test_df['date'].iloc[0].date()} 至 {test_df['date'].iloc[-1].date()}")
    
    # 1. 寻找最优参数
    best_ma, best_bt, best_st = optimize_parameters(train_df)
    
    # 2. 全样本回测（使用最优参数运行整个数据集，并生成报告）
    print("使用最优参数进行全样本回测...")
    full_res = run_backtest(df, best_ma, best_bt, best_st)
    
    # 计算测试集上的指标用于报告
    test_res = run_backtest(test_df, best_ma, best_bt, best_st)
    
    print("\n========== 回测结果报告 ==========")
    print(f"[最优参数] MA周期={best_ma}, 买入阈值={best_bt}, 卖出阈值={best_st}")
    print(f"\n【全样本表现】")
    print(f"总收益:      {full_res['total_return']*100:.2f}%")
    print(f"年化收益:    {full_res['annual_return']*100:.2f}%")
    print(f"最大回撤:    {full_res['max_dd']*100:.2f}%")
    print(f"夏普比率:    {full_res['sharpe']:.2f}")
    print(f"胜率:        {full_res['win_rate']*100:.2f}%")
    print(f"交易次数:    {full_res['trades_count']}")
    
    if test_res:
        print(f"\n【测试集(盲测)表现】")
        print(f"总收益:      {test_res['total_return']*100:.2f}%")
        print(f"年化收益:    {test_res['annual_return']*100:.2f}%")
        print(f"最大回撤:    {test_res['max_dd']*100:.2f}%")
        print(f"夏普比率:    {test_res['sharpe']:.2f}")
        print(f"胜率:        {test_res['win_rate']*100:.2f}%")
        print(f"交易次数:    {test_res['trades_count']}")
    print("==================================\n")
    
    # 保存指标到JSON
    stats = {
        "best_params": {
            "ma_period": int(best_ma),
            "buy_threshold": float(best_bt),
            "sell_threshold": float(best_st)
        },
        "full_sample": {
            "total_return": float(full_res['total_return']),
            "annual_return": float(full_res['annual_return']),
            "max_drawdown": float(full_res['max_dd']),
            "sharpe_ratio": float(full_res['sharpe']),
            "win_rate": float(full_res['win_rate']),
            "trades": int(full_res['trades_count'])
        },
        "test_sample": {
            "total_return": float(test_res['total_return']) if test_res else 0,
            "annual_return": float(test_res['annual_return']) if test_res else 0,
            "max_drawdown": float(test_res['max_dd']) if test_res else 0,
            "sharpe_ratio": float(test_res['sharpe']) if test_res else 0,
            "win_rate": float(test_res['win_rate']) if test_res else 0,
            "trades": int(test_res['trades_count']) if test_res else 0
        }
    }
    
    with open(os.path.join(output_dir, "statistics.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
        
    # 绘制拼接图
    plot_equity_curve(full_res, train_end_date)
    
    # 构建并输出综合仪表盘 HTML
    print("正在生成综合仪表盘(包含基准对比与交易明细)...")
    html_content = build_dashboard(stats, full_res['trade_logs'], full_res['df'], symbol)
    dashboard_path = os.path.join(output_dir, "dashboard.html")
    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"综合仪表盘已保存至: {dashboard_path}")
    
if __name__ == "__main__":
    main()
