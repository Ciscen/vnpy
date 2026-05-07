import pandas as pd
from main import fetch_data, run_backtest
from datetime import datetime

def main():
    # 按照此前寻找到的最优参数：MA周期=20, 买入阈值=-0.05, 卖出阈值=0.02
    symbol = "000063"
    start_date = (datetime.now().replace(year=datetime.now().year - 10)).strftime("%Y%m%d")
    df = fetch_data(symbol=symbol, start_date=start_date)
    
    print("重新运行回测获取交易日志...")
    res = run_backtest(df, ma_period=20, buy_threshold=-0.05, sell_threshold=0.02)
    
    trade_logs = res['trade_logs']
    
    # 仅保留卖出记录，因为卖出记录包含了盈亏(pnl_pct)、持仓天数(hold_days)和退出原因(reason)
    sells = trade_logs[trade_logs['direction'] == 'Short'].copy()
    
    if sells.empty:
        print("没有交易记录。")
        return
    
    sells['pnl_pct'] = sells['pnl_pct'].astype(float)
    sells['hold_days'] = sells['hold_days'].astype(int)
    
    total_trades = len(sells)
    winning_trades = sells[sells['pnl_pct'] > 0]
    losing_trades = sells[sells['pnl_pct'] <= 0]
    
    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    win_rate = win_count / total_trades if total_trades > 0 else 0
    
    avg_win = winning_trades['pnl_pct'].mean() if win_count > 0 else 0
    avg_loss = losing_trades['pnl_pct'].mean() if loss_count > 0 else 0
    
    avg_hold_win = winning_trades['hold_days'].mean() if win_count > 0 else 0
    avg_hold_loss = losing_trades['hold_days'].mean() if loss_count > 0 else 0
    
    gross_profit = winning_trades['pnl_pct'].sum()
    gross_loss = abs(losing_trades['pnl_pct'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    print("\n========== 交易数据深度分析 ==========")
    print(f"总交易次数 (平仓): {total_trades}")
    print(f"胜率: {win_rate * 100:.2f}% ({win_count} 胜 / {loss_count} 负)")
    print(f"平均盈利: {avg_win:.2f}% | 平均亏损: {avg_loss:.2f}%")
    print(f"盈亏比 (Avg Win / Avg Loss): {abs(avg_win / avg_loss):.2f}" if avg_loss != 0 else "盈亏比: N/A")
    print(f"利润因子 (总盈利 / 总亏损): {profit_factor:.2f}")
    print(f"盈利交易平均持仓天数: {avg_hold_win:.1f} 天")
    print(f"亏损交易平均持仓天数: {avg_hold_loss:.1f} 天")
    
    print("\n--- 按平仓原因分类统计 ---")
    reasons = sells['reason'].unique()
    for reason in reasons:
        subset = sells[sells['reason'] == reason]
        count = len(subset)
        avg_pnl = subset['pnl_pct'].mean()
        win_c = len(subset[subset['pnl_pct'] > 0])
        win_r = win_c / count if count > 0 else 0
        avg_h = subset['hold_days'].mean()
        print(f"[{reason}]")
        print(f"  次数: {count} ({count/total_trades*100:.1f}%) | 胜率: {win_r*100:.1f}%")
        print(f"  平均盈亏: {avg_pnl:.2f}% | 平均持仓: {avg_h:.1f} 天")

    print("\n========== 优化点建议分析 ==========")
    
if __name__ == "__main__":
    main()