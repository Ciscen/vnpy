import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import akshare as ak

def _build_equity_chart(df: pd.DataFrame) -> go.Figure:
    dates = df["date"].dt.strftime('%Y-%m-%d').tolist()
    balance = df["equity"].tolist()
    drawdown = df["drawdown"].tolist()

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["资金曲线 (Balance)", "回撤 (Drawdown)"],
        vertical_spacing=0.08,
    )

    fig.add_trace(
        go.Scatter(x=dates, y=balance, mode="lines", name="Balance"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=dates, y=[d * 100 for d in drawdown], fill="tozeroy",
                   fillcolor="rgba(255,0,0,0.15)", mode="lines",
                   line=dict(color="red"), name="Drawdown (%)"),
        row=2, col=1,
    )

    fig.update_layout(
        height=800, width=1100,
        title_text="均值回归策略 - 回测资金与回撤",
        showlegend=False,
        plot_bgcolor="white",
    )
    for i in range(1, 3):
        fig.update_xaxes(showgrid=True, gridcolor="LightGray", row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="LightGray", row=i, col=1)

    return fig

def _build_excess_return_chart(strategy_df: pd.DataFrame, symbol: str) -> go.Figure:
    # 获取沪深300作为基准
    try:
        start = strategy_df['date'].iloc[0].strftime("%Y%m%d")
        end = strategy_df['date'].iloc[-1].strftime("%Y%m%d")
        
        # 考虑到 ak.stock_zh_index_daily 的起止日期有时候并不严谨，获取一段时间并裁剪
        for attempt in range(3):
            try:
                bm_df = ak.stock_zh_index_daily(symbol="sh000300")
                break
            except:
                import time
                time.sleep(2)
        
        bm_df['date'] = pd.to_datetime(bm_df['date'])
        bm_df = bm_df[(bm_df['date'] >= strategy_df['date'].iloc[0]) & (bm_df['date'] <= strategy_df['date'].iloc[-1])]
        
        # 对齐日期，保留策略数据的 close 作为个股基准
        merged = pd.merge(strategy_df[['date', 'equity', 'close']], bm_df[['date', 'close']], on='date', how='left', suffixes=('', '_bm')).ffill()
        
        # 计算相对收益
        strategy_ret = merged['equity'] / merged['equity'].iloc[0] - 1
        bm_ret = merged['close_bm'] / merged['close_bm'].iloc[0] - 1
        stock_ret = merged['close'] / merged['close'].iloc[0] - 1
        excess_ret = strategy_ret - stock_ret
        
        dates_str = merged['date'].dt.strftime('%Y-%m-%d').tolist()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(go.Scatter(x=dates_str, y=(strategy_ret * 100).tolist(), mode='lines', name='策略收益(%)', line=dict(color='red')), secondary_y=False)
        fig.add_trace(go.Scatter(x=dates_str, y=(stock_ret * 100).tolist(), mode='lines', name=f'基准收益-{symbol}(%)', line=dict(color='orange')), secondary_y=False)
        fig.add_trace(go.Scatter(x=dates_str, y=(bm_ret * 100).tolist(), mode='lines', name='基准收益-沪深300(%)', line=dict(color='blue')), secondary_y=False)
        fig.add_trace(go.Scatter(x=dates_str, y=(excess_ret * 100).tolist(), mode='lines', name='相对个股超额收益(%)', line=dict(color='green'), fill='tozeroy', fillcolor='rgba(39, 174, 96, 0.2)'), secondary_y=False)
        
        fig.update_layout(
            height=600, width=1100,
            title_text="策略与基准对比 (超额收益)",
            plot_bgcolor="white",
            hovermode="x unified"
        )
        fig.update_xaxes(showgrid=True, gridcolor="LightGray")
        fig.update_yaxes(showgrid=True, gridcolor="LightGray", secondary_y=False)
        return fig
    except Exception as e:
        print(f"基准图表生成失败: {e}")
        return None

def build_dashboard(stats, trade_logs, df, symbol):
    equity_fig = _build_equity_chart(df)
    excess_fig = _build_excess_return_chart(df, symbol)
    
    equity_html = pio.to_html(equity_fig, include_plotlyjs=False, full_html=False, div_id="equity-chart")
    excess_html = pio.to_html(excess_fig, include_plotlyjs=False, full_html=False, div_id="excess-chart") if excess_fig else ""
    
    trades_json = trade_logs.to_json(orient='records', force_ascii=False)
    
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>均值回归策略回测仪表盘</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f6fa; color: #2c3e50; }}
.header {{ background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 24px 32px; }}
.header h1 {{ font-size: 24px; font-weight: 600; }}
.tab-bar {{ display: flex; background: #fff; border-bottom: 2px solid #e0e0e0; padding: 0 16px; position: sticky; top: 0; z-index: 100; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
.tab-btn {{ padding: 14px 24px; cursor: pointer; border: none; background: none; font-size: 14px; font-weight: 500; color: #666; border-bottom: 3px solid transparent; transition: all 0.2s; }}
.tab-btn:hover {{ color: #3498db; background: #f8f9fa; }}
.tab-btn.active {{ color: #3498db; border-bottom-color: #3498db; }}
.tab-content {{ display: none; padding: 24px; max-width: 1200px; margin: 0 auto; }}
.tab-content.active {{ display: block; }}
.card {{ background: #fff; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
.metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 20px; }}
.metric {{ background: #fff; border-radius: 8px; padding: 16px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
.metric .value {{ font-size: 28px; font-weight: 700; }}
.metric .label {{ font-size: 12px; color: #95a5a6; margin-top: 4px; }}
.positive {{ color: #27ae60; }}
.negative {{ color: #e74c3c; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{ background: #f8f9fa; padding: 10px 12px; text-align: left; border-bottom: 2px solid #e0e0e0; position: sticky; top: 0; }}
td {{ padding: 8px 12px; border-bottom: 1px solid #f0f0f0; }}
tr:hover {{ background: #f8f9fa; }}
.table-wrap {{ max-height: 600px; overflow-y: auto; }}
</style>
</head>
<body>
<div class="header">
  <h1>均值回归策略回测仪表盘 (基于 {symbol})</h1>
</div>
<div class="tab-bar">
  <button class="tab-btn active" onclick="switchTab('overview')">策略总览</button>
  <button class="tab-btn" onclick="switchTab('trades')">交易明细</button>
</div>

<div id="tab-overview" class="tab-content active">
  <div class="metrics-grid">
    <div class="metric"><div class="value { 'positive' if stats['full_sample']['total_return'] >= 0 else 'negative' }">{stats['full_sample']['total_return']*100:.2f}%</div><div class="label">总收益率</div></div>
    <div class="metric"><div class="value { 'positive' if stats['full_sample']['annual_return'] >= 0 else 'negative' }">{stats['full_sample']['annual_return']*100:.2f}%</div><div class="label">年化收益率</div></div>
    <div class="metric"><div class="value negative">{stats['full_sample']['max_drawdown']*100:.2f}%</div><div class="label">最大回撤</div></div>
    <div class="metric"><div class="value">{stats['full_sample']['sharpe_ratio']:.2f}</div><div class="label">夏普比率</div></div>
    <div class="metric"><div class="value">{stats['full_sample']['win_rate']*100:.2f}%</div><div class="label">胜率</div></div>
    <div class="metric"><div class="value">{stats['full_sample']['trades']}</div><div class="label">总交易次数</div></div>
  </div>
  <div class="card">{equity_html}</div>
  <div class="card">{excess_html}</div>
</div>

<div id="tab-trades" class="tab-content">
  <div class="card">
    <h3>交易记录</h3>
    <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>日期</th><th>方向</th><th>价格</th><th>数量</th><th>原因</th><th>入场价</th><th>盈亏(%)</th><th>持仓天数</th>
            </tr>
          </thead>
          <tbody id="trade-tbody"></tbody>
        </table>
    </div>
  </div>
</div>

<script>
const trades = {trades_json};
function switchTab(name) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}}
function renderTrades() {{
  const tbody = document.getElementById('trade-tbody');
  tbody.innerHTML = trades.map(t => {{
    const dirColor = t.direction === 'Long' ? '#27ae60' : '#e74c3c';
    const pnlClass = t.pnl_pct >= 0 ? 'positive' : 'negative';
    return `<tr>
      <td>${{t.datetime}}</td>
      <td style="color:${{dirColor}};font-weight:bold">${{t.direction}}</td>
      <td>${{t.price.toFixed(2)}}</td>
      <td>${{t.volume}}</td>
      <td>${{t.reason}}</td>
      <td>${{t.entry_price ? t.entry_price.toFixed(2) : '-'}}</td>
      <td class="${{t.direction === 'Short' ? pnlClass : ''}}">${{t.direction === 'Short' ? t.pnl_pct.toFixed(2) + '%' : '-'}}</td>
      <td>${{t.direction === 'Short' ? t.hold_days : '-'}}</td>
    </tr>`;
  }}).join('');
}}
renderTrades();
</script>
</body>
</html>'''
    return html
