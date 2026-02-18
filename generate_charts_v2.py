"""
Generate professional charts for Backtest V2
- Equity curve (% return)
- Drawdown chart
- Skew impact analysis
- P&L distribution with capital context
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import json

# Style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

COLORS = {
    'blue': '#2563EB',
    'red': '#DC2626',
    'green': '#16A34A',
    'orange': '#EA580C',
    'gray': '#6B7280',
    'light_blue': '#BFDBFE',
    'light_red': '#FECACA',
    'dark': '#1F2937',
}


def load_data():
    """Load backtest results and equity curves."""
    results_flat = pd.read_csv('backtest_v2_flat.csv')
    results_skew = pd.read_csv('backtest_v2_skew.csv')
    equity_flat = np.load('equity_flat.npy')
    equity_skew = np.load('equity_skew.npy')

    with open('backtest_v2_metrics.json', 'r') as f:
        metrics = json.load(f)

    return results_flat, results_skew, equity_flat, equity_skew, metrics


def plot_main_dashboard(results_flat, equity_flat, metrics_flat):
    """
    Main dashboard: 6-panel figure
    1. Equity curve (% return)
    2. Drawdown chart
    3. P&L distribution
    4. Win rate by stock
    5. Return per trade scatter
    6. Summary statistics table
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Short Straddle Backtest — Portfolio Performance Dashboard',
                 fontsize=14, fontweight='bold', y=0.98)

    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # --- 1. Equity curve (% return) ---
    ax1 = fig.add_subplot(gs[0, :2])
    returns_pct = (equity_flat / equity_flat[0] - 1) * 100
    ax1.plot(returns_pct, color=COLORS['blue'], linewidth=1.5)
    ax1.fill_between(range(len(returns_pct)), 0, returns_pct,
                     where=returns_pct >= 0, alpha=0.15, color=COLORS['blue'])
    ax1.fill_between(range(len(returns_pct)), 0, returns_pct,
                     where=returns_pct < 0, alpha=0.15, color=COLORS['red'])

    # Mark OOS boundary
    oos_idx = len(results_flat[pd.to_datetime(results_flat['earnings_date']).dt.year < 2024])
    ax1.axvline(x=oos_idx, color=COLORS['gray'], linestyle='--', alpha=0.7, linewidth=0.8)
    ax1.text(oos_idx + 2, returns_pct.max() * 0.9, 'OOS starts',
             fontsize=8, color=COLORS['gray'])

    ax1.set_ylabel('Cumulative Return (%)')
    ax1.set_xlabel('Trade Number')
    ax1.set_title(f'Equity Curve — ${metrics_flat["initial_capital"]/1000:.0f}K Initial Capital')
    ax1.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
    ax1.grid(True, alpha=0.2)

    # --- 2. Drawdown chart ---
    ax2 = fig.add_subplot(gs[1, :2])
    running_max = np.maximum.accumulate(equity_flat)
    dd_pct = (running_max - equity_flat) / running_max * 100

    ax2.fill_between(range(len(dd_pct)), 0, -dd_pct, color=COLORS['red'], alpha=0.4)
    ax2.plot(-dd_pct, color=COLORS['red'], linewidth=0.8)

    max_dd_idx = np.argmax(dd_pct)
    ax2.annotate(f'Max DD: -{dd_pct[max_dd_idx]:.2f}%',
                xy=(max_dd_idx, -dd_pct[max_dd_idx]),
                xytext=(max_dd_idx + 20, -dd_pct[max_dd_idx] - 0.5),
                fontsize=8, color=COLORS['red'],
                arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=0.8))

    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Trade Number')
    ax2.set_title('Underwater Chart (Drawdown from Peak)')
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim(ax2.get_ylim()[0] * 1.2, 0.5)

    # --- 3. P&L distribution ---
    ax3 = fig.add_subplot(gs[0, 2])
    pnl = results_flat['total_pnl']
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    ax3.hist(wins, bins=30, alpha=0.7, color=COLORS['green'], label=f'Wins ({len(wins)})')
    ax3.hist(losses, bins=15, alpha=0.7, color=COLORS['red'], label=f'Losses ({len(losses)})')
    ax3.axvline(pnl.mean(), color=COLORS['dark'], linestyle='--', linewidth=1,
               label=f'Mean: ${pnl.mean():.0f}')
    ax3.set_xlabel('P&L per Trade ($)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('P&L Distribution')
    ax3.legend(fontsize=7)

    # --- 4. Win rate by stock ---
    ax4 = fig.add_subplot(gs[1, 2])
    stock_wr = results_flat.groupby('ticker')['total_pnl'].apply(
        lambda x: (x > 0).mean() * 100
    ).sort_values()

    colors_wr = [COLORS['green'] if wr >= 70 else COLORS['orange'] if wr >= 50 else COLORS['red']
                 for wr in stock_wr.values]
    bars = ax4.barh(stock_wr.index, stock_wr.values, color=colors_wr, alpha=0.8, height=0.6)

    for bar, val in zip(bars, stock_wr.values):
        ax4.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}%', va='center', fontsize=7)

    ax4.set_xlabel('Win Rate (%)')
    ax4.set_title('Win Rate by Stock')
    ax4.set_xlim(0, 105)

    # --- 5. Return per trade vs stock move ---
    ax5 = fig.add_subplot(gs[2, :2])
    colors_scatter = [COLORS['green'] if p > 0 else COLORS['red']
                     for p in results_flat['total_pnl']]
    ax5.scatter(results_flat['spot_move_abs_pct'], results_flat['return_on_capital'],
               c=colors_scatter, alpha=0.5, s=15, edgecolors='none')

    # Correlation line
    z = np.polyfit(results_flat['spot_move_abs_pct'], results_flat['return_on_capital'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, results_flat['spot_move_abs_pct'].max(), 100)
    ax5.plot(x_line, p(x_line), color=COLORS['dark'], linewidth=1, linestyle='--', alpha=0.7)

    corr = results_flat[['spot_move_abs_pct', 'return_on_capital']].corr().iloc[0, 1]
    ax5.set_xlabel('Absolute Stock Move (%)')
    ax5.set_ylabel('Return on Capital (%)')
    ax5.set_title(f'Trade Return vs Stock Move (corr = {corr:.2f})')
    ax5.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
    ax5.grid(True, alpha=0.2)

    # --- 6. Summary metrics table ---
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')

    m = metrics_flat
    table_data = [
        ['Initial Capital', f'${m["initial_capital"]:,.0f}'],
        ['Final Capital', f'${m["final_capital"]:,.0f}'],
        ['Total Return', f'{m["total_return_pct"]:.2f}%'],
        ['CAGR', f'{m["cagr"]:.2f}%'],
        ['Ann. Volatility', f'{m["annualized_vol"]:.2f}%'],
        ['Sharpe (ann.)', f'{m["sharpe_annualized"]:.2f}'],
        ['Sortino', f'{m["sortino"]:.2f}'],
        ['Max Drawdown', f'{m["max_drawdown_pct"]:.2f}%'],
        ['Calmar Ratio', f'{m["calmar"]:.2f}'],
        ['Win Rate', f'{m["win_rate"]:.1f}%'],
        ['Profit Factor', f'{m["profit_factor"]:.2f}'],
        ['Trades', f'{m["total_trades"]}'],
        ['Avg Margin Used', f'{m["avg_margin_pct"]:.1f}%'],
    ]

    table = ax6.table(cellText=table_data, colLabels=['Metric', 'Value'],
                     loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)

    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(COLORS['dark'])
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#F3F4F6')
        cell.set_edgecolor('#E5E7EB')

    ax6.set_title('Summary Statistics', fontsize=10, fontweight='bold', pad=10)

    plt.savefig('docs/backtest_v2_dashboard.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: backtest_v2_dashboard.png")
    plt.close()


def plot_skew_analysis(results_flat, results_skew, equity_flat, equity_skew, metrics):
    """
    Skew analysis: 4-panel figure comparing flat IV vs skew model.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Volatility Skew Impact Analysis',
                 fontsize=13, fontweight='bold', y=0.98)

    # --- 1. Equity curves comparison ---
    ax = axes[0, 0]
    ret_flat = (equity_flat / equity_flat[0] - 1) * 100
    ret_skew = (equity_skew / equity_skew[0] - 1) * 100

    ax.plot(ret_flat, color=COLORS['blue'], linewidth=1.5, label='Flat IV (no skew)')
    ax.plot(ret_skew, color=COLORS['orange'], linewidth=1.5, label='With Skew')
    ax.set_ylabel('Cumulative Return (%)')
    ax.set_xlabel('Trade Number')
    ax.set_title('Equity Curve: Flat IV vs Skew Model')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # --- 2. Per-trade P&L comparison ---
    ax = axes[0, 1]
    pnl_diff = results_skew['total_pnl'] - results_flat['total_pnl']

    ax.bar(range(len(pnl_diff)), pnl_diff,
           color=[COLORS['green'] if d > 0 else COLORS['red'] for d in pnl_diff],
           alpha=0.6, width=1.0)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Trade Number')
    ax.set_ylabel('P&L Difference ($)')
    ax.set_title(f'Skew Impact per Trade (avg: ${pnl_diff.mean():.1f})')
    ax.grid(True, alpha=0.2)

    # --- 3. Skew P&L impact by stock move direction ---
    ax = axes[1, 0]
    df = results_skew.copy()
    df['move_direction'] = np.where(df['spot_move_pct'] >= 0, 'Stock Up', 'Stock Down')
    df['pnl_diff'] = results_skew['total_pnl'] - results_flat['total_pnl']

    up_trades = df[df['move_direction'] == 'Stock Up']
    down_trades = df[df['move_direction'] == 'Stock Down']

    ax.scatter(up_trades['spot_move_abs_pct'], up_trades['pnl_diff'],
              color=COLORS['green'], alpha=0.4, s=15, label=f'Stock Up (n={len(up_trades)})')
    ax.scatter(down_trades['spot_move_abs_pct'], down_trades['pnl_diff'],
              color=COLORS['red'], alpha=0.4, s=15, label=f'Stock Down (n={len(down_trades)})')

    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Absolute Stock Move (%)')
    ax.set_ylabel('Skew P&L Difference ($)')
    ax.set_title('Skew Impact by Move Direction')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # --- 4. Comparison metrics table ---
    ax = axes[1, 1]
    ax.axis('off')

    mf = metrics['flat']
    ms = metrics['skew']

    table_data = [
        ['Total Return', f'{mf["total_return_pct"]:.2f}%', f'{ms["total_return_pct"]:.2f}%'],
        ['CAGR', f'{mf["cagr"]:.2f}%', f'{ms["cagr"]:.2f}%'],
        ['Sharpe (ann.)', f'{mf["sharpe_annualized"]:.2f}', f'{ms["sharpe_annualized"]:.2f}'],
        ['Max Drawdown', f'{mf["max_drawdown_pct"]:.2f}%', f'{ms["max_drawdown_pct"]:.2f}%'],
        ['Calmar', f'{mf["calmar"]:.2f}', f'{ms["calmar"]:.2f}'],
        ['Win Rate', f'{mf["win_rate"]:.1f}%', f'{ms["win_rate"]:.1f}%'],
        ['Profit Factor', f'{mf["profit_factor"]:.2f}', f'{ms["profit_factor"]:.2f}'],
        ['Avg P&L/Trade', f'${mf["avg_pnl_per_trade"]:.0f}', f'${ms["avg_pnl_per_trade"]:.0f}'],
    ]

    table = ax.table(cellText=table_data,
                    colLabels=['Metric', 'Flat IV', 'With Skew'],
                    loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(COLORS['dark'])
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#F3F4F6')
        cell.set_edgecolor('#E5E7EB')

    ax.set_title('Flat IV vs Skew: Metric Comparison', fontsize=10, fontweight='bold', pad=10)

    plt.savefig('docs/backtest_v2_skew_analysis.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: backtest_v2_skew_analysis.png")
    plt.close()


def plot_annual_performance(results_flat, metrics_flat):
    """Annual breakdown: P&L by year and in-sample vs OOS."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Annual Performance Breakdown', fontsize=13, fontweight='bold')

    df = results_flat.copy()
    df['year'] = pd.to_datetime(df['earnings_date']).dt.year

    # --- 1. Annual P&L ---
    ax = axes[0]
    annual = df.groupby('year').agg(
        total_pnl=('total_pnl', 'sum'),
        trades=('total_pnl', 'count'),
        win_rate=('total_pnl', lambda x: (x > 0).mean() * 100)
    )

    colors_bar = [COLORS['blue'] if y < 2024 else COLORS['orange'] for y in annual.index]
    bars = ax.bar(annual.index, annual['total_pnl'], color=colors_bar, alpha=0.8, width=0.6)

    for bar, (yr, row) in zip(bars, annual.iterrows()):
        y_pos = bar.get_height() if bar.get_height() > 0 else bar.get_height() - 100
        ax.text(bar.get_x() + bar.get_width()/2, y_pos + 50,
               f'WR:{row["win_rate"]:.0f}%\nn={row["trades"]:.0f}',
               ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Year')
    ax.set_ylabel('Total P&L ($)')
    ax.set_title('Annual P&L (blue=in-sample, orange=OOS)')
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
    ax.grid(True, alpha=0.2, axis='y')

    # --- 2. Monthly return heatmap-style ---
    ax = axes[1]
    df['month'] = pd.to_datetime(df['earnings_date']).dt.month
    monthly = df.groupby(['year', 'month'])['return_on_capital'].sum().unstack(fill_value=0)

    im = ax.imshow(monthly.values, cmap='RdYlGn', aspect='auto', vmin=-2, vmax=2)

    ax.set_xticks(range(12))
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                        'Jul','Aug','Sep','Oct','Nov','Dec'], fontsize=7)
    ax.set_yticks(range(len(monthly.index)))
    ax.set_yticklabels(monthly.index)
    ax.set_title('Return on Capital by Year/Month (%)')

    # Add text
    for i in range(monthly.shape[0]):
        for j in range(monthly.shape[1]):
            val = monthly.values[i, j]
            if abs(val) > 0.01:
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=6,
                       color='white' if abs(val) > 1 else 'black')

    plt.colorbar(im, ax=ax, shrink=0.6, label='Return (%)')

    plt.savefig('docs/backtest_v2_annual.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: backtest_v2_annual.png")
    plt.close()


if __name__ == "__main__":
    print("Loading data...")
    results_flat, results_skew, equity_flat, equity_skew, metrics = load_data()

    print("Generating main dashboard...")
    plot_main_dashboard(results_flat, equity_flat, metrics['flat'])

    print("Generating skew analysis...")
    plot_skew_analysis(results_flat, results_skew, equity_flat, equity_skew, metrics)

    print("Generating annual performance...")
    plot_annual_performance(results_flat, metrics['flat'])

    print("\nAll charts generated.")
