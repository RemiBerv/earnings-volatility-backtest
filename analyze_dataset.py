"""
Dataset Analysis and Visualization
Explore patterns in the earnings options dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

def load_and_validate(filepath='earnings_options_dataset.csv'):
    """Load and validate dataset"""
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    
    print(f"\n{'='*70}")
    print("DATASET VALIDATION")
    print(f"{'='*70}")
    
    # Basic info
    print(f"\nðŸ“Š Shape: {df.shape[0]} events, {df.shape[1]} columns")
    print(f"ðŸ“… Date Range: {df['earnings_date'].min()} to {df['earnings_date'].max()}")
    print(f"ðŸ¢ Stocks: {df['ticker'].nunique()} unique tickers")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\nâš ï¸  Missing values detected:")
        print(missing[missing > 0])
    else:
        print("\nâœ… No missing values")
    
    # Data quality checks
    print(f"\n{'='*70}")
    print("DATA QUALITY CHECKS")
    print(f"{'='*70}")
    
    # Realized vol should be positive
    invalid_rv = (df['realized_vol'] <= 0).sum()
    print(f"Invalid realized vol (â‰¤0): {invalid_rv} ({invalid_rv/len(df)*100:.1f}%)")
    
    # IV should be positive
    invalid_iv = ((df['near_iv_entry'] <= 0) | (df['far_iv_entry'] <= 0)).sum()
    print(f"Invalid IV (â‰¤0): {invalid_iv} ({invalid_iv/len(df)*100:.1f}%)")
    
    # Option prices should be positive
    invalid_prices = ((df['near_call_mid_entry'] <= 0) | (df['near_put_mid_entry'] <= 0)).sum()
    print(f"Invalid option prices (â‰¤0): {invalid_prices} ({invalid_prices/len(df)*100:.1f}%)")
    
    # IV crush should be negative (IV drops post-earnings)
    positive_crush = (df['iv_crush_pct'] > 0).sum()
    print(f"Unexpected IV increases: {positive_crush} ({positive_crush/len(df)*100:.1f}%)")
    
    return df


def analyze_volatility_patterns(df):
    """Analyze volatility patterns"""
    print(f"\n{'='*70}")
    print("VOLATILITY ANALYSIS")
    print(f"{'='*70}")
    
    # Realized vol distribution
    print(f"\nRealized Volatility:")
    print(f"  Mean: {df['realized_vol'].mean():.2%}")
    print(f"  Median: {df['realized_vol'].median():.2%}")
    print(f"  Std: {df['realized_vol'].std():.2%}")
    print(f"  Range: {df['realized_vol'].min():.2%} to {df['realized_vol'].max():.2%}")
    
    # IV/RV ratio
    print(f"\nIV/RV Ratio:")
    print(f"  Mean: {df['iv_rv_ratio'].mean():.3f}")
    print(f"  Median: {df['iv_rv_ratio'].median():.3f}")
    print(f"  >1.5 (high overpricing): {(df['iv_rv_ratio'] > 1.5).sum()} ({(df['iv_rv_ratio'] > 1.5).mean()*100:.1f}%)")
    
    # IV crush
    print(f"\nIV Crush:")
    print(f"  Mean: {df['iv_crush_pct'].mean():.1f}%")
    print(f"  Median: {df['iv_crush_pct'].median():.1f}%")
    print(f"  Range: {df['iv_crush_pct'].min():.1f}% to {df['iv_crush_pct'].max():.1f}%")
    
    # Term structure
    print(f"\nVolatility Term Structure:")
    print(f"  Slope - Mean: {df['slope'].mean():.4f}")
    print(f"  Ratio - Mean: {df['ratio'].mean():.3f}")
    print(f"  Contango (ratio < 1): {(df['ratio'] < 1).sum()} ({(df['ratio'] < 1).mean()*100:.1f}%)")
    
    # Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. RV distribution
    axes[0,0].hist(df['realized_vol'], bins=50, alpha=0.7, color='steelblue')
    axes[0,0].axvline(df['realized_vol'].mean(), color='red', linestyle='--', label=f'Mean: {df["realized_vol"].mean():.2%}')
    axes[0,0].set_xlabel('Realized Volatility')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Realized Volatility Distribution')
    axes[0,0].legend()
    
    # 2. IV vs RV
    axes[0,1].scatter(df['realized_vol'], df['near_iv_entry'], alpha=0.5, s=20)
    axes[0,1].plot([0, df['realized_vol'].max()], [0, df['realized_vol'].max()], 'r--', label='IV = RV')
    axes[0,1].set_xlabel('Realized Vol')
    axes[0,1].set_ylabel('Implied Vol (Near-term)')
    axes[0,1].set_title('IV vs RV')
    axes[0,1].legend()
    
    # 3. IV/RV ratio distribution
    axes[0,2].hist(df['iv_rv_ratio'], bins=50, alpha=0.7, color='green')
    axes[0,2].axvline(1.0, color='red', linestyle='--', label='Fair Value (IV=RV)')
    axes[0,2].axvline(df['iv_rv_ratio'].mean(), color='orange', linestyle='--', label=f'Mean: {df["iv_rv_ratio"].mean():.2f}')
    axes[0,2].set_xlabel('IV/RV Ratio')
    axes[0,2].set_ylabel('Frequency')
    axes[0,2].set_title('IV/RV Ratio Distribution')
    axes[0,2].legend()
    
    # 4. IV crush distribution
    axes[1,0].hist(df['iv_crush_pct'], bins=50, alpha=0.7, color='purple')
    axes[1,0].axvline(df['iv_crush_pct'].mean(), color='red', linestyle='--', label=f'Mean: {df["iv_crush_pct"].mean():.1f}%')
    axes[1,0].set_xlabel('IV Crush %')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Post-Earnings IV Crush')
    axes[1,0].legend()
    
    # 5. Term structure slope
    axes[1,1].hist(df['slope'], bins=50, alpha=0.7, color='orange')
    axes[1,1].axvline(0, color='red', linestyle='--', label='Flat Term Structure')
    axes[1,1].axvline(df['slope'].mean(), color='blue', linestyle='--', label=f'Mean: {df["slope"].mean():.4f}')
    axes[1,1].set_xlabel('Term Structure Slope')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Volatility Term Structure Slope')
    axes[1,1].legend()
    
    # 6. Term structure ratio
    axes[1,2].hist(df['ratio'], bins=50, alpha=0.7, color='coral')
    axes[1,2].axvline(1.0, color='red', linestyle='--', label='Flat (Far IV = Near IV)')
    axes[1,2].axvline(df['ratio'].mean(), color='blue', linestyle='--', label=f'Mean: {df["ratio"].mean():.3f}')
    axes[1,2].set_xlabel('Far IV / Near IV')
    axes[1,2].set_ylabel('Frequency')
    axes[1,2].set_title('Volatility Term Structure Ratio')
    axes[1,2].legend()
    
    plt.tight_layout()
    plt.savefig('volatility_analysis.png', dpi=300, bbox_inches='tight')
    print("\nðŸ’¾ Saved: volatility_analysis.png")
    plt.show()


def analyze_strategy_performance(df):
    """Calculate and analyze strategy P&L"""
    print(f"\n{'='*70}")
    print("STRATEGY PERFORMANCE")
    print(f"{'='*70}")
    
    # Short Straddle
    df['straddle_entry'] = df['near_call_ask_entry'] + df['near_put_ask_entry']
    df['straddle_exit'] = df['near_call_bid_exit'] + df['near_put_bid_exit']
    df['straddle_pnl'] = df['straddle_entry'] - df['straddle_exit']
    df['straddle_return'] = df['straddle_pnl'] / df['straddle_entry']
    
    # Calendar Spread
    df['calendar_entry'] = df['far_call_bid_entry'] - df['near_call_ask_entry']
    df['calendar_exit'] = df['far_call_ask_exit'] - df['near_call_bid_exit']
    df['calendar_pnl'] = df['calendar_exit'] - df['calendar_entry']
    df['calendar_return'] = df['calendar_pnl'] / abs(df['calendar_entry'])
    
    # Short Straddle Statistics
    print(f"\nðŸ“Š SHORT STRADDLE:")
    print(f"  Win Rate: {(df['straddle_pnl'] > 0).mean()*100:.1f}%")
    print(f"  Avg P&L: ${df['straddle_pnl'].mean():.2f}")
    print(f"  Avg Return: {df['straddle_return'].mean()*100:.2f}%")
    print(f"  Std Dev: ${df['straddle_pnl'].std():.2f}")
    print(f"  Sharpe: {df['straddle_pnl'].mean() / df['straddle_pnl'].std():.3f}")
    print(f"  Best: ${df['straddle_pnl'].max():.2f}")
    print(f"  Worst: ${df['straddle_pnl'].min():.2f}")
    print(f"  Max Drawdown: ${df['straddle_pnl'].cumsum().expanding().max().sub(df['straddle_pnl'].cumsum()).max():.2f}")
    
    # Calendar Spread Statistics
    print(f"\nðŸ“Š LONG CALENDAR SPREAD:")
    print(f"  Win Rate: {(df['calendar_pnl'] > 0).mean()*100:.1f}%")
    print(f"  Avg P&L: ${df['calendar_pnl'].mean():.2f}")
    print(f"  Avg Return: {df['calendar_return'].mean()*100:.2f}%")
    print(f"  Std Dev: ${df['calendar_pnl'].std():.2f}")
    print(f"  Sharpe: {df['calendar_pnl'].mean() / df['calendar_pnl'].std():.3f}")
    print(f"  Best: ${df['calendar_pnl'].max():.2f}")
    print(f"  Worst: ${df['calendar_pnl'].min():.2f}")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Straddle P&L distribution
    axes[0,0].hist(df['straddle_pnl'], bins=50, alpha=0.7, color='steelblue')
    axes[0,0].axvline(0, color='red', linestyle='--', label='Break-even')
    axes[0,0].axvline(df['straddle_pnl'].mean(), color='orange', linestyle='--', label=f'Mean: ${df["straddle_pnl"].mean():.2f}')
    axes[0,0].set_xlabel('P&L ($)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Short Straddle P&L Distribution')
    axes[0,0].legend()
    
    # 2. Calendar P&L distribution
    axes[0,1].hist(df['calendar_pnl'], bins=50, alpha=0.7, color='green')
    axes[0,1].axvline(0, color='red', linestyle='--', label='Break-even')
    axes[0,1].axvline(df['calendar_pnl'].mean(), color='orange', linestyle='--', label=f'Mean: ${df["calendar_pnl"].mean():.2f}')
    axes[0,1].set_xlabel('P&L ($)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Calendar Spread P&L Distribution')
    axes[0,1].legend()
    
    # 3. Cumulative P&L comparison
    axes[1,0].plot(df['straddle_pnl'].cumsum(), label='Short Straddle', linewidth=2)
    axes[1,0].plot(df['calendar_pnl'].cumsum(), label='Calendar Spread', linewidth=2)
    axes[1,0].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[1,0].set_xlabel('Event Number')
    axes[1,0].set_ylabel('Cumulative P&L ($)')
    axes[1,0].set_title('Cumulative P&L Over Time')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Return distribution comparison
    axes[1,1].hist(df['straddle_return']*100, bins=50, alpha=0.5, label='Straddle', color='steelblue')
    axes[1,1].hist(df['calendar_return']*100, bins=50, alpha=0.5, label='Calendar', color='green')
    axes[1,1].axvline(0, color='red', linestyle='--')
    axes[1,1].set_xlabel('Return (%)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Return Distribution Comparison')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('strategy_performance.png', dpi=300, bbox_inches='tight')
    print("\nðŸ’¾ Saved: strategy_performance.png")
    plt.show()
    
    return df


def conditional_analysis(df):
    """Analyze performance under different conditions"""
    print(f"\n{'='*70}")
    print("CONDITIONAL ANALYSIS")
    print(f"{'='*70}")
    
    # Split by IV/RV ratio
    median_iv_rv = df['iv_rv_ratio'].median()
    high_iv_rv = df[df['iv_rv_ratio'] > median_iv_rv]
    low_iv_rv = df[df['iv_rv_ratio'] <= median_iv_rv]
    
    print(f"\nðŸ” Performance by IV/RV Ratio:")
    print(f"\nHigh IV/RV (>{median_iv_rv:.2f}):")
    print(f"  Straddle Win Rate: {(high_iv_rv['straddle_pnl'] > 0).mean()*100:.1f}%")
    print(f"  Straddle Avg P&L: ${high_iv_rv['straddle_pnl'].mean():.2f}")
    print(f"  Calendar Win Rate: {(high_iv_rv['calendar_pnl'] > 0).mean()*100:.1f}%")
    print(f"  Calendar Avg P&L: ${high_iv_rv['calendar_pnl'].mean():.2f}")
    
    print(f"\nLow IV/RV (â‰¤{median_iv_rv:.2f}):")
    print(f"  Straddle Win Rate: {(low_iv_rv['straddle_pnl'] > 0).mean()*100:.1f}%")
    print(f"  Straddle Avg P&L: ${low_iv_rv['straddle_pnl'].mean():.2f}")
    print(f"  Calendar Win Rate: {(low_iv_rv['calendar_pnl'] > 0).mean()*100:.1f}%")
    print(f"  Calendar Avg P&L: ${low_iv_rv['calendar_pnl'].mean():.2f}")
    
    # Split by term structure slope
    median_slope = df['slope'].median()
    steep_slope = df[df['slope'] > median_slope]
    flat_slope = df[df['slope'] <= median_slope]
    
    print(f"\nðŸ” Performance by Term Structure Slope:")
    print(f"\nSteep Slope (>{median_slope:.4f}):")
    print(f"  Straddle Win Rate: {(steep_slope['straddle_pnl'] > 0).mean()*100:.1f}%")
    print(f"  Straddle Avg P&L: ${steep_slope['straddle_pnl'].mean():.2f}")
    print(f"  Calendar Win Rate: {(steep_slope['calendar_pnl'] > 0).mean()*100:.1f}%")
    print(f"  Calendar Avg P&L: ${steep_slope['calendar_pnl'].mean():.2f}")
    
    print(f"\nFlat Slope (â‰¤{median_slope:.4f}):")
    print(f"  Straddle Win Rate: {(flat_slope['straddle_pnl'] > 0).mean()*100:.1f}%")
    print(f"  Straddle Avg P&L: ${flat_slope['straddle_pnl'].mean():.2f}")
    print(f"  Calendar Win Rate: {(flat_slope['calendar_pnl'] > 0).mean()*100:.1f}%")
    print(f"  Calendar Avg P&L: ${flat_slope['calendar_pnl'].mean():.2f}")
    
    # Timing analysis (AMC vs BMO)
    amc = df[df['timing'] == 'AMC']
    bmo = df[df['timing'] == 'BMO']
    
    if len(bmo) > 0:
        print(f"\nðŸ” Performance by Timing:")
        print(f"\nAMC (After Market Close):")
        print(f"  Events: {len(amc)}")
        print(f"  Straddle Win Rate: {(amc['straddle_pnl'] > 0).mean()*100:.1f}%")
        print(f"  Straddle Avg P&L: ${amc['straddle_pnl'].mean():.2f}")
        
        print(f"\nBMO (Before Market Open):")
        print(f"  Events: {len(bmo)}")
        print(f"  Straddle Win Rate: {(bmo['straddle_pnl'] > 0).mean()*100:.1f}%")
        print(f"  Straddle Avg P&L: ${bmo['straddle_pnl'].mean():.2f}")


if __name__ == "__main__":
    print("="*70)
    print("EARNINGS OPTIONS DATASET - ANALYSIS")
    print("="*70)
    
    # Load data
    df = load_and_validate('earnings_options_dataset.csv')
    
    # Volatility analysis
    analyze_volatility_patterns(df)
    
    # Strategy performance
    df = analyze_strategy_performance(df)
    
    # Conditional analysis
    conditional_analysis(df)
    
    print(f"\n{'='*70}")
    print("âœ… Analysis complete!")
    print(f"{'='*70}")
    print("\nGenerated files:")
    print("  - volatility_analysis.png")
    print("  - strategy_performance.png")
