"""
Configuration file for Earnings Volatility Dataset Builder
Modify these parameters to customize your dataset
"""

# =============================================================================
# API CONFIGURATION
# =============================================================================

# Alpha Vantage API Key
# Get your free key at: https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_API_KEY = "YOUR_API_KEY_HERE"

# =============================================================================
# STOCK SELECTION
# =============================================================================

# Top 50 S&P 500 stocks (high liquidity, regular earnings)
# You can modify this list or change the number
STOCK_UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'V', 'JNJ',
    'WMT', 'JPM', 'MA', 'PG', 'UNH', 'HD', 'CVX', 'MRK', 'ABBV', 'KO',
    'PEP', 'COST', 'AVGO', 'MCD', 'CSCO', 'TMO', 'ACN', 'ABT', 'DIS', 'NKE',
    'ADBE', 'CRM', 'NFLX', 'CMCSA', 'AMD', 'INTC', 'TXN', 'QCOM', 'PM', 'UNP',
    'BA', 'ORCL', 'HON', 'IBM', 'GE', 'CAT', 'LOW', 'UPS', 'SBUX', 'RTX'
]

# Or use a smaller set for testing
TEST_UNIVERSE = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

# =============================================================================
# DATE RANGE
# =============================================================================

# Historical period for backtest
START_DATE = '2022-01-01'  # Start date
END_DATE = '2024-12-31'    # End date

# Note: Earlier dates (2020-2021) available but may have data quality issues
# Recommended: Start from 2022 for most reliable free data

# =============================================================================
# VOLATILITY PARAMETERS
# =============================================================================

# Yang-Zhang realized volatility window (days)
REALIZED_VOL_WINDOW = 30

# Risk-free rate (annualized)
# You can update this based on historical rates
RISK_FREE_RATE = 0.04  # 4% (approximate 2022-2024 average)

# =============================================================================
# OPTIONS PARAMETERS
# =============================================================================

# Bid-ask spread assumptions
NEAR_TERM_SPREAD_PCT = 0.03    # 3% spread for short-dated options
FAR_TERM_SPREAD_PCT = 0.025    # 2.5% spread for longer-dated options

# IV crush assumptions (post-earnings volatility drop)
IV_CRUSH_MIN = 0.50  # Minimum 50% of pre-earnings IV
IV_CRUSH_MAX = 0.70  # Maximum 70% of pre-earnings IV

# Pre-earnings IV premium (multiplier on realized vol)
# How much IV inflates before earnings
IV_PREMIUM_10_DAYS_OUT = 1.05   # 5% premium
IV_PREMIUM_5_DAYS_OUT = 1.15    # 15% premium
IV_PREMIUM_2_DAYS_OUT = 1.30    # 30% premium
IV_PREMIUM_1_DAY_OUT = 1.40     # 40% premium

# Term structure (far-month IV relative to near-month)
TERM_STRUCTURE_DECAY_MIN = 0.75  # Far IV at least 75% of near IV
TERM_STRUCTURE_DECAY_MAX = 0.85  # Far IV at most 85% of near IV

# =============================================================================
# OUTPUT
# =============================================================================

# Output file path
OUTPUT_CSV = 'earnings_options_dataset.csv'

# =============================================================================
# ADVANCED OPTIONS
# =============================================================================

# Strike increment for finding ATM
STRIKE_INCREMENT = 5.0  # $5 strikes for most stocks

# Days to far expiration (from near expiration)
FAR_EXPIRY_OFFSET_DAYS = 30

# Minimum historical data required (days)
MIN_HISTORY_DAYS = 60  # Need this much history to calculate volatility

# =============================================================================
# BACKTEST SETTINGS (for later use)
# =============================================================================

# Transaction costs
COMMISSION_PER_CONTRACT = 0.65  # Per contract fee
SLIPPAGE_PCT = 0.01  # 1% slippage on execution

# Position sizing
CONTRACTS_PER_TRADE = 1  # Start with 1 contract per trade

# =============================================================================
# NOTES
# =============================================================================

"""
IMPORTANT NOTES:

1. Alpha Vantage Free Tier Limits:
   - 5 API calls per minute
   - 500 calls per day
   - If building large datasets, you may need to throttle requests

2. Data Quality:
   - yfinance is free but may have occasional gaps
   - Synthetic options are realistic but not real market data
   - Good for strategy development and learning

3. Realism:
   - This generates Black-Scholes prices with calibrated parameters
   - Real options have skew, microstructure effects, and liquidity issues
   - Results will differ from live trading but methodology is sound

4. Next Steps:
   - Build dataset with this configuration
   - Analyze volatility term structure patterns
   - Backtest short straddle vs calendar spread
   - Optimize entry conditions based on IV/RV ratio, slope, etc.
"""
