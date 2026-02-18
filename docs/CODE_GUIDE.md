# Code Guide: Earnings Volatility Backtest

Complete walkthrough of the three scripts that power the backtest, charts, and PDF report.

---

## Pipeline Overview

```
Step 0: data_collection/collect_data.py  →  Fetches earnings + prices from APIs
Step 1: backtest_v2.py                   →  Runs 420 trades, computes P&L, equity curve
Step 2: generate_charts_v2.py            →  Creates 3 PNG charts from Step 1 outputs
Step 3: generate_enhanced_pdf.py         →  Builds 25-page PDF report from Steps 1+2
```

You run them in order. Each script reads the outputs of the previous one.

---

## Script 0: collect_data.py (Data Collection)

### What it does

Fetches two types of raw data from external APIs:

1. Earnings dates from Alpha Vantage (free API)
2. Daily OHLCV prices from Yahoo Finance (free, no API key needed)

### How Alpha Vantage works

Alpha Vantage is a financial data provider with a REST API. You send an HTTP GET request with parameters, and it returns JSON.

```python
url = "https://www.alphavantage.co/query"
params = {
    'function': 'EARNINGS',     # Which endpoint to call
    'symbol': 'AAPL',           # Which stock
    'apikey': 'YOUR_KEY'        # Authentication
}
response = requests.get(url, params=params)
data = response.json()
```

The response looks like:
```json
{
  "quarterlyEarnings": [
    {
      "reportedDate": "2024-10-31",
      "fiscalDateEnding": "2024-09-30",
      "reportedEPS": "1.64",
      "estimatedEPS": "1.60",
      "surprise": "0.04",
      "surprisePercentage": "2.5"
    }
  ]
}
```

Key fields:
- reportedDate = the day earnings were announced (this is our "earnings_date")
- reportedEPS = actual earnings per share
- estimatedEPS = what analysts expected
- surprise = reportedEPS - estimatedEPS
- surprisePercentage = how far off the estimate was

### How yfinance works

yfinance is a Python library that scrapes Yahoo Finance data. No API key needed.

```python
stock = yf.Ticker('AAPL')
df = stock.history(start='2019-01-01', end='2026-01-01', interval='1d')
```

Returns a DataFrame with columns: Open, High, Low, Close, Volume, Dividends, Stock Splits.
We keep only Open/High/Low/Close/Volume (OHLCV).

### Rate limiting

Alpha Vantage free tier: 5 calls per minute. The script waits 12 seconds between calls (60/5 = 12).

If you hit the limit, the API returns a "Note" field instead of data. The script detects this and waits 60 seconds before retrying.

### Checkpoint system

After each stock, the script saves a checkpoint file listing all completed tickers. If the script crashes (network error, power loss, etc.), you re-run it and it skips already-completed tickers.

```python
# Save progress
completed.add('AAPL')
save_checkpoint(completed)        # Writes to collection_checkpoint.json

# On restart
completed = load_checkpoint()     # Reads back the set {'AAPL', 'MSFT', ...}
tickers = [t for t in all_tickers if t not in completed]  # Skip done ones
```

---

## File Map

| File | What it does | Outputs |
|------|-------------|---------|
| `data_collection/collect_data.py` | Fetches earnings + prices from APIs | `earnings_data_FINAL.csv`, `price_data_FINAL.csv` |
| `backtest_v2.py` | Backtest engine with capital management | `backtest_v2_flat.csv`, `backtest_v2_skew.csv`, `backtest_v2_metrics.json`, `equity_flat.npy`, `equity_skew.npy` |
| `generate_charts_v2.py` | Professional matplotlib charts | `backtest_v2_dashboard.png`, `backtest_v2_annual.png`, `backtest_v2_skew_analysis.png` |
| `generate_enhanced_pdf.py` | 25-page PDF with reportlab | `Rapport_Enhanced.pdf` |
| `run_all.py` | Master script, runs Steps 1-3 in order | All backtest outputs |

---

## Script 1: backtest_v2.py (Backtest Engine)

### What it computes

For each of the 420 earnings events:
1. Finds the entry price (close of T-1, the day before earnings)
2. Finds the exit price (close of T, earnings day)
3. Estimates implied volatility from realized volatility
4. Prices the call and put using Black-Scholes
5. Simulates selling the straddle at bid and buying it back at ask
6. Computes P&L after commissions and slippage
7. Updates the equity curve

### Key functions explained

#### `calculate_yang_zhang_volatility(prices_df, window=30)`

**Purpose:** Estimate how much a stock has been moving over the last 30 days.

**Why Yang-Zhang instead of simple standard deviation?**
Standard deviation only uses closing prices. Yang-Zhang uses Open, High, Low, Close. This captures overnight gaps (close-to-open) and intraday range (high-low), making it more accurate.

**The math (simplified):**
```
overnight_variance = (log(Open / previous Close))^2
intraday_variance  = (log(Close / Open))^2
range_component    = log(High/Open) * (log(High/Open) - log(Close/Open))
                   + log(Low/Open)  * (log(Low/Open)  - log(Close/Open))
```

These three components are averaged over 30 days with a weighting factor `k`, then multiplied by 252 (trading days) and square-rooted to get annualized volatility.

**Example:** If the function returns 0.30, it means the stock has been moving at a pace equivalent to 30% per year.

---

#### `black_scholes_price(S, K, T, r, sigma, option_type='call')`

**Purpose:** Price a European option given 5 inputs.

**Inputs:**
- `S` = spot price (current stock price, e.g., $150.00)
- `K` = strike price (the price at which the option pays off, e.g., $150)
- `T` = time to expiration in years (7 days = 7/365 = 0.0192 years)
- `r` = risk-free rate (0.04 = 4%)
- `sigma` = implied volatility (0.35 = 35% annualized)

**The formula:**

```python
d1 = (log(S/K) + (r + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
d2 = d1 - sigma * sqrt(T)

call_price = S * N(d1) - K * e^(-rT) * N(d2)
put_price  = K * e^(-rT) * N(-d2) - S * N(-d1)
```

Where `N()` is the cumulative normal distribution function (from `scipy.stats.norm.cdf`).

**Intuition for d1 and d2:**
- d1 tells you how far "in the money" the option is, adjusted for time and volatility
- d2 is the same thing but for the strike payment (discounted)
- N(d1) is roughly the probability-weighted exposure to the stock
- N(d2) is roughly the probability the option expires in the money

**Example:**
```python
black_scholes_price(150, 150, 7/365, 0.04, 0.35, 'call')
# Returns ~$3.45 (the call is worth $3.45 per share)
```

---

#### `calculate_greeks(S, K, T, r, sigma, option_type='call')`

**Purpose:** Measure how sensitive the option price is to changes in inputs.

**Delta** = How much the option price changes when the stock moves $1
- Call delta is between 0 and 1 (ATM call is about 0.50)
- Put delta is between -1 and 0 (ATM put is about -0.50)
- For a straddle: delta_call + delta_put is close to 0 (delta-neutral)

**Gamma** = How fast delta changes when the stock moves
- High gamma = delta shifts fast = dangerous for short positions
- Short straddle has large negative gamma: if stock jumps, losses accelerate

**Vega** = How much the option price changes when IV moves 1%
- Long options have positive vega (benefit from IV increase)
- Short straddle has negative vega (benefits from IV crush)

**Theta** = How much the option loses per day from time passing
- Short straddle has positive theta (earns money each day)
- This is the source of income for the strategy

**Code:**
```python
gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
```
Note: `norm.pdf(d1)` is the probability density function (the bell curve height at d1). Gamma is largest when the option is ATM and short-dated, because the bell curve is tallest at the center.

---

#### `calculate_position_size(capital, spot_price, iv_entry, max_risk_pct=0.02)`

**Purpose:** Determine how many contracts to trade without risking too much capital.

**Logic:**
```
expected_move = stock_price * IV * sqrt(1/252)
                      |           |        |
              e.g. $150    e.g. 35%   1 day in annual terms

dollar_risk_per_contract = 2 * expected_move * 100 shares
                                    |
                        2x because straddle is exposed to both directions

max_contracts = floor(2% of capital / dollar_risk_per_contract)
```

**Example with $100,000 capital, AAPL at $150, IV = 35%:**
```
expected_move = 150 * 0.35 * sqrt(1/252) = $3.31
risk_per_contract = 2 * 3.31 * 100 = $662
max_risk = 100,000 * 0.02 = $2,000
n_contracts = floor(2000 / 662) = 3
```

---

#### `run_backtest_v2(earnings_df, prices_df, use_skew=False)`

**Purpose:** The main loop that processes all 420 earnings events.

**Step by step for each trade:**

1. **Find entry/exit dates**
   ```python
   # Entry = last trading day before earnings
   entry_rows = ticker_prices[ticker_prices['date'] <= target_entry]
   entry_row = entry_rows.iloc[-1]  # Last row = closest date
   ```
   This handles weekends: if earnings is Monday, entry is Friday.

2. **Estimate IV from realized vol**
   ```python
   iv_atm_entry = realized_vol * 1.35   # IV is 35% above RV before earnings
   iv_atm_exit  = iv_atm_entry * 0.65   # IV drops 35% after earnings
   ```
   This is the "synthetic IV" model. Real IV would come from market option prices.

3. **Apply skew (optional)**
   ```python
   iv_put_entry  = iv_atm_entry + 0.05   # Puts cost more (downside protection demand)
   iv_call_entry = iv_atm_entry - 0.03   # Calls cost less
   ```

4. **Price options at entry (sell at bid)**
   ```python
   call_mid = black_scholes_price(spot, strike, 7/365, 0.04, iv_call, 'call')
   call_bid = call_mid * (1 - 0.03/2) * (1 - 0.01)  # Bid = mid - half spread - slippage
   ```
   You sell at bid (lower price), which is realistic: you never get mid-price execution.

5. **Price options at exit (buy back at ask)**
   ```python
   call_ask_exit = call_mid_exit * (1 + 0.03/2) * (1 + 0.01)  # Ask = mid + half spread + slippage
   ```
   You buy back at ask (higher price). The spread costs you both ways.

6. **P&L = what you collected - what you paid back**
   ```python
   pnl_per_share = premium_received - buyback_cost - commissions
   total_pnl = pnl_per_share * 100 * n_contracts
   ```

---

#### `compute_portfolio_metrics(results_df, equity_curve)`

**CAGR** (Compound Annual Growth Rate):
```python
cagr = (final_capital / initial_capital) ^ (1 / years) - 1
```
Example: $100K to $290K in 6.88 years = 16.7% per year.

**Sharpe Ratio** (risk-adjusted return):
```python
sharpe = mean_return * sqrt(trades_per_year) / std_return * sqrt(trades_per_year)
       = mean_return / std_return * sqrt(trades_per_year)
```
The sqrt(trades_per_year) annualizes the ratio. With 61 trades/year and low variance, the annualized Sharpe inflates to 5.12. The per-trade Sharpe is 0.41.

**Max Drawdown** (worst peak-to-trough decline):
```python
running_max = np.maximum.accumulate(equity)  # Running high-water mark
drawdown = running_max - equity              # How far below the peak
max_dd = max(drawdown / running_max * 100)   # Worst decline as %
```

**Sortino Ratio** (like Sharpe but only penalizes downside):
```python
downside_vol = std(returns[returns < 0]) * sqrt(trades_per_year)
sortino = annualized_return / downside_vol
```

**Profit Factor** (total wins / total losses):
```python
profit_factor = sum(winning_pnl) / abs(sum(losing_pnl))
```
A profit factor of 4.81 means you make $4.81 for every $1 you lose.

---

## Script 2: generate_charts_v2.py (Charts)

### `plot_main_dashboard()` - 6-panel figure

Panel 1: Equity curve in % return (not dollars, for comparability)
Panel 2: Underwater chart showing drawdowns from peak
Panel 3: Histogram of P&L per trade (green wins, red losses)
Panel 4: Horizontal bar chart of win rate per stock
Panel 5: Scatter plot of trade return vs stock move magnitude
Panel 6: Summary metrics table

**Key matplotlib patterns used:**

```python
# Fill between for equity curve coloring
ax.fill_between(x, 0, y, where=y >= 0, alpha=0.15, color='blue')

# Annotation with arrow pointing to max drawdown
ax.annotate('Max DD: -3.14%', xy=(idx, value),
            arrowprops=dict(arrowstyle='->', color='red'))

# Table inside a subplot
table = ax.table(cellText=data, colLabels=headers, loc='center')
```

### `plot_skew_analysis()` - 4-panel comparison

Compares the flat IV model vs the skew model side by side.
Shows where skew helps (stock up = higher premium collected from expensive puts) and where it hurts (stock down = put skew expansion).

### `plot_annual_performance()` - Year-by-year breakdown

Left panel: bar chart of annual P&L with win rates.
Right panel: heatmap of monthly returns (useful for spotting earnings season clusters in Jan/Apr/Jul/Oct).

---

## Script 3: generate_enhanced_pdf.py (PDF Report)

Uses `reportlab` to build a 25-page PDF with:
- Title page with key metrics
- Table of contents
- 14 sections covering strategy, data, methodology, and results
- Embedded charts (the 3 PNGs from Script 2)
- Styled tables, formula blocks, note boxes, and warning boxes

**Key reportlab patterns:**

```python
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, Image, PageBreak
from reportlab.lib.styles import ParagraphStyle

# Create a document
doc = SimpleDocTemplate("output.pdf")

# Build content as a list of "flowables"
story = []
story.append(Paragraph("Title", style))
story.append(Table(data, colWidths=[...]))
story.append(Image("chart.png", width=450, height=250))
story.append(PageBreak())

# Render
doc.build(story)
```

---

## Data Flow Diagram

```
Alpha Vantage API ──┐
                    ├── collect_data.py ──┬── earnings_data_FINAL.csv ──┐
Yahoo Finance ──────┘                    └── price_data_FINAL.csv ─────┤
                                                                       │
                                                                       ▼
                                                                backtest_v2.py
                                                                       │
                                                          ┌────────────┼────────────┐
                                                          ▼            ▼            ▼
                                                    flat.csv      skew.csv    metrics.json
                                                    equity_flat   equity_skew
                                                          │            │            │
                                                          └────────────┼────────────┘
                                                                       │
                                                                       ▼
                                                          generate_charts_v2.py
                                                                       │
                                                              ┌────────┼────────┐
                                                              ▼        ▼        ▼
                                                         dashboard  annual   skew_analysis
                                                           .png      .png       .png
                                                              │        │        │
                                                              └────────┼────────┘
                                                                       │
                                                    all intermediate files
                                                                       │
                                                                       ▼
                                                       generate_enhanced_pdf.py
                                                                       │
                                                                       ▼
                                                           Rapport_Enhanced.pdf
```

---

## Key Python Concepts Used

### numpy operations
```python
np.log(x)                    # Natural logarithm
np.sqrt(x)                   # Square root
np.maximum.accumulate(arr)   # Running maximum (for drawdown)
arr.rolling(30).mean()       # 30-day rolling average (pandas)
```

### scipy.stats.norm
```python
norm.cdf(x)   # Cumulative normal distribution (area under bell curve up to x)
norm.pdf(x)   # Probability density function (height of bell curve at x)
```

### pandas patterns
```python
# Filter rows
df[df['ticker'] == 'AAPL']

# Find closest date (last row before target)
df[df['date'] <= target].iloc[-1]

# Group and aggregate
df.groupby('ticker')['pnl'].mean()

# Rolling calculation per group
for ticker in df['ticker'].unique():
    subset = df[df['ticker'] == ticker]
    subset['vol'] = calculate_vol(subset)
```

---

## How to Modify the Backtest

### Change the IV model
Edit lines 44-45 in `backtest_v2.py`:
```python
IV_INFLATION = 1.35   # Try 1.20 (conservative) or 1.50 (aggressive)
IV_CRUSH = 0.65       # Try 0.50 (more crush) or 0.75 (less crush)
```

### Change position sizing
Edit line 34:
```python
MAX_RISK_PER_TRADE = 0.02   # Try 0.01 (conservative) or 0.05 (aggressive)
```

### Add a new stock
Add tickers to `earnings_data_FINAL.csv` and `price_data_FINAL.csv` using the `collect_data_simple.py` script, then re-run the pipeline.

### Test a different DTE
Edit line 53:
```python
NEAR_DTE = 7   # Try 14 (2 weeks) or 3 (closer to expiry)
```
