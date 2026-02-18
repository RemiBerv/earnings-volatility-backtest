"""
=============================================================================
DATA COLLECTION - Earnings Dates + Price History
=============================================================================

This script collects the raw data needed for the backtest:
    1. Earnings dates + EPS data from Alpha Vantage API
    2. Daily OHLCV price data from Yahoo Finance (yfinance)

Prerequisites:
    pip install pandas yfinance requests

You need a free Alpha Vantage API key:
    https://www.alphavantage.co/support/#api-key

How it works:
    - For each stock, it calls Alpha Vantage to get quarterly earnings dates
    - Then calls Yahoo Finance to get daily price history
    - Saves everything to two CSV files
    - Has checkpointing: if it crashes, you restart and it picks up where it left off

Output:
    - earnings_data_FINAL.csv  (earnings dates + EPS surprise data)
    - price_data_FINAL.csv     (daily Open, High, Low, Close, Volume)

Time estimate:
    - 15 stocks: ~3 minutes
    - 100 stocks: ~20 minutes
    - The bottleneck is Alpha Vantage's rate limit (5 calls per minute)

=============================================================================
"""

import pandas as pd
import yfinance as yf
import requests
import time
from datetime import datetime
import json
import os

# =============================================================================
# CONFIGURATION - MODIFY THESE
# =============================================================================

# Your Alpha Vantage API key (free at https://www.alphavantage.co/support/#api-key)
API_KEY = "YOUR_API_KEY_HERE"

# Period to collect earnings for
START_DATE = '2019-01-01'
END_DATE = '2026-01-01'

# For prices, take a wider window (needed to compute 30-day realized volatility
# before the first earnings date)
PRICE_START = '2018-01-01'
PRICE_END = '2026-03-01'

# =============================================================================
# STOCK UNIVERSE
# =============================================================================
# These are the 15 tech stocks used in the backtest.
# You can expand this list to any S&P 500 stocks.
# The full 100-stock list is in sp500_top100_list.txt.

TICKERS_BACKTEST = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'META', 'ADBE', 'CRM', 'CSCO', 'INTC',
    'AMD', 'QCOM', 'TXN', 'NFLX', 'ORCL',
]

# Full S&P 500 top 100 (uncomment to collect a larger universe)
# TICKERS_FULL = [
#     # Technology (25)
#     'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AVGO', 'ORCL', 'ADBE',
#     'CRM', 'CSCO', 'INTC', 'AMD', 'QCOM', 'TXN', 'NOW', 'INTU',
#     'IBM', 'AMAT', 'MU', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'PANW', 'GOOG',
#     # Financials (15)
#     'BRK.B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'AXP', 'C',
#     'BLK', 'SCHW', 'CB', 'SPGI', 'PGR',
#     # Healthcare (15)
#     'UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'DHR',
#     'CVS', 'AMGN', 'BMY', 'GILD', 'CI', 'ISRG',
#     # Consumer Discretionary (12)
#     'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TGT',
#     'BKNG', 'CMG', 'MAR', 'F',
#     # Consumer Staples (8)
#     'WMT', 'PG', 'KO', 'PEP', 'COST', 'PM', 'MDLZ', 'CL',
#     # Energy (8)
#     'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO',
#     # Industrials + Other (10+)
#     'BA', 'CAT', 'GE', 'RTX', 'UNP', 'HON', 'UPS', 'LMT', 'DE', 'MMM',
#     'NFLX', 'DIS', 'CMCSA', 'T', 'LIN', 'APD', 'SHW',
# ]

# Choose which list to use
TICKERS = TICKERS_BACKTEST

# Checkpoint file (allows resume after interruption)
CHECKPOINT_FILE = 'collection_checkpoint.json'


# =============================================================================
# FUNCTION: GET EARNINGS DATES FROM ALPHA VANTAGE
# =============================================================================

def get_earnings_dates(ticker, api_key):
    """
    Call the Alpha Vantage EARNINGS endpoint to get historical quarterly earnings.

    The API returns a JSON object with two arrays:
        - "annualEarnings": yearly EPS (we don't use this)
        - "quarterlyEarnings": what we want, each entry has:
            - reportedDate: the actual date earnings were announced
            - fiscalDateEnding: end of the fiscal quarter
            - reportedEPS: actual EPS number reported
            - estimatedEPS: Wall Street consensus estimate
            - surprise: reportedEPS - estimatedEPS
            - surprisePercentage: surprise as % of estimate

    Rate limit: Alpha Vantage free tier allows 5 calls per minute.
    The script waits 12 seconds between calls to stay under this limit.

    Parameters:
        ticker: stock symbol (e.g., 'AAPL')
        api_key: your Alpha Vantage API key

    Returns:
        list of dicts, each containing one quarterly earnings event
    """

    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'EARNINGS',      # Historical earnings (not EARNINGS_CALENDAR)
        'symbol': ticker,
        'apikey': api_key
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        # Handle rate limit (Alpha Vantage returns a "Note" field when you're too fast)
        if 'Note' in data:
            print(f"    Rate limit hit, waiting 60s...")
            time.sleep(60)
            return get_earnings_dates(ticker, api_key)  # Recursive retry

        # Handle API errors
        if 'Error Message' in data:
            print(f"    API Error: {data['Error Message']}")
            return []

        if 'quarterlyEarnings' not in data:
            print(f"    No earnings data available")
            return []

        # Parse each quarterly earnings entry
        earnings = []
        for q in data['quarterlyEarnings']:
            try:
                date = pd.to_datetime(q['reportedDate'])

                # Only keep earnings within our target period
                if START_DATE <= str(date) <= END_DATE:
                    earnings.append({
                        'date': date,
                        'fiscalDateEnding': q.get('fiscalDateEnding', ''),
                        'reportedEPS': q.get('reportedEPS', ''),
                        'estimatedEPS': q.get('estimatedEPS', ''),
                        'surprise': q.get('surprise', ''),
                        'surprisePercentage': q.get('surprisePercentage', '')
                    })
            except Exception:
                # Skip entries with bad data (missing date, etc.)
                continue

        # Wait 12 seconds to respect rate limit (5 calls / 60 sec = 12 sec each)
        time.sleep(12)

        return earnings

    except Exception as e:
        print(f"    Error: {e}")
        time.sleep(12)
        return []


# =============================================================================
# FUNCTION: GET PRICE DATA FROM YAHOO FINANCE
# =============================================================================

def get_price_data(ticker):
    """
    Download daily OHLCV data from Yahoo Finance using the yfinance library.

    OHLCV stands for:
        Open  = first price of the day
        High  = highest price during the day
        Low   = lowest price during the day
        Close = last price of the day (most commonly used)
        Volume = number of shares traded

    We need Open/High/Low for the Yang-Zhang volatility estimator.
    We need Close for option pricing (spot price at entry/exit).
    Volume is informational.

    Parameters:
        ticker: stock symbol (e.g., 'AAPL')

    Returns:
        pandas DataFrame with columns: date, ticker, Open, High, Low, Close, Volume
        Returns None if download fails
    """

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=PRICE_START, end=PRICE_END, interval='1d')

        if df.empty:
            print(f"    No price data")
            return None

        # Clean up the DataFrame
        df = df.reset_index()
        df = df.rename(columns={'Date': 'date'})
        df['ticker'] = ticker

        # Keep only the columns we need (yfinance also returns Dividends, Stock Splits)
        cols = ['date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = df[cols]

        return df

    except Exception as e:
        print(f"    Price error: {e}")
        return None


# =============================================================================
# CHECKPOINT FUNCTIONS (for crash recovery)
# =============================================================================

def save_checkpoint(completed_tickers):
    """Save the list of already-processed tickers to disk."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({'completed': list(completed_tickers)}, f)


def load_checkpoint():
    """Load previously completed tickers (empty set if no checkpoint)."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            data = json.load(f)
            return set(data.get('completed', []))
    return set()


# =============================================================================
# MAIN COLLECTION FUNCTION
# =============================================================================

def collect_data():
    """
    Main function that orchestrates the entire data collection.

    Workflow:
        1. Check API key is set
        2. Load checkpoint (if resuming after crash)
        3. For each ticker:
            a. Fetch earnings dates from Alpha Vantage
            b. Fetch price data from Yahoo Finance
            c. Save checkpoint every 10 stocks
        4. Save final CSV files

    The checkpoint system means you can:
        - Press Ctrl+C to stop
        - Re-run the script
        - It picks up from where it stopped (skipping already-done tickers)
    """

    print("=" * 70)
    print("DATA COLLECTION - EARNINGS VOLATILITY STRATEGY")
    print("=" * 70)

    # Check API key
    if API_KEY == "YOUR_API_KEY_HERE":
        print("\nERROR: Set your Alpha Vantage API key on line 44!")
        print("Get a free key at: https://www.alphavantage.co/support/#api-key")
        return

    tickers = TICKERS
    print(f"\nStocks to process: {len(tickers)}")
    print(f"Period: {START_DATE} to {END_DATE}")

    # Resume from checkpoint if available
    completed = load_checkpoint()
    if completed:
        print(f"Resuming: {len(completed)} stocks already done")
        tickers = [t for t in tickers if t not in completed]

    # Storage for collected data
    all_earnings = []
    all_prices = []

    # Load existing data if any (from previous partial runs)
    if os.path.exists('earnings_data.csv'):
        all_earnings = pd.read_csv('earnings_data.csv').to_dict('records')
        print(f"Loaded {len(all_earnings)} existing earnings records")

    total = len(tickers)
    start_time = time.time()

    print(f"\nStarting collection...")
    print(f"Estimated time: ~{total * 12 / 60:.0f} minutes")
    print("=" * 70)

    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{total}] {ticker}...", end=' ')

        try:
            # Step 1: Get earnings dates from Alpha Vantage
            earnings = get_earnings_dates(ticker, API_KEY)
            print(f"Earnings: {len(earnings)}", end=' | ')

            # Step 2: Get price data from Yahoo Finance
            prices = get_price_data(ticker)
            price_count = len(prices) if prices is not None else 0
            print(f"Prices: {price_count} days", end=' | ')

            # Step 3: Store results
            if earnings and prices is not None:
                for earning in earnings:
                    all_earnings.append({
                        'ticker': ticker,
                        'earnings_date': earning['date'],
                        'fiscalDateEnding': earning['fiscalDateEnding'],
                        'reportedEPS': earning['reportedEPS'],
                        'estimatedEPS': earning['estimatedEPS'],
                        'surprise': earning['surprise'],
                        'surprisePercentage': earning['surprisePercentage']
                    })
                all_prices.append(prices)
                print("OK")
            else:
                print("Incomplete")

            # Step 4: Update checkpoint
            completed.add(ticker)
            save_checkpoint(completed)

            # Step 5: Periodic save (every 10 stocks)
            if i % 10 == 0:
                print(f"\nSaving intermediate results...")
                pd.DataFrame(all_earnings).to_csv('earnings_data.csv', index=False)
                if all_prices:
                    pd.concat(all_prices, ignore_index=True).to_csv('price_data.csv', index=False)
                elapsed = time.time() - start_time
                remaining = (elapsed / i) * (total - i)
                print(f"    Elapsed: {elapsed/60:.1f} min | Remaining: {remaining/60:.1f} min")

        except KeyboardInterrupt:
            print("\n\nInterrupted - saving progress...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

    # =============================================================================
    # FINAL SAVE
    # =============================================================================

    print("\n" + "=" * 70)
    print("SAVING FINAL FILES...")
    print("=" * 70)

    # Save earnings
    earnings_df = pd.DataFrame(all_earnings)
    earnings_df.to_csv('earnings_data_FINAL.csv', index=False)
    print(f"Earnings saved: {len(earnings_df)} events")

    # Save prices
    if all_prices:
        prices_df = pd.concat(all_prices, ignore_index=True)
        prices_df = prices_df.sort_values(['ticker', 'date']).reset_index(drop=True)
        prices_df.to_csv('price_data_FINAL.csv', index=False)
        print(f"Prices saved: {len(prices_df)} rows across {prices_df['ticker'].nunique()} stocks")

    # Summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Stocks processed: {len(completed)}/{len(TICKERS)}")
    print(f"Total earnings: {len(earnings_df)}")
    print(f"Period: {earnings_df['earnings_date'].min()} to {earnings_df['earnings_date'].max()}")
    print(f"\nBreakdown by year:")
    earnings_df['year'] = pd.to_datetime(earnings_df['earnings_date']).dt.year
    print(earnings_df['year'].value_counts().sort_index())

    # Clean up checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    print(f"\nDone! Total time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"\nOutput files:")
    print(f"  earnings_data_FINAL.csv  ({len(earnings_df)} rows)")
    print(f"  price_data_FINAL.csv     ({len(prices_df)} rows)")
    print(f"\nNext step: python run_all.py (to run the backtest)")

    return earnings_df, prices_df


# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":

    print("\nBefore starting:")
    print("1. Set your API_KEY on line 44")
    print("2. This will take ~3 minutes for 15 stocks")
    print("3. You can interrupt (Ctrl+C) and resume later")

    input("\nPress ENTER to start...")

    collect_data()
