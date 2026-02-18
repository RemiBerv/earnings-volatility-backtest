"""
=============================================================================
BACKTEST V2 - SHORT STRADDLE ON EARNINGS (WITH CAPITAL MANAGEMENT & SKEW)
=============================================================================

Upgrades from V1:
    - Capital management: $100,000 starting capital, position sizing
    - Returns in % of capital (not just dollars)
    - Portfolio metrics: CAGR, annualized vol, max drawdown %, Calmar ratio
    - Equity curve and drawdown chart
    - Skew analysis: model asymmetric IV for puts vs calls
    - Professional risk metrics

Input files:
    - earnings_data_FINAL.csv  (420 earnings events, 15 stocks)
    - price_data_FINAL.csv     (30,330 daily OHLCV records)
=============================================================================
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import timedelta, datetime
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

# Capital and position sizing
INITIAL_CAPITAL = 100_000     # $100,000 starting capital
MAX_RISK_PER_TRADE = 0.02    # 2% of current capital at risk per trade
CONTRACTS_PER_TRADE = 1       # 1 contract = 100 shares (used for sizing)

# Market parameters
RISK_FREE_RATE = 0.04
BID_ASK_SPREAD = 0.03
SLIPPAGE = 0.01
COMMISSION = 0.65             # $ per contract per leg

# IV estimation
IV_INFLATION = 1.35
IV_CRUSH = 0.65

# Skew parameters (realistic asymmetry)
# Puts are more expensive than calls before earnings (downside protection demand)
PUT_SKEW_ADDON = 0.05         # Put IV = ATM IV + 5 vol points
CALL_SKEW_ADDON = -0.03       # Call IV = ATM IV - 3 vol points

# Option parameters
NEAR_DTE = 7


# =============================================================================
# YANG-ZHANG REALIZED VOLATILITY
# =============================================================================

def calculate_yang_zhang_volatility(prices_df, window=30):
    """
    Yang-Zhang realized volatility estimator.
    Uses all four OHLC prices for a more accurate estimate than
    simple close-to-close volatility.

    Components:
        - Overnight (close-to-open gaps)
        - Intraday (open-to-close)
        - Rogers-Satchell (uses high and low range)

    Returns annualized volatility (multiplied by sqrt(252)).
    """
    o = np.log(prices_df['Open'] / prices_df['Close'].shift(1))
    c = np.log(prices_df['Close'] / prices_df['Open'])
    h = np.log(prices_df['High'] / prices_df['Open'])
    l = np.log(prices_df['Low'] / prices_df['Open'])

    rs = h * (h - c) + l * (l - c)
    cc = c ** 2
    oo = o ** 2

    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    rs_vol = rs.rolling(window).mean()
    cc_vol = cc.rolling(window).mean()
    oo_vol = oo.rolling(window).mean()

    yz_variance = oo_vol + k * cc_vol + (1 - k) * rs_vol
    annualized_vol = np.sqrt(yz_variance * 252)

    return annualized_vol


# =============================================================================
# BLACK-SCHOLES OPTION PRICING
# =============================================================================

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    Black-Scholes European option price.

    S = spot price, K = strike, T = time to expiry (years),
    r = risk-free rate, sigma = implied volatility (annualized).

    Returns the theoretical mid-price in dollars per share.
    """
    if T <= 0:
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price


# =============================================================================
# GREEKS CALCULATION
# =============================================================================

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Analytical Greeks from the Black-Scholes model.

    Returns dict with:
        delta  = dPrice/dSpot
        gamma  = d2Price/dSpot2
        vega   = dPrice/dIV (per 1% IV move)
        theta  = dPrice/dTime (per calendar day, negative for longs)
    """
    if T <= 0:
        return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = -norm.cdf(-d1)

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    if option_type == 'call':
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
                r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) +
                r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}


# =============================================================================
# POSITION SIZING
# =============================================================================

def calculate_position_size(capital, spot_price, iv_entry, max_risk_pct=0.02):
    """
    Determine how many contracts to trade based on capital at risk.

    The "expected move" from options pricing is:
        expected_move = spot * IV * sqrt(DTE/365)

    For a short straddle, the max loss proxy is approximately
    2x the expected move (covering both directions).

    We size the position so that max_loss <= max_risk_pct * capital.

    Returns:
        n_contracts: integer number of contracts (minimum 1)
        dollar_risk: estimated dollar risk for this position
        risk_pct: actual risk as % of capital
    """
    # Expected move over 1 day (earnings is ~1 day holding)
    expected_move_pct = iv_entry * np.sqrt(1 / 252)

    # Dollar risk per contract: 2x expected move * 100 shares
    # The 2x accounts for tail scenarios beyond the expected move
    dollar_risk_per_contract = 2.0 * expected_move_pct * spot_price * 100

    # Maximum contracts allowed
    max_dollar_risk = capital * max_risk_pct
    n_contracts = max(1, int(max_dollar_risk / dollar_risk_per_contract))

    # Cap at a reasonable number (avoid over-concentration)
    n_contracts = min(n_contracts, 10)

    actual_risk = n_contracts * dollar_risk_per_contract
    risk_pct = actual_risk / capital * 100

    return n_contracts, actual_risk, risk_pct


# =============================================================================
# BACKTEST ENGINE (V2 WITH CAPITAL MANAGEMENT AND SKEW)
# =============================================================================

def run_backtest_v2(earnings_df, prices_df, use_skew=False):
    """
    Run the full backtest with capital management.

    For each earnings event:
        1. Size the position based on current capital and risk budget
        2. Price options (with or without skew)
        3. Compute P&L per contract and scale by position size
        4. Update equity curve

    Parameters:
        use_skew: if True, apply asymmetric IV to puts and calls
                  (put IV = ATM IV + 5%, call IV = ATM IV - 3%)

    Returns:
        DataFrame with all trade details and portfolio metrics
    """

    print("=" * 70)
    label = "WITH SKEW" if use_skew else "FLAT IV (NO SKEW)"
    print(f"BACKTEST V2 - SHORT STRADDLE [{label}]")
    print("=" * 70)

    # Prepare price data
    prices_df['date'] = pd.to_datetime(prices_df['date'], utc=True).dt.tz_convert(None).dt.normalize()
    prices_df = prices_df.sort_values(['ticker', 'date'])

    # Calculate Yang-Zhang volatility for each stock
    print("Computing Yang-Zhang volatility...")
    vol_data = []
    for ticker in prices_df['ticker'].unique():
        tp = prices_df[prices_df['ticker'] == ticker].copy()
        tp['realized_vol'] = calculate_yang_zhang_volatility(tp, window=30)
        vol_data.append(tp)
    prices_df = pd.concat(vol_data, ignore_index=True)

    # Prepare earnings dates
    earnings_df['earnings_date'] = pd.to_datetime(earnings_df['earnings_date']).dt.normalize()

    results = []
    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    skipped = 0

    print(f"Starting capital: ${capital:,.0f}")
    print(f"Processing {len(earnings_df)} earnings events...\n")

    for idx, earning in earnings_df.iterrows():
        ticker = earning['ticker']
        earnings_date = earning['earnings_date']

        # Find entry date (last trading day <= T-1)
        target_entry = earnings_date - timedelta(days=1)
        target_exit = earnings_date

        ticker_prices = prices_df[prices_df['ticker'] == ticker].sort_values('date')

        entry_rows = ticker_prices[ticker_prices['date'] <= target_entry]
        if entry_rows.empty:
            skipped += 1
            continue
        entry_row = entry_rows.iloc[-1]

        exit_rows = ticker_prices[ticker_prices['date'] >= target_exit]
        if exit_rows.empty:
            skipped += 1
            continue
        exit_row = exit_rows.iloc[0]

        spot_entry = entry_row['Close']
        spot_exit = exit_row['Close']
        realized_vol = entry_row['realized_vol']

        if np.isnan(realized_vol) or realized_vol <= 0:
            skipped += 1
            continue

        # IV estimation
        iv_atm_entry = realized_vol * IV_INFLATION
        iv_atm_exit = iv_atm_entry * IV_CRUSH

        # Apply skew if enabled
        if use_skew:
            iv_call_entry = iv_atm_entry + CALL_SKEW_ADDON   # ATM IV - 3%
            iv_put_entry = iv_atm_entry + PUT_SKEW_ADDON      # ATM IV + 5%
            iv_call_exit = iv_atm_exit + CALL_SKEW_ADDON * 0.5   # Skew flattens post-earnings
            iv_put_exit = iv_atm_exit + PUT_SKEW_ADDON * 0.5
        else:
            iv_call_entry = iv_atm_entry
            iv_put_entry = iv_atm_entry
            iv_call_exit = iv_atm_exit
            iv_put_exit = iv_atm_exit

        # ATM strike (rounded to $5)
        strike = round(spot_entry / 5) * 5

        # DTE
        near_dte_years = NEAR_DTE / 365
        exit_dte_years = max((NEAR_DTE - 1) / 365, 0.001)

        # === POSITION SIZING ===
        n_contracts, dollar_risk, risk_pct = calculate_position_size(
            capital, spot_entry, iv_atm_entry, MAX_RISK_PER_TRADE
        )

        # === ENTRY: Price options and sell at bid ===
        call_mid_entry = black_scholes_price(spot_entry, strike, near_dte_years,
                                             RISK_FREE_RATE, iv_call_entry, 'call')
        put_mid_entry = black_scholes_price(spot_entry, strike, near_dte_years,
                                            RISK_FREE_RATE, iv_put_entry, 'put')

        call_bid_entry = call_mid_entry * (1 - BID_ASK_SPREAD / 2) * (1 - SLIPPAGE)
        put_bid_entry = put_mid_entry * (1 - BID_ASK_SPREAD / 2) * (1 - SLIPPAGE)

        premium_per_share = call_bid_entry + put_bid_entry

        # === EXIT: Reprice and buy back at ask ===
        call_mid_exit = black_scholes_price(spot_exit, strike, exit_dte_years,
                                            RISK_FREE_RATE, iv_call_exit, 'call')
        put_mid_exit = black_scholes_price(spot_exit, strike, exit_dte_years,
                                           RISK_FREE_RATE, iv_put_exit, 'put')

        call_ask_exit = call_mid_exit * (1 + BID_ASK_SPREAD / 2) * (1 + SLIPPAGE)
        put_ask_exit = put_mid_exit * (1 + BID_ASK_SPREAD / 2) * (1 + SLIPPAGE)

        cost_per_share = call_ask_exit + put_ask_exit

        # === P&L CALCULATION ===
        # Per share
        commission_per_share = (4 * COMMISSION) / 100  # 4 legs, 100 shares per contract
        pnl_per_share = premium_per_share - cost_per_share - commission_per_share

        # Per contract (100 shares)
        pnl_per_contract = pnl_per_share * 100

        # Total P&L for this trade (scaled by position size)
        total_pnl = pnl_per_contract * n_contracts

        # Return on capital for this trade
        return_on_capital = total_pnl / capital * 100

        # Notional exposure (premium * 100 shares * n_contracts)
        notional_premium = premium_per_share * 100 * n_contracts

        # Margin estimate: for short straddle, brokers typically require
        # ~20% of underlying value + premium received
        margin_required = (0.20 * spot_entry * 100 + premium_per_share * 100) * n_contracts
        margin_pct = margin_required / capital * 100

        # Update capital
        capital += total_pnl
        equity_curve.append(capital)

        # === GREEKS ===
        call_greeks = calculate_greeks(spot_entry, strike, near_dte_years,
                                       RISK_FREE_RATE, iv_call_entry, 'call')
        put_greeks = calculate_greeks(spot_entry, strike, near_dte_years,
                                      RISK_FREE_RATE, iv_put_entry, 'put')

        straddle_delta = call_greeks['delta'] + put_greeks['delta']
        straddle_gamma = call_greeks['gamma'] + put_greeks['gamma']
        straddle_vega = call_greeks['vega'] + put_greeks['vega']
        straddle_theta = call_greeks['theta'] + put_greeks['theta']

        # === SKEW METRICS (only meaningful with skew enabled) ===
        put_call_iv_spread = iv_put_entry - iv_call_entry
        skew_pnl_impact = 0.0
        if use_skew:
            # P&L difference attributable to skew:
            # compare with what flat IV would give
            flat_call_entry = black_scholes_price(spot_entry, strike, near_dte_years,
                                                  RISK_FREE_RATE, iv_atm_entry, 'call')
            flat_put_entry = black_scholes_price(spot_entry, strike, near_dte_years,
                                                 RISK_FREE_RATE, iv_atm_entry, 'put')
            flat_premium = (flat_call_entry * (1 - BID_ASK_SPREAD/2) * (1 - SLIPPAGE) +
                           flat_put_entry * (1 - BID_ASK_SPREAD/2) * (1 - SLIPPAGE))
            skew_pnl_impact = (premium_per_share - flat_premium)

        results.append({
            'ticker': ticker,
            'earnings_date': earnings_date,
            'entry_date': entry_row['date'],
            'exit_date': exit_row['date'],

            # Stock
            'spot_entry': round(spot_entry, 2),
            'spot_exit': round(spot_exit, 2),
            'spot_move_pct': round((spot_exit - spot_entry) / spot_entry * 100, 2),
            'spot_move_abs_pct': round(abs(spot_exit - spot_entry) / spot_entry * 100, 2),

            # Volatility
            'realized_vol': round(realized_vol, 4),
            'iv_call_entry': round(iv_call_entry, 4),
            'iv_put_entry': round(iv_put_entry, 4),
            'iv_atm_entry': round(iv_atm_entry, 4),
            'iv_atm_exit': round(iv_atm_exit, 4),
            'iv_rv_ratio': round(iv_atm_entry / realized_vol, 2),
            'iv_crush_pct': round((iv_atm_exit - iv_atm_entry) / iv_atm_entry * 100, 1),
            'put_call_iv_spread': round(put_call_iv_spread, 4),

            # Strike
            'strike': strike,

            # Position sizing
            'n_contracts': n_contracts,
            'margin_required': round(margin_required, 0),
            'margin_pct_capital': round(margin_pct, 2),

            # P&L per share
            'premium_per_share': round(premium_per_share, 2),
            'cost_per_share': round(cost_per_share, 2),
            'pnl_per_share': round(pnl_per_share, 2),

            # P&L total (scaled)
            'total_pnl': round(total_pnl, 2),
            'return_on_capital': round(return_on_capital, 4),
            'capital_after': round(capital, 2),

            # Greeks (per share, short position so negate)
            'delta': round(-straddle_delta, 4),
            'gamma': round(-straddle_gamma, 4),
            'vega': round(-straddle_vega, 4),
            'theta': round(-straddle_theta, 4),

            # Skew
            'skew_pnl_impact': round(skew_pnl_impact, 4),
        })

    print(f"Backtest complete: {len(results)} valid trades ({skipped} skipped)")
    print(f"Final capital: ${capital:,.2f}")
    print(f"Total return: {(capital / INITIAL_CAPITAL - 1) * 100:.2f}%")

    df = pd.DataFrame(results)

    return df, np.array(equity_curve)


# =============================================================================
# PORTFOLIO METRICS
# =============================================================================

def compute_portfolio_metrics(results_df, equity_curve):
    """
    Compute professional portfolio metrics:
        - CAGR (compound annual growth rate)
        - Annualized volatility of returns
        - Max drawdown in % and $
        - Calmar ratio (CAGR / max drawdown)
        - Sharpe ratio (annualized)
        - Sortino ratio
        - Return on capital
    """

    metrics = {}

    # Basic
    metrics['initial_capital'] = INITIAL_CAPITAL
    metrics['final_capital'] = equity_curve[-1]
    metrics['total_return_pct'] = (equity_curve[-1] / INITIAL_CAPITAL - 1) * 100
    metrics['total_pnl'] = equity_curve[-1] - INITIAL_CAPITAL

    # Time period
    first_date = pd.to_datetime(results_df['earnings_date'].min())
    last_date = pd.to_datetime(results_df['earnings_date'].max())
    years = (last_date - first_date).days / 365.25
    metrics['years'] = round(years, 2)

    # CAGR
    if years > 0:
        metrics['cagr'] = ((equity_curve[-1] / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
    else:
        metrics['cagr'] = 0

    # Per-trade returns (as % of capital at time of trade)
    trade_returns = results_df['return_on_capital'].values

    # Annualized volatility
    # We have ~60 trades per year (15 stocks * 4 quarters)
    trades_per_year = len(results_df) / max(years, 1)
    metrics['trades_per_year'] = round(trades_per_year, 1)

    trade_vol = np.std(trade_returns)
    metrics['annualized_vol'] = trade_vol * np.sqrt(trades_per_year)

    # Sharpe ratio (annualized)
    avg_return = np.mean(trade_returns)
    if trade_vol > 0:
        metrics['sharpe_annualized'] = (avg_return * trades_per_year) / (trade_vol * np.sqrt(trades_per_year))
    else:
        metrics['sharpe_annualized'] = 0

    # Sortino ratio (uses only downside deviation)
    downside_returns = trade_returns[trade_returns < 0]
    if len(downside_returns) > 0:
        downside_vol = np.std(downside_returns) * np.sqrt(trades_per_year)
        metrics['sortino'] = (avg_return * trades_per_year) / downside_vol if downside_vol > 0 else 0
    else:
        metrics['sortino'] = float('inf')

    # Max drawdown ($ and %)
    equity = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity)
    drawdown_dollar = running_max - equity
    drawdown_pct = drawdown_dollar / running_max * 100

    metrics['max_drawdown_dollar'] = np.max(drawdown_dollar)
    metrics['max_drawdown_pct'] = np.max(drawdown_pct)

    # Calmar ratio = CAGR / max drawdown %
    if metrics['max_drawdown_pct'] > 0:
        metrics['calmar'] = metrics['cagr'] / metrics['max_drawdown_pct']
    else:
        metrics['calmar'] = float('inf')

    # Win rate
    metrics['win_rate'] = (results_df['total_pnl'] > 0).mean() * 100
    metrics['total_trades'] = len(results_df)

    # Profit factor
    gross_wins = results_df[results_df['total_pnl'] > 0]['total_pnl'].sum()
    gross_losses = abs(results_df[results_df['total_pnl'] < 0]['total_pnl'].sum())
    metrics['profit_factor'] = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    # Average trade metrics
    metrics['avg_pnl_per_trade'] = results_df['total_pnl'].mean()
    metrics['avg_return_per_trade'] = avg_return
    metrics['best_trade_pct'] = trade_returns.max()
    metrics['worst_trade_pct'] = trade_returns.min()

    # Margin utilization
    metrics['avg_margin_pct'] = results_df['margin_pct_capital'].mean()
    metrics['max_margin_pct'] = results_df['margin_pct_capital'].max()

    return metrics


# =============================================================================
# DRAWDOWN SERIES
# =============================================================================

def compute_drawdown_series(equity_curve):
    """
    Compute the drawdown time series from the equity curve.

    Returns:
        drawdown_pct: array of drawdown in % at each point
        drawdown_dollar: array of drawdown in $ at each point
    """
    equity = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity)
    drawdown_dollar = running_max - equity
    drawdown_pct = drawdown_dollar / running_max * 100
    return drawdown_pct, drawdown_dollar


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("\nLoading data...")
    earnings = pd.read_csv('data/earnings_data_FINAL.csv')
    prices = pd.read_csv('data/price_data_FINAL.csv')
    print(f"Loaded: {len(earnings)} earnings, {len(prices)} price records\n")

    # === RUN 1: Without skew (baseline) ===
    results_flat, equity_flat = run_backtest_v2(earnings, prices.copy(), use_skew=False)

    # === RUN 2: With skew ===
    results_skew, equity_skew = run_backtest_v2(earnings, prices.copy(), use_skew=True)

    # === COMPUTE METRICS ===
    metrics_flat = compute_portfolio_metrics(results_flat, equity_flat)
    metrics_skew = compute_portfolio_metrics(results_skew, equity_skew)

    # === PRINT COMPARISON ===
    print("\n" + "=" * 70)
    print("PORTFOLIO METRICS COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<30} {'Flat IV':>15} {'With Skew':>15}")
    print("-" * 60)
    for key in ['total_return_pct', 'cagr', 'annualized_vol', 'sharpe_annualized',
                'sortino', 'max_drawdown_pct', 'calmar', 'win_rate',
                'profit_factor', 'avg_margin_pct']:
        v1 = metrics_flat[key]
        v2 = metrics_skew[key]
        if key in ['total_return_pct', 'cagr', 'annualized_vol', 'max_drawdown_pct',
                    'win_rate', 'avg_margin_pct']:
            print(f"  {key:<28} {v1:>14.2f}% {v2:>14.2f}%")
        else:
            print(f"  {key:<28} {v1:>15.3f} {v2:>15.3f}")

    print(f"\n  {'Final capital':<28} ${metrics_flat['final_capital']:>13,.0f} ${metrics_skew['final_capital']:>13,.0f}")

    # === SAVE RESULTS ===
    results_flat.to_csv('backtest_v2_flat.csv', index=False)
    results_skew.to_csv('backtest_v2_skew.csv', index=False)
    print("\nSaved: backtest_v2_flat.csv, backtest_v2_skew.csv")

    # === SAVE METRICS ===
    import json
    with open('backtest_v2_metrics.json', 'w') as f:
        json.dump({
            'flat': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in metrics_flat.items()},
            'skew': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in metrics_skew.items()}
        }, f, indent=2, default=str)

    # Save equity curves
    np.save('equity_flat.npy', equity_flat)
    np.save('equity_skew.npy', equity_skew)

    print("Saved: backtest_v2_metrics.json, equity curves")
    print("\nDone. Run generate_charts_v2.py next.")
