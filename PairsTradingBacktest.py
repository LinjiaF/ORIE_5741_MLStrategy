import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import yfinance as yf
from pykalman import KalmanFilter

def plot_spread(prices, errors, stds, neg_stds):
    # Plot the spread
    plt.figure(figsize=(20, 6))
    plt.plot(prices.index[1:], errors, color='blue', label='KF Spread')
    plt.plot(prices.index[1:], stds, linestyle='--', color='red', label='Std Dev')
    plt.plot(prices.index[1:], neg_stds, linestyle='--', color='red', label='-Std Dev')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel("Days")
    plt.ylabel("KF Spread")
    plt.title("Kalman Filter Spread")
    plt.legend()
    plt.show()

def plot_spread_rollingOLS(prices, errors, stds, neg_stds, window):
    # Plot the spread
    plt.figure(figsize=(20, 6))
    plt.plot(prices.index[2*window-2:], errors, color='blue', label='z-Score')
    plt.plot(prices.index[2*window-1:], stds, linestyle='--', color='red', label='+1')
    plt.plot(prices.index[2*window-1:], neg_stds, linestyle='--', color='red', label='-1')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel("Days")
    plt.ylabel("z-Score")
    plt.title("z-Score of Spread")
    plt.legend()
    plt.show()

def plot_nav(prices, nav, window):
    # Plot Unit NAV
    plt.figure(figsize=(20, 10))
    plt.plot(prices.index[window-1:], nav, color='blue', label='Cumulative Unit NAV')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel('Days')
    plt.ylabel('Unit NAV')
    plt.title('Cumulative Unit NAV')
    plt.legend()
    plt.show()

def plot_nav_sp500(prices, nav, sp500_nav, USTreasury_nav):
    # Get S&P 500 data from Yahoo Finance
    # index = prices.index.strftime('%Y-%m-%d')
    # sp500 = yf.download('^GSPC', start=index[0], end=index[-1])['Adj Close']
    # sp500 = pdr.get_data_yahoo('^GSPC', start=prices.index[0], end=prices.index[-1])['Adj Close']

    # Plot Unit NAV and S&P 500
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(prices.index, nav, color='red', label='Portfolio')
    ax.plot(prices.index[1:], sp500_nav, color='blue', label='S&P 500')
    ax.plot(prices.index[1:], USTreasury_nav, color='lightblue', label='10 Year US Treasury')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('Days')
    ax.set_ylabel('Cumulative Unit NAV')
    ax.set_title('Cumulative Unit NAV')
    ax.legend()
    plt.show()

def plot_nav_sp500_rollingOLS(prices, nav, sp500_nav, USTreasury_nav, window):
    # Get S&P 500 data from Yahoo Finance
    # index = prices.index.strftime('%Y-%m-%d')
    # sp500 = yf.download('^GSPC', start=index[0], end=index[-1])['Adj Close']
    # sp500 = pdr.get_data_yahoo('^GSPC', start=prices.index[0], end=prices.index[-1])['Adj Close']

    # Plot Unit NAV and S&P 500
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(prices.index[2*window-2:], nav, color='red', label='Portfolio')
    ax.plot(prices.index[2*window-1:], sp500_nav, color='blue', label='S&P 500')
    ax.plot(prices.index[2*window-1:], USTreasury_nav, color='lightblue', label='10 Year US Treasury')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('Days')
    ax.set_ylabel('Cumulative Unit NAV')
    ax.set_title('Cumulative Unit NAV')
    ax.legend()
    plt.show()

def plot_cum_returns(prices, cum_returns):
    # Plot cumulative returns
    plt.figure(figsize=(20, 10))
    plt.plot(prices.index[1:], cum_returns, color='blue', label='Cumulative Returns')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel('Days')
    plt.ylabel('Cumulative returns')
    plt.title('Cumulative Returns')
    plt.legend()
    plt.show()

def plot_drawdown(prices, drawdowns):
    # Plot the drawdowns
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(prices.index, drawdowns, color='blue')
    # ax.plot(prices.index[1:], drawdowns, color='blue')
    ax.fill_between(prices.index, drawdowns, 0, color='red', alpha=0.3)
    # ax.fill_between(prices.index[1:], drawdowns, 0, color='red', alpha=0.3)
    ax.set_xlabel('Time')
    ax.set_ylabel('Drawdown')
    ax.set_title('Drawdown')

    plt.show()

def plot_drawdown_sp500(prices, drawdowns, sp500_drawdowns, USTreasury_drawdowns):
    # Get S&P 500 data from Yahoo Finance
    # index = prices.index.strftime('%Y-%m-%d')
    # sp500 = yf.download('^GSPC', start=index[0], end=index[-1])['Adj Close']
    # sp500_nav = sp500 / sp500.iloc[0]
    # sp500_highwatermarks = np.maximum.accumulate(sp500_nav)
    # sp500_drawdowns = (sp500_nav - sp500_highwatermarks) / sp500_highwatermarks
    # Plot the drawdowns
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(prices.index, drawdowns, color='red', label='Portfolio Drawdown')
    # ax.plot(prices.index[1:], drawdowns, color='blue')
    ax.fill_between(prices.index, drawdowns, 0, color='red', alpha=0.3)
    # ax.fill_between(prices.index[1:], drawdowns, 0, color='red', alpha=0.3)

    ax.plot(prices.index[1:], sp500_drawdowns, color='blue', label='S&P 500 Drawdown')
    ax.fill_between(prices.index[1:], sp500_drawdowns, 0, color='blue', alpha=0.3)

    ax.plot(prices.index[1:], USTreasury_drawdowns, color='orange', label='10 Year US Treasury Drawdown')
    ax.fill_between(prices.index[1:], USTreasury_drawdowns, 0, color='orange', alpha=0.3)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Drawdown')
    ax.set_title('Drawdown')
    ax.legend()
    plt.show()

def plot_drawdown_sp500_rollingOLS(prices, drawdowns, sp500_drawdowns, USTreasury_drawdowns, window):
    # Get S&P 500 data from Yahoo Finance
    # index = prices.index.strftime('%Y-%m-%d')
    # sp500 = yf.download('^GSPC', start=index[0], end=index[-1])['Adj Close']
    # sp500_nav = sp500 / sp500.iloc[0]
    # sp500_highwatermarks = np.maximum.accumulate(sp500_nav)
    # sp500_drawdowns = (sp500_nav - sp500_highwatermarks) / sp500_highwatermarks
    # Plot the drawdowns
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(prices.index[2*window-2:], drawdowns, color='red', label='Portfolio Drawdown')
    # ax.plot(prices.index[1:], drawdowns, color='blue')
    ax.fill_between(prices.index[2*window-2:], drawdowns, 0, color='red', alpha=0.3)
    # ax.fill_between(prices.index[1:], drawdowns, 0, color='red', alpha=0.3)

    ax.plot(prices.index[2*window-1:], sp500_drawdowns, color='blue', label='S&P 500 Drawdown')
    ax.fill_between(prices.index[2*window-1:], sp500_drawdowns, 0, color='blue', alpha=0.3)

    ax.plot(prices.index[2*window-1:], USTreasury_drawdowns, color='orange', label='10 Year US Treasury Drawdown')
    ax.fill_between(prices.index[2*window-1:], USTreasury_drawdowns, 0, color='orange', alpha=0.3)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Drawdown')
    ax.set_title('Drawdown')
    ax.legend()
    plt.show()

def calculate_daily_returns(portfolio_nav):
    daily_returns = []

    for i in range(1, len(portfolio_nav)):
        daily_return = (portfolio_nav[i] - portfolio_nav[i - 1]) / portfolio_nav[i - 1]
        daily_returns.append(daily_return)

    return daily_returns

def calculate_drawdown(portfolio_nav):
    max_so_far = 0
    drawdowns = []

    for nav in portfolio_nav:
        max_so_far = max(max_so_far, nav)
        drawdown = -(max_so_far - nav) / max_so_far
        drawdowns.append(drawdown)

    return drawdowns

def max_drawdown(daily_returns):
    # Convert daily returns to a Pandas Series object
    daily_returns = pd.Series(daily_returns)

    # Compute the cumulative returns at each time point
    cumulative_unit_nav = (1 + daily_returns).cumprod()
    # cumulative_returns = np.cumprod(1 + daily_returns) - 1

    # Compute the highwatermarks at each time point
    highwatermarks = np.maximum.accumulate(cumulative_unit_nav)

    # Compute the drawdowns at each time point
    drawdowns = (cumulative_unit_nav - highwatermarks) / highwatermarks

    # Compute the maximum drawdown and the start and end dates of the maximum drawdown period
    max_dd = np.min(drawdowns)
    # max_dd_start_date = np.argmax(np.maximum.accumulate(cumulative_unit_nav) - cumulative_unit_nav[:np.argmax(np.maximum.accumulate(cumulative_unit_nav))])
    # max_dd_end_date = np.argmax(cumulative_unit_nav[:np.argmax(np.maximum.accumulate(cumulative_unit_nav))] + max_dd)

    return max_dd

def pnl_metrics(daily_returns, sp500_nav, sp500_drawdowns, USTreasury_nav, USTreasury_drawdowns):
    # Calculate annualized return
    annualized_return = np.mean(daily_returns) * 252

    # Calculate annualized volatility
    annualized_volatility = np.std(daily_returns) * np.sqrt(252)

    # Calculate Sharpe Ratio
    sharpe_ratio = (annualized_return - 0.02) / annualized_volatility

    # Calculate Max Drawdown
    max_dd = max_drawdown(daily_returns)
    # max_drawdown = max_drawdown(cum_returns)

    # Calculate S&P 500 annualized return, volatility and max drawdown
    sp500_annualized_return = np.mean(sp500_nav.pct_change().dropna()) * 252
    sp500_annualized_volatility = np.std(sp500_nav.pct_change().dropna()) * np.sqrt(252)
    sp500_sharpe_ratio = (sp500_annualized_return - 0.02) / sp500_annualized_volatility
    sp500_max_dd = np.min(sp500_drawdowns)

    # Calculate US Treasury annualized return, volatility and max drawdown
    USTreasury_annualized_return = np.mean(USTreasury_nav.pct_change().dropna()) * 252
    USTreasury_annualized_volatility = np.std(USTreasury_nav.pct_change().dropna()) * np.sqrt(252)
    USTreasury_max_dd = np.min(USTreasury_drawdowns)


    results = pd.DataFrame({'KF Pairs Trading': [annualized_return, annualized_volatility, sharpe_ratio, max_dd],
                           'S&P 500': [sp500_annualized_return, sp500_annualized_volatility, sp500_sharpe_ratio, sp500_max_dd],
                           '10 Year US Treasury': [USTreasury_annualized_return, USTreasury_annualized_volatility, np.nan, USTreasury_max_dd]},
                           index=['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 'Max Drawdown'])
    # results = results.to_string(header=True,
    #                             col_space=10,
    #                             justify='left',
    #                             float_format="%.3f")
    return results

