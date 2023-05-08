import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from pykalman import KalmanFilter
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

def draw_date_coloured_scatterplot(etfs, prices):
    """
    Create a scatterplot of the two ETF prices, which is
    coloured by the date of the price to indicate the 
    changing relationship between the sets of prices    
    """
    plt.figure(figsize=(5.5, 4))

    # Create a yellow-to-red colourmap where yellow indicates
    # early dates and red indicates later dates
    plen = len(prices)
    colour_map = plt.cm.get_cmap('coolwarm')    
    colours = np.linspace(0.1, 1, plen)
    
    # Create the scatterplot object
    scatterplot = plt.scatter(
        prices[etfs[0]], prices[etfs[1]], 
        s=30, c=colours, cmap=colour_map, 
        edgecolor='k', alpha=0.8
    )
    
    # Add a colour bar for the date colouring and set the 
    # corresponding axis tick labels to equal string-formatted dates
    colourbar = plt.colorbar(scatterplot)
    colourbar.ax.set_yticklabels(
        [str(p.date()) for p in prices[::plen//9].index]
    )
    plt.xlabel(etfs[0])
    plt.ylabel(etfs[1])
    plt.show()

# full 503: beta0, beta1 503-29 = 474
def calc_slope_intercept_rollingOLS(etfs, prices, window=30):
    """
    Utilise the rolling regression to calculate the slope and intercept of the regressed
    ETF prices.

    etfs: ('AUM', 'AMJ')
    prices: DataFrame

    return beta0, beta1
    """
    x = prices[etfs[0]]
    y = prices[etfs[1]]
    
    # Fit the RollingOLS model
    model = RollingOLS(y, sm.add_constant(x), window=window)
    rolling_result = model.fit()
    params_filled = rolling_result.params # .fillna(0)
    
    # assert len(params_filled['const']) == len(prices)
    return params_filled[etfs[0]], params_filled['const']
    # return params_filled[etfs[0]].iloc[window-1:], params_filled['const'].iloc[window-1:]

def draw_slope_intercept_changes_rollingOLS(prices, beta0, beta1):
    """
    Plot the slope and intercept changes from the 
    Rolling OLS calculated values.
    """
    pd.DataFrame(
        dict(
            slope=beta0, 
            intercept=beta1
        ), index=prices.index
    ).plot(subplots=True, figsize=(20, 12), color=['blue','red'])
    # plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# full 503: zscores_30_1
def calc_zscore_rollingOLS(etfs, prices, beta0, beta1, window=30):
    hedge_ratio = beta0
    intercept = beta1
    # print("beta0: ", beta0.head(40))
    spreads = prices[etfs[1]] - (hedge_ratio * prices[etfs[0]])
    spreads_mavg30 = spreads.rolling(window).mean()# .iloc[window-1:]
    spreads_mavg1 = spreads.rolling(window=1).mean()# .iloc[window-1:]
    stds_30 = spreads.rolling(window).std()# .iloc[window-1:]
    zscores_30_1 = (spreads_mavg1 - spreads_mavg30)/stds_30
    # print("z scores: ", zscores_30_1.head(40))
    return zscores_30_1

# Initialize the context
class Context:
    def __init__(self ):
        self.tickers = None # ["AMJ", "AMU"]
        self.qty = 20000
        self.invested = False
        self.beta0 = None
        self.beta1 = None
        self.zscores = None
        self.days = 0
        self.spreads = []
        self.stds = []
        self.direction = None
        self.position_value = 0
        self.pnl_history = []
        self.nav_history = [1]
        self.position_value = 0
        self.daily_returns = []
        ### DEBUG
        self.etf1_price_prev = None
        self.etf2_price_prev = None
        self.just_updated = False
        # window size
        self.window = None

# Define the update_positions function
def update_positions(context, etf1_price, etf2_price, hedge_ratio):
    if (context.invested and not context.just_updated) or (not context.invested and context.just_updated) :
        if context.direction == "long":
            pnl = context.qty * (etf2_price - context.etf2_price_prev) - int(context.qty * hedge_ratio) * (etf1_price - context.etf1_price_prev)
        else:
            pnl = -context.qty * (etf2_price - context.etf2_price_prev) + int(context.qty * hedge_ratio) * (etf1_price - context.etf1_price_prev)
        return pnl
    else:
        # print("update_positions: no pnl")
        return 0

# Define the initialize function
def initialize(context, etfs, prices, window):
    context.beta0, context.beta1 = calc_slope_intercept_rollingOLS(etfs, prices, window)
    context.zscores = calc_zscore_rollingOLS(etfs, prices, context.beta0, context.beta1, window)
    context.zscores = context.zscores.iloc[2*window-2:]
    # print("z scores: ", context.zscores.head(40))
    context.beta0 = context.beta0.iloc[2*window-2:]
    context.beta1 = context.beta1.iloc[2*window-2:]
    # print("beta0: ", context.beta0.head(40))
    context.tickers = etfs
    context.window = window

# Define the handle_data function
def handle_data(context, data, etfs):
    # global etf1_price_prev, etf2_price_prev
    # Calculate the current hedge ratio and forecast error
    hedge_ratio = context.beta0.iloc[context.days]
    z_score = context.zscores.iloc[context.days]
    # print(context.days)
    # print(z_score)
    
    # Trading signals and execution
    # open position
    if (not context.invested) and z_score < -1:
        # print(f"Long {context.qty} units of {etfs[1]} and short {int(context.qty * hedge_ratio)} units of {etfs[0]} on day {context.days}")
        context.direction = "long"
        context.invested = True
        context.just_updated = True
    
    elif (not context.invested) and z_score > 1:
        # print(f"Short {context.qty} units of {etfs[1]} and Long {int(context.qty * hedge_ratio)} units of {etfs[0]} on day {context.days}")
        context.direction = "short"
        context.invested = True
        context.just_updated = True
    
    # close position
    elif context.invested and (z_score > -0.5 or z_score < 0.5):
        # print(f"Closing positions on day {context.days}")
        context.invested = False
        context.just_updated = True

    else:
        context.just_updated = False

    etf1_price = data[etfs[0]].iloc[-1]
    etf2_price = data[etfs[1]].iloc[-1]
    pnl = update_positions(context, etf1_price, etf2_price, hedge_ratio)
    # print("days: ", context.days, "pnl: ", pnl)
    # print("invested: ", context.invested, "just_updated: ", context.just_updated)
    context.pnl_history.append(pnl)
    context.position_value = context.qty * etf2_price + int(context.qty * hedge_ratio) * etf1_price
    
    daily_return = pnl / context.position_value
    context.daily_returns.append(daily_return)
    
    nav = context.nav_history[-1] * (1 + pnl / context.position_value)
    context.nav_history.append(nav)
    context.etf1_price_prev = etf1_price
    context.etf2_price_prev = etf2_price

    context.just_updated = False
    context.stds.append(1)
    context.days += 1
