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

def calc_slope_intercept_kalman(etfs, prices):
    """
    Utilise the Kalman Filter from the PyKalman package
    to calculate the slope and intercept of the regressed
    ETF prices.
    """
    delta = 1e-5 # 1e-3
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.vstack(
        [prices[etfs[0]], np.ones(prices[etfs[0]].shape)]
    ).T[:, np.newaxis]
    
    kf = KalmanFilter(
        n_dim_obs=1, 
        n_dim_state=2,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=1.0, # 0.01
        transition_covariance=trans_cov
    )
    
    state_means, state_covs = kf.filter(prices[etfs[1]].values)
    return state_means, state_covs    

def draw_slope_intercept_changes(prices, state_means):
    """
    Plot the slope and intercept changes from the 
    Kalman Filter calculated values.
    """
    pd.DataFrame(
        dict(
            slope=state_means[:, 0], 
            intercept=state_means[:, 1]
        ), index=prices.index
    ).plot(subplots=True, figsize=(20, 12), color=['blue','red'])
    # plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

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
    params_filled = rolling_result.params.fillna(0)
    
    assert len(params_filled['const']) == len(prices)

    return params_filled[etfs[0]], params_filled['const']

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

# Initialize the context
class Context:
    def __init__(self ):
        self.tickers = None # ["AMJ", "AMU"]
        self.qty = 20000
        self.invested = False
        self.state_means = None
        self.state_covs = None
        self.days = 0
        self.errors = []
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
        # KF parameters
        self.delta = 1e-5# np.power(0.01, 2)
        self.wt = self.delta/(1-self.delta) * np.eye(2)
        self.vt = 1.0
        self.R = None
        self.theta = np.zeros(2)
        self.P = np.zeros((2, 2))
        self.C = None
        # window size
        self.window = 50

# Define the update_positions function
# 多空方向跟底下注释的相反！
def update_positions(context, etf1_price, etf2_price, hedge_ratio):
    if (context.invested and not context.just_updated) or (not context.invested and context.just_updated) :
        if context.direction == "long":
            pnl = context.qty * (etf2_price - context.etf2_price_prev) - int(context.qty * hedge_ratio) * (etf1_price - context.etf1_price_prev)
        else:
            pnl = -context.qty * (etf2_price - context.etf2_price_prev) + int(context.qty * hedge_ratio) * (etf1_price - context.etf1_price_prev)
        return pnl
    else:
        return 0

# Define the initialize function
def initialize(context, etfs, prices):
    context.state_means, context.state_covs = calc_slope_intercept_kalman(etfs, prices)
    context.tickers = etfs

# Define the handle_data function
def handle_data(context, data, etfs, k):
    # global etf1_price_prev, etf2_price_prev
    # Calculate the current hedge ratio and forecast error
    current_state_mean = context.state_means[context.days]
    current_state_cov = context.state_covs[context.days]
    hedge_ratio = current_state_mean[0]
    intercept = current_state_mean[1]
    forecast_error = data[etfs[1]].iloc[-1] - (hedge_ratio * data[etfs[0]].iloc[-1] + intercept)
    context.errors.append(forecast_error)

    # std_dev 1: implement the Kalman Filter from scratch to calculate the standard deviation of the prediction
    #############################
    x = data[etfs[0]].iloc[-1]
    y = data[etfs[1]].iloc[-1]
    F = np.asarray([x, 1.0]).reshape((1, 2))
    # et = y - (F.dot(beta_kf.values[i,:])[0])
    if context.R is not None:
        context.R = context.C + context.wt
    else:
        context.R = np.zeros((2, 2))
    
    yhat = F.dot(context.theta)[0]
    # forecast_error = y - yhat
    Qt = F.dot(context.R).dot(F.T)[0][0] + context.vt
    std_dev = k * np.sqrt(Qt)
    context.stds.append(std_dev)

    At = context.R.dot(F.T) / Qt
    context.theta = context.theta + At.flatten() * forecast_error
    context.C = context.R - At * F.dot(context.R)

    #############################

    # # std_dev 2: calculate the standard deviation of the forecast error using rolling window
    # std_dev = np.std(context.errors[-context.window:])
    # context.stds.append(std_dev)

    # # std_dev 3: Calculate the current standard deviation of the forecast error
    # # 肯定错的，这个formula不符合逻辑
    # std_dev = np.sqrt(current_state_cov[0, 0])
    # context.stds.append(std_dev)
    
    # Trading signals and execution
    # open position
    if (not context.invested) and forecast_error < -std_dev:
        # print(f"Long {context.qty} units of {etfs[1]} and short {int(context.qty * hedge_ratio)} units of {etfs[0]} on day {context.days}")
        context.direction = "long"
        context.invested = True
        context.just_updated = True
        
    elif (not context.invested) and forecast_error > std_dev:
        # print(f"Short {context.qty} units of {etfs[1]} and Long {int(context.qty * hedge_ratio)} units of {etfs[0]} on day {context.days}")
        context.direction = "short"
        context.invested = True
        context.just_updated = True
        
    # close position
    elif context.invested and (forecast_error > -1 * std_dev or forecast_error < 1 * std_dev):
        # print(f"Closing positions on day {context.days}")
        context.invested = False
        context.just_updated = True

    else:
        context.just_updated = False

    etf1_price = data[etfs[0]].iloc[-1]
    etf2_price = data[etfs[1]].iloc[-1]
    pnl = update_positions(context, etf1_price, etf2_price, hedge_ratio)
    context.pnl_history.append(pnl)
    context.position_value = context.qty * etf2_price + int(context.qty * hedge_ratio) * etf1_price
        
    daily_return = pnl / context.position_value
    context.daily_returns.append(daily_return)
        
    nav = context.nav_history[-1] * (1 + pnl / context.position_value)
    context.nav_history.append(nav)
    context.etf1_price_prev = etf1_price
    context.etf2_price_prev = etf2_price
    
    context.just_updated = False
    context.days += 1


# # Define the handle_data function
# def handle_data(context, data, etfs):
#     ### DEBUG
#     # global etf1_price_prev, etf2_price_prev
#     # Calculate the current hedge ratio and forecast error
#     current_state_mean = context.state_means[context.days]
#     current_state_cov = context.state_covs[context.days]
#     hedge_ratio = current_state_mean[0]
#     intercept = current_state_mean[1]
#     forecast_error = data[etfs[1]].iloc[-1] - (hedge_ratio * data[etfs[0]].iloc[-1] + intercept)
#     context.errors.append(forecast_error)
# #     print("AMU price: ", data['AMU'].iloc[-1])
# #     print("hedge ratio: ", hedge_ratio)
# #     print("forecast_error: ", forecast_error)
    
#     # Calculate the current standard deviation of the forecast error
#     std_dev = np.sqrt(current_state_cov[0, 0])
#     context.stds.append(std_dev)
# #     print("sqrt(Qt): ", std_dev,"\n")
    
#     # Trading signals and execution
#     # open position
#     if (not context.invested) and forecast_error < -std_dev:
#         # print(f"Short {context.qty} units of {etfs[1]} and long {int(context.qty * hedge_ratio)} units of {etfs[0]} on day {context.days}")
# #         print(f"Long {context.qty} units of {etfs[1]} and short {int(context.qty * hedge_ratio)} units of {etfs[0]} on day {context.days}")
#         context.direction = "long"
#         context.invested = True
        
#     elif (not context.invested) and forecast_error > std_dev:
#         # print(f"Long {context.qty} units of {etfs[1]} and short {int(context.qty * hedge_ratio)} units of {etfs[0]} on day {context.days}")
# #         print(f"Short {context.qty} units of {etfs[1]} and Long {int(context.qty * hedge_ratio)} units of {etfs[0]} on day {context.days}")
#         context.direction = "short"
#         context.invested = True
        
#     # close position
#     elif context.invested and (forecast_error > -1 * std_dev or forecast_error < 1 * std_dev):
#         # print(f"Closing positions on day {context.days}")
#         context.invested = False
    
#     # Update position value, pnl, and net asset value
#     if context.invested:
#         etf1_price = data[etfs[0]].iloc[-1]
#         etf2_price = data[etfs[1]].iloc[-1]
#         # pnl = update_positions(context, etf1_price, etf2_price, etf1_price_prev, etf2_price_prev,hedge_ratio)
#         pnl = update_positions(context, etf1_price, etf2_price, hedge_ratio)
#         context.pnl_history.append(pnl)
#         context.position_value = context.qty * etf2_price + int(context.qty * hedge_ratio) * etf1_price
        
#         daily_return = pnl / context.position_value
#         context.daily_returns.append(daily_return)
        
#         nav = context.nav_history[-1] * (1 + pnl / context.position_value)
#         context.nav_history.append(nav)
#         ## DEBUG
#         context.etf1_price_prev = etf1_price
#         context.etf2_price_prev = etf2_price
#     else:
#         context.pnl_history.append(0)
#         context.daily_returns.append(0)
#         if len(context.nav_history) > 0:
#             context.nav_history.append(context.nav_history[-1])
#         ### DEBUG
#         context.etf1_price_prev = data[etfs[0]].iloc[-1]
#         context.etf2_price_prev = data[etfs[1]].iloc[-1]

    
    
#     context.days += 1