import pandas
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


#Reads stock data using the get_data module you have already written.
#Computes the daily value for your technical indicator over an entire training or test period in a single call.
#Returns a DataFrame of the indicator for the same time period.


def get_data(start_date, end_date, symbol, column_name):
    data_index = pandas.date_range(start=start_date, end=end_date)
    new_data_frame =  pandas.DataFrame(index = data_index)
    df = pandas.read_csv(('data/' + symbol) + ".csv", index_col='Date', parse_dates = True)
    d_frame = new_data_frame.join(df, how='inner')
    d_frame = d_frame[[column_name]]
    d_frame = d_frame.rename(columns={column_name:  symbol})
    
    return d_frame

#
def bollinger_band(start_date, end_date, symbol, rate=25):
    
    #take in a dataframe of 1 column: disney stock prices (and volume?)
    d_frame = get_data(start_date, end_date, symbol, 'Adj Close')
    d_frame['SMA'] = d_frame[symbol].rolling(rate).mean()
    d_frame['STD'] = d_frame[symbol].rolling(rate).std() * 2
    d_frame['UPPER'] = d_frame['SMA'] + d_frame['STD'] 
    d_frame['LOWER'] = d_frame['SMA'] - d_frame['STD'] 
    del d_frame['STD']
    """
    plt.plot(d_frame[symbol], label = symbol)
    plt.plot(d_frame['UPPER'], label = 'UPPER')
    plt.plot(d_frame['LOWER'], label = 'LOWER')
    plt.plot(d_frame['SMA'], label = 'SMA')
    
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Bollinger Band")
    plt.legend()
    #plt.show()
    """

    
    
    return d_frame

#Relative Strength Index, Momentum indicator
def rsi(start_date, end_date, symbol, rate = 14, periods=4, ema=True):
    
    #take in a dataframe of 1 column: disney stock prices (and volume?)
    d_frame = get_data(start_date, end_date, symbol, 'Adj Close')
    #d_frame['RSI'] = d_frame['DIS'].rolling(rate).mean()
    
    close_delta = d_frame.diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    if ema == True:
	    # Use exponential moving average
        ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window = periods, adjust=False).mean()
        ma_down = down.rolling(window = periods, adjust=False).mean()
        
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    
    """

    col1 = 'steelblue'
    col2 = 'red'

    #define subplots
    fig,ax = plt.subplots()

    #add first line to plot
    ax.plot(rsi[symbol], label = 'RSI')

    #add x-axis label
    ax.set_xlabel('Date', fontsize=14)

    #add y-axis label
    ax.set_ylabel('RSI', color=col1, fontsize=16)

    #define second y-axis that shares x-axis with current plot
    ax2 = ax.twinx()

    #add second line to plot
    ax2.plot(d_frame[symbol], color = col2)

    #add second y-axis label
    ax2.set_ylabel('Price', color=col2, fontsize=16)
    plt.show()
    """

    

    return rsi



    
#On balance Volume, Momentum indicator
def obv(start_date, end_date, symbol, rate=10):
    
    #rolling window
    #take in a dataframe of 1 column: disney stock prices (and volume?)
    vol_frame = get_data(start_date, end_date, symbol, 'Volume')
    price_frame = get_data(start_date, end_date, symbol, 'Adj Close')

    #price_frame["OBV"] = (np.sign(price_frame.diff()) * vol_frame).fillna(0).cumsum()
    price_frame["OBV"] = (np.sign(price_frame.diff()) * vol_frame).rolling(rate).sum()
    """

    col1 = 'steelblue'
    col2 = 'red'

    #define subplots
    fig,ax = plt.subplots()

    #add first line to plot
    ax.plot(price_frame['OBV'], label = 'OBV')

    #add x-axis label
    ax.set_xlabel('Date', fontsize=14)

    #add y-axis label
    ax.set_ylabel('OBV', color=col1, fontsize=16)

    #define second y-axis that shares x-axis with current plot
    ax2 = ax.twinx()

    #add second line to plot
    ax2.plot(price_frame[symbol], color = col2)

    #add second y-axis label
    ax2.set_ylabel('Price', color=col2, fontsize=16)
    plt.show()
    
    
    """
    del price_frame[symbol]
    
    return price_frame


#bollinger_band('2008-01-01','2009-12-31','DIS')
#rsi('2008-01-01','2009-12-31','DIS')
#obv('2008-01-01','2009-12-31','DIS')