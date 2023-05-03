from cgi import print_arguments
from datetime import date
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

'''
Author: Deven Kanwal
Date Written: 2/14/22
This program takes in a .csv file of historical stock data of a given list of stocks and creates a dataframe with each column
being a given stock in the list and each row the date and the cells are a given stock's adjusted close price. All dates the 
market was open and SPY was traded are the dates reflectedin the DataFrame. The program also gives the option to enable the 'plot'
flag which will plot of the given daily portfolio value vs SPY's daily value. 
'''


def get_data(start_date, end_date, symbols, column_name = 'Adj Close', include_spy=True):
    df_1 = pd.read_csv("data/SPY.csv", index_col="Date", parse_dates=True, usecols=['Date', column_name])
    range_1 = pd.date_range(start=start_date, end=end_date)
    df_2 = pd.DataFrame(index=range_1)
    df_3 = df_2.join(df_1)
    df_3 = df_3.dropna()
    df_3 = df_3.rename(columns={'Adj Close': 'SPY'})
    
    for symbol in symbols:
        sentence = 'data/' + symbol + '.csv'
        df_for = pd.read_csv(sentence, index_col='Date', parse_dates=True, usecols=['Date', column_name])
        df_for = df_for.rename(columns={'Adj Close' : symbol})
        if symbol != 'SPY':
            df_3 = df_3.join(df_for)
    
    return df_3

def assess_portfolio(start_date, end_date, symbols, allocations, starting_value=1000000, risk_free_rate=0.0, sample_freq=252, plot_returns=False):
    df = get_data(start_date, end_date, symbols)
    df_2 = df.copy()
    df_2 = df_2.loc[start_date:end_date, symbols] #slices out SPY column
    df_2 = df_2.ffill().bfill() #forward fills NaN values
    df_2 /= df_2.iloc[0]
    df_2 = df_2 * allocations
    df_2 = df_2 * starting_value
    df_2['Portfolio'] = df_2.sum(axis=1)
    df_2 = df_2.round(2) #rounds the cells of the dataframe
    cumulative_return = (df_2['Portfolio']/df_2['Portfolio'].iloc[0]) - 1
    final_cumulative_return = cumulative_return[-1]
    average_daily_return = ((df_2['Portfolio']/df_2['Portfolio'].shift()) - 1).mean()
    stdev_daily_return = ((df_2['Portfolio']/df_2['Portfolio'].shift()) - 1).std()
    sharpe_ratio = (average_daily_return - risk_free_rate)/stdev_daily_return * math.sqrt(sample_freq)
    end_value = df_2.loc[end_date,'Portfolio']
    
    if plot_returns == True:

        df_3 = df['SPY'] #dataframe with just SPY for modification purposes
        cumul_spy = df_3/df_3.iloc[0] - 1
        
        cumulative_return.plot()
        cumul_spy.plot()
        plt.title("Daily Portfolio Value vs. SPY")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.legend(['Portfolio', 'SPY'])
        plt.show()

    
    
    return final_cumulative_return, average_daily_return, stdev_daily_return, sharpe_ratio, end_value
    



def backtest_helper(start_date, end_date, symbols):
    df = get_data(start_date, end_date, symbols)
    df_2 = df.copy()
    df_2 = df_2.loc[start_date:end_date, symbols] #slices out SPY column
    df_2 = df_2.ffill().bfill() #forward fills NaN values
    

    return df_2
    



    

    


