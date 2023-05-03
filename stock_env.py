import argparse
from cgi import print_arguments
import pandas
import numpy as np
from tech_ind import *
from TabularQLearner import TabularQLearner
from assess import backtest_helper
import matplotlib.pyplot as plt

pandas.set_option('display.max_rows', None)


class StockEnvironment:

  def __init__ (self, fixed = 9.95, floating = 0.005, starting_cash = 1000000, share_limit = 1000, num_bins = 3):
    self.shares = share_limit
    self.fixed_cost = fixed
    self.floating_cost = floating
    self.starting_cash = starting_cash
    self.num_bins = num_bins


  def prepare_world(self, start_date, end_date, symbol): 
    """
    Read the relevant price data and calculate some indicators.
    Return a DataFrame containing everything you need.
    """
    #get data here
    column_name = 'Adj Close'
    data_index = pandas.date_range(start=start_date, end=end_date)
    new_data_frame =  pandas.DataFrame(index = data_index)
    df = backtest_helper(start_date=start_date, end_date=end_date, symbols=[symbol])
    d_frame = df
    '''d_frame = new_data_frame.join(df, how='inner')
    d_frame = d_frame[[column_name]]
    d_frame = d_frame.rename(columns={column_name:  symbol})'''
    
    

    bb_frame = bollinger_band(start_date, end_date, symbol)

    rsi_frame = rsi(start_date, end_date, symbol)
 
    rsi_frame = rsi(start_date, end_date, symbol) / 100
    obv_frame = obv(start_date, end_date, symbol, rate=10)
    bb_frame['BBR'] = (bb_frame[symbol] - bb_frame['LOWER']) / (bb_frame['UPPER'] - bb_frame['LOWER']) #normalized bollinger bands into BBR
    obv_frame['OBV'] /= obv_frame['OBV'].abs().rolling(10).sum() #normalized OBV by dividing by the total volume in the roling rate period #need to tweak this cause 10 is hard coded in

    num_points = len(bb_frame) - 25 #len(bb_frame.dropna()) #this is hard coded in becasue it's the number of nan values
    points_per_bin = num_points//self.num_bins
    bb_bin_list = []
    obv_bin_list = []
    rsi_bin_list = []
    sorted_bbs = bb_frame['BBR'].sort_values()
    sorted_obv = obv_frame['OBV'].sort_values()
    sorted_rsi = rsi_frame[symbol].sort_values()
    for i in range(self.num_bins - 1):
        bb_bin_list.append(sorted_bbs[(i+1) * points_per_bin])
        obv_bin_list.append(sorted_obv[(i+1) * points_per_bin])
        rsi_bin_list.append(sorted_rsi[(i+1) * points_per_bin])
    quantized_frame = pandas.DataFrame(bb_frame['BBR'], columns = ['BBR'])
    quantized_frame['RSI'] = rsi_frame[symbol]
    quantized_frame['OBV'] = obv_frame['OBV']
    quantized_frame = quantized_frame.dropna() # drop nan values
    new_quantized_frame = quantized_frame.copy()

    #quantize for bin 0

    new_quantized_frame.loc[quantized_frame['BBR'] < bb_bin_list[0], 'BBR'] = 0
    new_quantized_frame.loc[quantized_frame['RSI'] < rsi_bin_list[0], 'RSI'] = 0
    new_quantized_frame.loc[quantized_frame['OBV'] < obv_bin_list[0], 'OBV'] = 0
    

    for i in range(1, self.num_bins - 1):

        new_quantized_frame.loc[quantized_frame['BBR'].between(bb_bin_list[i -1], bb_bin_list[i]), 'BBR'] = i
        new_quantized_frame.loc[quantized_frame['RSI'].between(rsi_bin_list[i -1], rsi_bin_list[i]), 'RSI'] = i
        new_quantized_frame.loc[quantized_frame['OBV'].between(obv_bin_list[i -1], obv_bin_list[i]), 'OBV'] = i

    #for bin num_bins
    new_quantized_frame.loc[quantized_frame['BBR'] > bb_bin_list[self.num_bins -2], 'BBR'] = self.num_bins - 1
    new_quantized_frame.loc[quantized_frame['RSI'] > rsi_bin_list[self.num_bins -2], 'RSI'] = self.num_bins - 1
    new_quantized_frame.loc[quantized_frame['OBV'] > obv_bin_list[self.num_bins -2], 'OBV'] = self.num_bins - 1
    

    new_quantized_frame['Bases'] = new_quantized_frame['BBR'] * self.num_bins **1 + new_quantized_frame['RSI']* self.num_bins**2 + new_quantized_frame['OBV']*self.num_bins**3

    d_frame = d_frame.iloc[24:]
    
    return new_quantized_frame, d_frame


  
  def calc_state(self, day, holdings): 

    if holdings == -self.shares:
        return 0
    if holdings == 0:
        return 1
    if holdings == self.shares:
        return 2

    pass

  
  def train_learner(self, start = '2018-01-01', end = '2019-12-31', symbol = 'DIS', trips = 0, dyna = 0,
                     eps = 0.0, eps_decay = 0.0):
    """
    Construct a Q-Learning trader and train it through many iterations of a stock
    world.  Store the trained learner in an instance variable for testing.

    Print a summary result of what happened at the end of each trip.
    Feel free to include portfolio stats or other information, but AT LEAST:

    Trip 499 net result: $13600.00
    """
    """
    for each day in the training data

    """

    quant_frame, d_frame = self.prepare_world(start, end, symbol)

 
    

    df = backtest_helper(start_date=start, end_date=end, symbols=[symbol])

    benchmark_cost = self.shares * df[symbol][0]
    benchmark_cash = self.starting_cash - benchmark_cost
    benchmark_shares_result = self.shares * df[symbol][-1]
    benchmark_result = benchmark_shares_result + benchmark_cash
    benchmark_net_result = benchmark_result - self.starting_cash
    
    
    
    states = (2*self.num_bins**0) + ((self.num_bins -1) * self.num_bins **1) + ((self.num_bins -1)* self.num_bins**2) + ((self.num_bins -1)*self.num_bins**3) + 1
  
    
    learner = TabularQLearner(states=81, actions=3, epsilon=eps, epsilon_decay=eps_decay, dyna=dyna)

    for y in range(trips):
        cash = self.starting_cash
        actions_frame = d_frame.copy(deep=True)
        portfolio_val = d_frame.copy(deep=True)
        prev_portfolio_value = cash
        curr_portfolio_value = 0
        holdings = 0
        transaction_cost = 0
        
        
        for i in range(0, len(quant_frame)): # for each day in the traiing data (how do I do this?) #for i in len dataframe
        
            curr_portfolio_value = d_frame[symbol][i] * holdings + cash - transaction_cost
            
            
            s = self.calc_state(i, holdings)
            s+= quant_frame['Bases'][i]
   
 
            s = int(s)
            r = (curr_portfolio_value - prev_portfolio_value)/999999
      
            
            
            if i == 0:
                a = learner.test(s)
            else:
                a = learner.train(s, r)
            
            
            if a == 0:
                if holdings == self.shares * -1:
                    holdings -= 0
                    actions_frame[symbol][i] = 1
                    transaction_cost = 0
                    
                else:
                    curr_date = i
                    cash_gain = d_frame[symbol][curr_date] * abs(self.shares)
                    transaction_cost = ((cash_gain*self.floating_cost) + self.fixed_cost)
                    #transaction_cost = cash_gain - ((cash_gain*self.floating_cost) + self.fixed_cost)
                    holdings -= self.shares
                    cash += cash_gain
                    #cash += transaction_cost
                    #curr_portfolio_value = d_frame[symbol][curr_date] * holdings + cash
                    actions_frame[symbol][i] = a
                    
            elif a == 1:
                holdings += 0
                actions_frame[symbol][i] = a
                transaction_cost = 0
                
            elif a == 2:
                if holdings == self.shares:
                    holdings += 0
                    actions_frame[symbol][i] = 1
                    transaction_cost = 0
                    
                else:
                    curr_date = i
                    cash_loss = d_frame[symbol][curr_date] * abs(self.shares)
                    transaction_cost = ((cash_loss*self.floating_cost) + self.fixed_cost)
                    holdings += self.shares
                    cash -= cash_loss
                    #curr_portfolio_value = d_frame[symbol][curr_date] * holdings + cash 
                    actions_frame[symbol][i] = a 
                     
            if (i == len(quant_frame) - 1):
              curr_portfolio_value = d_frame[symbol][i] * holdings + cash - transaction_cost
            portfolio_val[symbol][i] = curr_portfolio_value
            prev_portfolio_value = curr_portfolio_value
            
        portfolio_val = portfolio_val.div(portfolio_val.iloc[0])
        net_value = curr_portfolio_value - self.starting_cash
        actions_frame = actions_frame[actions_frame[symbol] != 1]
        
        #print('New Action frame')
        #print(actions_frame)
        #portfolio_metrics = assess_strategy(actions_frame, self.starting_cash, self.fixed_cost, self.floating_cost, start, end)
        #print(portfolio_metrics)
        
        print("For trip number " + str(y) + ":")
        print("   Final Portfolio Value: $" + str(curr_portfolio_value))
        print("   Net Result: $" + str(net_value))
        print('Benchmark Result: $' + str(benchmark_net_result))
    plt.plot(portfolio_val, label='Q Learned Strategy')
    plt.xlabel('Date')
    plt.ylabel('Total Portfolio Value')
    plt.title('Q Learned Strategy Total Portfolio Value vs. Time')
    plt.legend()
    plt.show()
    self.learner = learner
    return 
  
  
  def test_learner(self, start = None, end = None, symbol = None):
    """
    Evaluate a trained Q-Learner on a particular stock trading task.

    Print a summary result of what happened during the test.
    Feel free to include portfolio stats or other information, but AT LEAST:

    Test trip, net result: $31710.00
    Benchmark result: $6690.0000
    """

    learner = self.learner
    quant_frame, d_frame = self.prepare_world(start, end, symbol)
    cash = self.starting_cash
    curr_portfolio_value = cash
    holdings = 0
    df = backtest_helper(start_date=start, end_date=end, symbols=[symbol])

    benchmark_cost = self.shares * df[symbol][0]
    benchmark_cash = self.starting_cash - benchmark_cost
    benchmark_shares_result = self.shares * df[symbol][-1]
    benchmark_result = benchmark_shares_result + benchmark_cash
    benchmark_net_result = benchmark_result - self.starting_cash

    for i in range(0, len(quant_frame)): # for each day in the traiing data (how do I do this?) #for i in len dataframe
        
        curr_portfolio_value = d_frame[symbol][i] * holdings + cash
        s = self.calc_state(i, holdings)
        s+= quant_frame['Bases'][i]
        s = int(s)
        
            
        a = learner.test(s)
            
        if a == 0:
            if holdings == self.shares * -1:
                holdings -= 0
            else:
                holdings -= self.shares
                cash += d_frame[symbol][i] * self.shares
                curr_portfolio_value = d_frame[symbol][i] * holdings + cash - (self.fixed_cost + self.floating_cost * d_frame[symbol][i] * holdings)
        elif a == 1:
            holdings += 0
        elif a == 2:
            if holdings == self.shares:
                holdings += 0
            else:
                holdings += self.shares
                cash -= d_frame[symbol][i] * self.shares
                curr_portfolio_value = d_frame[symbol][i] * holdings + cash - (self.fixed_cost + self.floating_cost * d_frame[symbol][i] * holdings)            
        
    net_result = curr_portfolio_value - self.starting_cash

    print('____________________________________________________')
    print('Test trip net result: $' + str(net_result))
    print('Benchmark result: $' + str(benchmark_net_result))







    pass




if __name__ == '__main__':
  # Load the requested stock for the requested dates, instantiate a Q-Learning agent,
  # and let it start trading.

  parser = argparse.ArgumentParser(description='Stock environment for Q-Learning.')

  date_args = parser.add_argument_group('date arguments')
  date_args.add_argument('--train_start', default='2018-01-01', metavar='DATE', help='Start of training period.')
  date_args.add_argument('--train_end', default='2019-12-31', metavar='DATE', help='End of training period.')
  date_args.add_argument('--test_start', default='2020-01-01', metavar='DATE', help='Start of testing period.')
  date_args.add_argument('--test_end', default='2021-12-31', metavar='DATE', help='End of testing period.')

  learn_args = parser.add_argument_group('learning arguments')
  learn_args.add_argument('--dyna', default=0, type=int, help='Dyna iterations per experience.')
  learn_args.add_argument('--eps', default=0.99, type=float, metavar='EPSILON', help='Starting epsilon for epsilon-greedy.')
  learn_args.add_argument('--eps_decay', default=0.99995, type=float, metavar='DECAY', help='Decay rate for epsilon-greedy.')

  sim_args = parser.add_argument_group('simulation arguments')
  sim_args.add_argument('--cash', default=200000, type=float, help='Starting cash for the agent.')
  sim_args.add_argument('--fixed', default=9.95, type=float, help='Fixed transaction cost.')
  sim_args.add_argument('--floating', default='0.005', type=float, help='Floating transaction cost.')
  sim_args.add_argument('--shares', default=1000, type=int, help='Number of shares to trade (also position limit).')
  sim_args.add_argument('--symbol', default='DIS', help='Stock symbol to trade.')
  sim_args.add_argument('--trips', default=500, type=int, help='Round trips through training data.')

  args = parser.parse_args()

  

  
  # Create an instance of the environment class.
  env = StockEnvironment( fixed = args.fixed, floating = args.floating, starting_cash = args.cash,
                          share_limit = args.shares )

  #env.prepare_world('2018-01-01','2019-12-31','DIS', 3)
  
  #env.calc_state('2018-01-01','2019-12-31','DIS', 8)

  # Construct, train, and store a Q-learning trader.
  env.train_learner( start = args.train_start, end = args.train_end,
                     symbol = args.symbol, trips = args.trips, dyna = args.dyna,
                     eps = args.eps, eps_decay = args.eps_decay)

  # Test the learned policy and see how it does.

  # In sample.
  #env.test_learner( start = args.train_start, end = args.train_end, symbol = args.symbol)

  # Out of sample.  Only do this once you are fully satisfied with the in sample performance!
  env.test_learner( start = args.test_start, end = args.test_end, symbol = args.symbol )

  
  
    

