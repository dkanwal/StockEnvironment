# StockEnvironment
 Trains a tabular Q learner to create a trading strategy for a single stock.

Author: Deven Kanwal

Note: Program requires MatPlotLib, Pandas, and Numpy to run.

HOW TO RUN:
    Run from the repository's directory in your terminal with the following
    command: "python3 stock_env.py"

    The user can specify the following parameters from the command line:
        Training start date: --train_start
        Training end date: --train_end
        Testing start date: --test_start
        Testing end date: --test_end
        * Note - dates must be in the following format 'YYYY-MM-DD'

        Starting cash: --cash
        Fixed cost per trade: --fixed
        Floating cost per trade: --floating
        Stock symbol to test: --symbol
        Number of trips: --trips
    
    It is recommended that the user run the program on a number of symbols. An 
    example of specifying a symbol when running is as follows:
    "python3 stock_env.py --symbol AAPL" - for running the program on Apple's 
    stock data. 

    All stock data the program can use is stored in the data folder.

-------------------------------------------------------------------------------

TabularQLearner:
    This program builds a tabular Q learner with the option of performing dyna
    learning to update Q values. Q values are mapped to a statespace which is
    stored in a Pandas DataFrame. 

tech_ind:
    This program contains the code that takes in stock data and produces a 
    DataFrame of values for each of three stock market indicators: 
    Bollinger Bands, Relative Strength Index, and On Balance Volume. When called
    by stock_env.py, these indicators' DataFrames are fetched individually for
    quantization. 

assess:
    This program takes in a .csv file of historical stock data of a given list 
    of stocks and creates a dataframe with each column being a given stock in 
    the list and each row the date and the cells are a given stock's adjusted 
    close price. All dates the market was open and SPY was traded are the dates 
    reflectedin the DataFrame. The program also gives the option to enable the 
    'plot' flag which will plot of the given daily portfolio value vs SPY's 
    daily value. It is used to fetch and organize stock data and trading results
    from the Q learner's strategy.

stock_env:
    This program creates the quantized enviornment for the selected stock in
    conjunction with the three indicators from tech_ind.py. It simulates a 
    portfolio that begins with $200K cash to invest and trains the Q learner 
    from TabularQLearner.py to find an optimal strategy within the date range.
    The strategy that the learner converges on is then tested on unseen data. 


