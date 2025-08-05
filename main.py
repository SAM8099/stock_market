from src.pipelines.train_pipeline import train

nifty50_stocks = ['RELIANCE.NS', 'INFY.NS', 'HDFCBANK.NS']

for stock in nifty50_stocks:
    train(stock)