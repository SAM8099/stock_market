##Mini Algo-Trading App
This app fetches historical stock data, applies an RSI + MA strategy, logs trades, and uses a simple ML model to predict stock movement.

#Nifty stock used : 
HDFCBANK, RELIANCE, INFY

#Model : 
Used GridSearchCV to obtain best model among Naive Bayes, XgBoost, Logistic Regression, Gradient Boosting, Decision tree and Random forests

#Strategy logic : 
Used Yahoo finance api for dataset. Obtain RSI and moving average for 20 and 50 respectively.

#Accuracy
Achieved a accuracy of 54% which is the average for current stock prediction models.
 

