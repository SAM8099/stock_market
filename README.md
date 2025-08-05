📈 Mini Algo-Trading App
This Streamlit-based application automates the process of stock data analysis using a combination of technical indicators and machine learning.

🧠 What It Does
✅ Fetches historical stock data using Yahoo Finance API

📊 Applies technical indicators like RSI, MA(20), and MA(50)

🧪 Trains and evaluates ML models using GridSearchCV

💹 Implements a basic RSI + MA crossover strategy

📈 Predicts stock movement using best-performing model

🗂️ Logs trades and analytics to Google Sheets

📌 Stock Universe
Focused on top Nifty stocks:

HDFCBANK

RELIANCE

INFY

🧮 Technical Indicators Used
RSI (Relative Strength Index)

Buy signal: RSI < 30

Moving Averages

20-day MA

50-day MA

Confirmation when MA20 > MA50

🧠 Machine Learning Models
The following models were tested using GridSearchCV for hyperparameter tuning:

Model	Notes:

Naive Bayes :	Required only positive features

XGBoost :	use_label_encoder=False

Logistic Regression :	max_iter=1000

Gradient Boosting :	Basic sklearn implementation

Decision Tree :	Default params

Random Forest :	Default params

🔍 Best model is automatically selected based on validation accuracy.

📊 Performance
Achieved accuracy: ~54%

⚠️ This is typical for short-term stock prediction models due to market noise.

Link to streamlit app = https://stockmarket-nl7nvqvfsxgsu9hlybvbhm.streamlit.app/
