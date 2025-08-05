import os
import sys
import gspread
import pandas as pd
import ta 
from src.logger import logging
from src.exception import CustomException
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the input features DataFrame by applying necessary preprocessing steps.
    This function is a placeholder and should be replaced with actual transformation logic.
    """
    # Example transformation: fill NaN values with 0
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df.dropna(inplace=True)

    df['Buy'] = (df['RSI'] < 30) & (df['MA20'] > df['MA50'])
    df['Sell'] = (df['RSI'] > 70) & (df['MA20'] < df['MA50'])
    df.dropna(inplace=True)
    
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    for col in df.select_dtypes(include='bool').columns:
            df[col] = df[col].astype(int)
    return df
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = f1_score(y_train, y_train_pred)

            test_model_score = f1_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

# def log_dataframe_to_sheet(df: pd.DataFrame, sheet_name: str, tab_name: str, creds_file: str = "creds.json"):

#     try:
#         # Set up Google Sheets credentials and client
#         scope = [
#             "https://spreadsheets.google.com/feeds",
#             "https://www.googleapis.com/auth/drive"
#         ]
#         creds = ServiceAccountCredentials.from_json_keyfile_name(creds_file, scope)
#         client = gspread.authorize(creds)

#         # Open or create spreadsheet
#         try:
#             sheet = client.open(sheet_name)
#         except gspread.SpreadsheetNotFound:
#             sheet = client.create(sheet_name)

#         # Delete existing worksheet with same name if it exists
#         try:
#             worksheet = sheet.worksheet(tab_name)
#             sheet.del_worksheet(worksheet)
#         except gspread.exceptions.WorksheetNotFound:
#             pass

#         # Add new worksheet and upload data
#         worksheet = sheet.add_worksheet(title=tab_name, rows=str(len(df)), cols=str(len(df.columns)))
#         worksheet.update([df.columns.values.tolist()] + df.values.tolist())

#         logging.info(f"Data successfully written to '{sheet_name}' -> '{tab_name}'")
    
#     except Exception as e:
#         logging.error(f"Failed to log DataFrame to Google Sheets: {e}")
#         raise CustomException(e, sys)
