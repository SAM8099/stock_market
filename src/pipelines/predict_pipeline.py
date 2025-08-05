import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object ,  transform_features
import os

class PredictPipeline:
    def __init__(self, stock_symbol: str):
        self.stock_symbol = stock_symbol


    def predict(self,features):
        try:
            stock_folder = self.stock_symbol.replace('.', '_')
            logging.info(f"Loading model and preprocessor for stock: {stock_folder}")
            model_path=os.path.join("artifacts",f'{stock_folder}',"model.pkl")
            preprocessor_path=os.path.join('artifacts',f'{stock_folder}',"preprocessor.pkl")
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")

            data_scaled=preprocessor.transform(features)
            print("After Scaling")
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    """
    Collects feature data for a single stock entry and converts it into a DataFrame.
    """

    def __init__(self,
                 open_price: float,
                 high_price: float,
                 low_price: float,
                 close_price: float,
                 volume: float ):
        self.open_price = open_price
        self.high_price = high_price
        self.low_price = low_price
        self.close_price = close_price
        self.volume = volume


    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            data_dict = {
                "Open": [self.open_price],
                "High": [self.high_price],
                "Low": [self.low_price],
                "Close": [self.close_price],
                "Volume": [self.volume]
            }

            df = pd.DataFrame(data_dict)
            logging.info("Custom stock data converted to DataFrame")
            return df

        except Exception as e:
            logging.error(f"Exception in CustomData.get_data_as_dataframe: {e}")
            raise CustomException(e, sys)

