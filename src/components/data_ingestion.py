import yfinance as yf
from dataclasses import dataclass
from src.logger import logging
import sys
import os
import ta
from src.exception import CustomException
import pandas as pd
import numpy as np
from src.utils import  transform_features
@dataclass
class DataIngestionConfig:
    stock_symbol: str
    period: str = "6mo"
    interval: str = "1d"

    def __post_init__(self):
        base = "artifacts"
        stock_folder = self.stock_symbol.replace('.', '_')
        self.artifact_dir = os.path.join(base, stock_folder)
        os.makedirs(self.artifact_dir, exist_ok=True)
        self.train_data_path = os.path.join(self.artifact_dir, "train.csv")
        self.test_data_path = os.path.join(self.artifact_dir,"test.csv")
        self.raw_data_path = os.path.join(self.artifact_dir,"data.csv")

class DataIngestion:
    def __init__(self, stock_symbol: str, period: str = "6mo", interval: str = "1d"):
        self.stock_symbol = stock_symbol
        self.period = period
        self.interval = interval
        self.ingestion_config = DataIngestionConfig(stock_symbol=stock_symbol, period=period, interval=interval)

    def initiate_data_ingestion(self):
        try:
            df = yf.Ticker(self.stock_symbol).history(period=self.period, interval=self.interval)
            # Basic preprocessing for Titanic dataset
            logging.info('Performing basic preprocessing on dataset')
            
            # Drop unnecessary columns 
            df = df.drop(['Dividends', 'Stock Splits'], axis=1, errors='ignore')
 
            #log_dataframe_to_sheet(df, f"{self.data_transformation_config.stock_symbol}", f"{self.data_transformation_config.stock_symbol}_Data")
    
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df = transform_features(df)
            logging.info("Transformed features for the stock data")

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            def time_series_split(df, train_size=0.8):
                split_index = int(len(df) * train_size)
                train_df = df.iloc[:split_index]
                test_df = df.iloc[split_index:]
                return train_df, test_df
            train_set, test_set = time_series_split(df)

            train_set.to_csv(self.ingestion_config.train_data_path, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, header=True)

            logging.info(f"Ingestion of the {self.stock_symbol} data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error(f"Error fetching data for {self.stock_symbol}: {str(e)}")
            raise CustomException(e, sys)
