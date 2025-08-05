import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Add the parent directory to sys.path to allow imports from src package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def train(stock_symbol: str):
    try:
        logging.info(">>>>>> Algo-Trading System with ML started <<<<<<")
        
        # Data Ingestion
        logging.info(">>>>>> Data Ingestion Started <<<<<<")
        data_ingestion = DataIngestion(stock_symbol=stock_symbol)
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data ingestion completed. Train data path: {train_data_path}, Test data path: {test_data_path}")
        
        # Data Transformation
        logging.info(">>>>>> Data Transformation Started <<<<<<")
        data_transformation = DataTransformation(stock_symbol=stock_symbol)
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_data_path, 
            test_data_path
        )
        logging.info(f"Data transformation completed. Preprocessor saved at: {preprocessor_path}")
        
        # Model Training
        logging.info(">>>>>> Model Training Started <<<<<<")
        model_trainer = ModelTrainer(stock_symbol=stock_symbol)
        accuracy = model_trainer.initiate_model_trainer(
            train_arr,
            test_arr
        )
        logging.info(f"Model training completed. Best model accuracy: {accuracy:.4f}")
        
        logging.info(">>>>>> Stock Market Prediction Training Pipeline Completed <<<<<<")
        
        # Return the final performance metric
        return accuracy
    
    except Exception as e:
        logging.error(f"Exception occurred during training pipeline: {e}")
        raise CustomException(e, sys)
