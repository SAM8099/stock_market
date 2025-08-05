import sys
import ta
from dataclasses import dataclass
import pickle
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object , transform_features

@dataclass
class DataTransformationConfig:
    stock_symbol: str

    def __post_init__(self):
        # Create nested folder: artifacts/RELIANCE_NS/
        base_dir = "artifacts"
        stock_folder = self.stock_symbol.replace('.', '_')
        self.artifact_dir = os.path.join(base_dir, stock_folder)
        os.makedirs(self.artifact_dir, exist_ok=True)

        # Save the preprocessor file inside the stock-specific folder
        self.preprocessor_obj_file_path = os.path.join(self.artifact_dir,"preprocessor.pkl")
    
class DataTransformation:
    def __init__(self, stock_symbol: str):
        self.data_transformation_config=DataTransformationConfig(stock_symbol=stock_symbol)
    
    def get_data_transformer_object(self):
        try:
            logging.info("Obtaining data transformation object")
            
            numerical_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MA20', 'MA50']
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean"))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_cols)
                ],
                remainder='drop'
            )

            return preprocessor

        except Exception as e:
            logging.error(f"Error in get_data_transformer_object: {str(e)}")
            raise CustomException(e, sys)
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f"Train data shape: {train_df.shape}, Test data shape: {test_df.shape}")


            #log_dataframe_to_sheet(train_df, f"{self.data_transformation_config.stock_symbol}", f"{self.data_transformation_config.stock_symbol}_Train_Data")
            #log_dataframe_to_sheet(test_df, f"{self.data_transformation_config.stock_symbol}", f"{self.data_transformation_config.stock_symbol}_Test_Data")
            logging.info("Obtaining preprocessing object")
            preprocessing_obj= self.get_data_transformer_object()
            train_df.drop(columns=['Date', 'Dividends', 'Stock Splits', 'Buy', 'Sell'], inplace=True, errors='ignore')
            test_df.drop(columns=['Date', 'Dividends', 'Stock Splits', 'Buy', 'Sell'], inplace=True, errors='ignore')
            target_column_name = 'Target'
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing datasets")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            logging.info(f"Final training data shape: {train_arr.shape}")
            logging.info(f"Final testing data shape: {test_arr.shape}")
            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )   

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            logging.error(f"Error loading data for transformation: {str(e)}")
            raise CustomException(e, sys)