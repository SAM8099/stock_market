import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    stock_symbol: str
    def __post_init__(self):
        base = "artifacts"
        stock_folder = self.stock_symbol.replace('.', '_')
        self.artifact_dir = os.path.join(base, stock_folder)
        os.makedirs(self.artifact_dir, exist_ok=True)
        self.trained_model_file_path = os.path.join(self.artifact_dir,"model.pkl")

class ModelTrainer:
    def __init__(self, stock_symbol: str):
        self.model_trainer_config = ModelTrainerConfig(stock_symbol=stock_symbol)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
            
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Multinomial Naive Bayes": MultinomialNB(),
                "Xgboost": XGBClassifier(eval_metric='logloss'),
                "Logistic Regression": LogisticRegression(max_iter=1000)
            }
            
            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 5, 10, 15],
                    'min_samples_split': [2, 5, 10]
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'bootstrap': [True, False]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [50, 100, 200],
                    'subsample': [0.7, 0.8, 0.9],
                    'max_depth': [3, 4, 5]
                },
                "Logistic Regression": {
                    'C': [0.01, 0.1, 1, 10],
                    'solver': ['liblinear', 'saga']
                },
                "Xgboost": {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 4, 5],
                    'subsample': [0.7, 0.8, 0.9]
                },
                "Multinomial Naive Bayes": {}
                # No hyperparameters to tune for MultinomialNB
            }

            logging.info("Model evaluation started")
            model_report = evaluate_models(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test,
                models=models, 
                param=params
            )
            
            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name} with score: {best_model_score}")
            
            if best_model_score < 0.65:
                logging.warning(f"Best model score {best_model_score} is below threshold of 0.65")
                logging.info("Trying with default models without hyperparameter tuning")
                
                # Try again with default parameters if no good model found
                default_scores = {}
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    default_scores[name] = accuracy_score(y_test, y_pred)
                
                best_default_score = max(sorted(default_scores.values()))
                best_default_name = list(default_scores.keys())[
                    list(default_scores.values()).index(best_default_score)
                ]
                
                if best_default_score > best_model_score:
                    best_model_score = best_default_score
                    best_model_name = best_default_name
                    best_model = models[best_model_name]
                    logging.info(f"Default model performed better: {best_model_name} with score: {best_model_score}")
            
            if best_model_score < 0.4:
                raise CustomException("No best model found with acceptable performance (score < 0.6)", sys)
                
            logging.info(f"Best model found: {best_model_name}")

            # Final training of the best model
            best_model.fit(X_train, y_train)
            
            # Save the model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Model saved at {self.model_trainer_config.trained_model_file_path}")

            # Evaluate the final model
            y_pred = best_model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            logging.info(f"Final model metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
            
            return accuracy
            
        except Exception as e:
            logging.error(f"Exception occurred during model training: {e}")
            raise CustomException(e, sys)