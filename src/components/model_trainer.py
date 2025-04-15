import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define classification models
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"),
                "CatBoost Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }

            # Define hyperparameters for grid search
            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'splitter': ['best', 'random'],
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'criterion': ['gini', 'entropy']
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05],
                    'n_estimators': [50, 100, 200]
                },
                "Logistic Regression": {
                    'C': [0.1, 1, 10],
                    'solver': ['lbfgs', 'liblinear']
                },
                "XGBClassifier": {
                    'learning_rate': [0.1, 0.01, 0.05],
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7]
                },
                "CatBoost Classifier": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [50, 100, 200]
                },
                "AdaBoost Classifier": {
                    'learning_rate': [0.1, 0.01, 0.05],
                    'n_estimators': [50, 100, 200]
                }
            }

            # Evaluate models and find the best one
            best_model_name, best_model_score, best_test_score, best_model = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, param=params, scoring="accuracy"
            )

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Best Model CV Accuracy: {best_model_score:.4f}")
            logging.info(f"Best Model Test Accuracy: {best_test_score:.4f}")

            return best_test_score

        except Exception as e:
            raise CustomException(e, sys)