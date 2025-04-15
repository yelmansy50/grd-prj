import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
def evaluate_models(X_train, y_train, X_test, y_test, models, param,scoring="accuracy"):
    try:
        best_model_name = None
        best_model_score = -np.inf
        best_model = None
        best_test_score = None

        for model_name, model in models.items():
            para = param.get(model_name, {})

            # GridSearchCV for hyperparameter tuning
            gs = GridSearchCV(model, para, cv=3, n_jobs=-1)
            gs.fit(X_train, y_train)

            # Set the best parameters to the model
            best_model_candidate = gs.best_estimator_

            # Cross-validation score
            cv_scores = cross_val_score(best_model_candidate, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
            mean_cv_score = cv_scores.mean()

            # Train the model with best parameters
            best_model_candidate.fit(X_train, y_train)

            # Predictions
            y_test_pred = best_model_candidate.predict(X_test)

            # R2 scores
            test_model_score = r2_score(y_test, y_test_pred)

            # Check if this model is the best
            if mean_cv_score > best_model_score:
                best_model_score = mean_cv_score
                best_model_name = model_name
                best_model = best_model_candidate
                best_test_score = test_model_score

        return best_model_name, best_model_score, best_test_score, best_model

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)    