import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            # Updated column names based on your dataset
            numerical_columns = ["Time", "Length", "No."]
            categorical_columns = ["Source_Type", "Destination_Type", "Protocol"]

            # Define numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # Handle missing values
                    ("scaler", StandardScaler())  # Scale numerical features
                ]
            )

            # Define categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # Handle missing values
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),  # One-hot encode categorical features
                    ("scaler", StandardScaler(with_mean=False))  # Scale encoded features
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f"Train DataFrame shape: {train_df.shape}")
            logging.info(f"Test DataFrame shape: {test_df.shape}")

            logging.info("Obtaining preprocessing object")

            # Get the preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Updated target column name
            target_column_name = "label"

            # Encode the target column for multi-class classification
            label_encoder = LabelEncoder()
            train_df[target_column_name] = label_encoder.fit_transform(train_df[target_column_name])
            test_df[target_column_name] = label_encoder.transform(test_df[target_column_name])

            # Save the label encoder for future use
            save_object(
                file_path=os.path.join('artifacts', "label_encoder.pkl"),
                obj=label_encoder
            )

            # Separate input features and target variable for train and test datasets
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # Apply transformations
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Convert targets to proper format
            target_feature_train_arr = np.array(target_feature_train_df).reshape(-1, 1)
            target_feature_test_arr = np.array(target_feature_test_df).reshape(-1, 1)

            # DEBUG: Check array types and formats
            logging.info(f"Input features type: {type(input_feature_train_arr)}")
            logging.info(f"Target type: {type(target_feature_train_arr)}")

            # Handle sparse matrix case
            if hasattr(input_feature_train_arr, 'toarray'):
                input_feature_train_arr = input_feature_train_arr.toarray()
                input_feature_test_arr = input_feature_test_arr.toarray()
                logging.info("Converted sparse matrix to dense array")

            # Final verification
            logging.info(f"Final train features shape: {input_feature_train_arr.shape}")
            logging.info(f"Final train target shape: {target_feature_train_arr.shape}")

            # Safe concatenation
            train_arr = np.concatenate(
                [input_feature_train_arr, target_feature_train_arr],
                axis=1
            )
            test_arr = np.concatenate(
                [input_feature_test_arr, target_feature_test_arr],
                axis=1
            )

            logging.info(f"Combined train array shape: {train_arr.shape}")
            logging.info(f"Combined test array shape: {test_arr.shape}")

            logging.info(f"Saved preprocessing object.")

            # Save the preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            logging.error("Final verification before error:")
            if 'input_feature_train_arr' in locals():
                logging.error(f"Features shape: {input_feature_train_arr.shape}")
                logging.error(f"Features type: {type(input_feature_train_arr)}")
                logging.error(f"Features dtype: {input_feature_train_arr.dtype}")
            if 'target_feature_train_arr' in locals():
                logging.error(f"Target shape: {target_feature_train_arr.shape}")
                logging.error(f"Target type: {type(target_feature_train_arr)}")
            raise CustomException(e, sys)