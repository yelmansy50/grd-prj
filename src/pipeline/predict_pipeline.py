import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            # Debugging: Check if files exist
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")

            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        no: int,
        time: float,
        protocol: str,
        length: int,
        source_type: str,
        destination_type: str,
    ):
        self.no = no
        self.time = time
        self.protocol = protocol
        self.length = length
        self.source_type = source_type
        self.destination_type = destination_type

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "No.": [self.no],
                "Time": [self.time],
                "Protocol": [self.protocol],
                "Length": [self.length],
                "Source_Type": [self.source_type],
                "Destination_Type": [self.destination_type],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)