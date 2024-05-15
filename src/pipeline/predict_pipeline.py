import os
import sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class PredictionConfig:
    model_filepath=os.path.join("artifacts","model.pkl")
    preprocessor_filepath=os.path.join("artifacts","preprocessor.pkl")


class DataPreprocessor:
    def __init__(self, gender,race_ethnicity,parental_level_of_education,lunch,test_preparation_course,reading_score,writing_score):
        """
        Requires gender,race_ethnicity,parental_level_of_education,lunch,test_preparation_course,reading_score,writing_score and sets class params.
        """
        self.prediction_config=PredictionConfig()
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score


    def prepare_input_df(self):
        """
        Uses class variables to create input_dataframe in inference phase and returns that.
        """
        inputs = {
            "gender": self.gender,
            "race_ethnicity": self.race_ethnicity,
            "parental_level_of_education": self.parental_level_of_education,
            "lunch": self.lunch,
            "test_preparation_course": self.test_preparation_course,
            "reading_score": self.reading_score,
            "writing_score": self.writing_score
        }

        # creating dataframe from inputs dict
        input_df=pd.DataFrame([inputs])
        logging.info("Dataframe prepared for inference input.")

        return input_df
    

    def preprocess_input(self):
        """
        Creates input_df using class function and variables, transforms that and returns it.
        """
        # preparing input_df
        input_df=self.prepare_input_df()

        # loading preprocessor
        logging.info("Loads preprocessor and transforming input DataFrame.")
        preprocessor=load_object(self.prediction_config.preprocessor_filepath)

        transformed_input=preprocessor.transform(input_df)
        logging.info(f"Input transformed. Returning it. It is: {transformed_input}")

        return transformed_input


class Predictor:
    def __init__(self):
        self.prediction_config=PredictionConfig()


    def predict(self, transformed_input):
        """
        Requires transformed input, loads the model, makes prediction and returns the result.
        """
        logging.info("Gonna load the model and make prediction..")

        # loading model
        model=load_object(self.prediction_config.model_filepath)

        # making prediction
        result=model.predict(transformed_input)
        logging.info(f"Prediction done. Returning it. It is: {result}")

        return result

