import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_preprocessor_obj

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


@dataclass
class TransformationConfig:
    preprocessor_obj_path=os.path.join("artifacts", "preprocessor.pkl")
    num_feats=["reading_score","writing_score"]
    cat_feats=["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]
    target_feat="math_score"
    


class DataTransformer:
    def __init__(self):
        self.transformation_config=TransformationConfig()

    def get_preprocessor_obj(self):
        """
        This function prepares the data pre-processor object and returns it.
        """
        try:
            # creating pipeline for numerical features
            num_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            # creating pipeline for categorical features
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoding",OneHotEncoder()),
                    ("scaling",StandardScaler(with_mean=False))
                ]
            )

            # combining both pipelines and creating column transformer
            pre_processor=ColumnTransformer(
                [
                    ("numerical_features_transformer",num_pipeline,self.transformation_config.num_feats),
                    ("categorical_features_transformer",cat_pipeline,self.transformation_config.cat_feats)
                ]
            )

            # logging
            logging.info(f"Pre-processor prepared. Cat_feats: {self.transformation_config.cat_feats} and num_feats: {self.transformation_config.cat_feats}")

            return pre_processor
        
        except Exception as e:
            raise CustomException(e, sys)
    

    def initialize_data_transformation(self,train_data_path,test_data_path):
        """
        Require train and test data paths. Reads data using those paths, gets the pre-processor, processes/transforms the data, saves the preprocessor and returns the transformed train, test data and processor_file_path
        """
        try:
            # reading datasets
            logging.info("Gonna read data and apply transformation.")
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)

            preprocessor=self.get_preprocessor_obj()

            # separating dependent and independent features
            x_train_df=train_df.drop(self.transformation_config.target_feat, axis=1)
            y_train_df=train_df[self.transformation_config.target_feat]

            x_test_df=test_df.drop(self.transformation_config.target_feat, axis=1)
            y_test_df=test_df[self.transformation_config.target_feat]

            # transforming the inputs of train and test datasets
            x_train_df_transformed=preprocessor.fit_transform(x_train_df)
            x_test_df_transformed=preprocessor.transform(x_test_df)

            # combining the dependent and independent features
            train_df_processed=np.c_[ x_train_df_transformed,np.array(y_train_df) ]
            test_df_processed=np.c_[ x_test_df_transformed,np.array(y_test_df) ]

            # logging
            logging.info("Transformed/processed the train and test datasets successfully. Saving the pre-processor and returning processed train and test datasets.")

            # saving the pro-processor
            save_preprocessor_obj(
                preprocessor_filepath=self.transformation_config.preprocessor_obj_path,
                preprocessor_obj=preprocessor
            )

            # logging
            logging.info(f"Saved the preprocessor. Path: {self.transformation_config.preprocessor_obj_path}")

            return (
                train_df_processed,
                test_df_processed,
                self.transformation_config.preprocessor_obj_path
            )


        except Exception as e:
            raise CustomException(e, sys)
