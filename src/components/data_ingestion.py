import os, sys

from src.logger import logging
from src.exception import CustomException
from src.components import data_transformation
from src.components import model_trainer

import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


@dataclass
class DataSplittionParams:
    artifacts_dir: str = "artifacts"
    # making directory
    os.makedirs(artifacts_dir, exist_ok=True)

    raw_data_path: str = os.path.join(artifacts_dir, "raw.csv")
    train_data_path: str = os.path.join(artifacts_dir, "train.csv")
    test_data_path: str = os.path.join(artifacts_dir, "test.csv")


class DataSplittion:
    def __init__(self):
        logging.info("Going to read the data and split it.")
        self.data_splittion_config=DataSplittionParams()
        
    def initiate_data_saving(self):
        """
        Reads the data from the notebook/data directory and splits it into train and data parts. After that saves the raw, train, test data into the artifacts directory and returns the path of the train and test dataset.
        """
        # reading the data  (can be modified to read data from databases)
        data=pd.read_csv("notebook\data\stud.csv")
        logging.info("Data read successfully. Trying to split it.")

        try:
            # splitting the data into training and test and saving it.
            train_data, test_data=train_test_split(data, test_size=0.2, random_state=20)

            data.to_csv(self.data_splittion_config.raw_data_path, header=True, index=False)
            train_data.to_csv(self.data_splittion_config.train_data_path, header=True, index=False)
            test_data.to_csv(self.data_splittion_config.test_data_path, header=True, index=False)

            # logging
            logging.info(f"""Data splitted and saved successfully. Paths: {
                self.data_splittion_config.raw_data_path,
                self.data_splittion_config.train_data_path,
                self.data_splittion_config.test_data_path
            }""")

            return (
                self.data_splittion_config.train_data_path,
                self.data_splittion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)


# testing
if __name__=="__main__":
    data_splittion_obj=DataSplittion()
    train_df_path, test_df_path=data_splittion_obj.initiate_data_saving()

    data_transformation=data_transformation.DataTransformer()
    train_processed, test_processed, _ = data_transformation.initialize_data_transformation(train_df_path,test_df_path)

    model_trainer=model_trainer.ModelTrainer()
    model_trainer.initialize_training(train_df_transformed=train_processed, test_df_transformed=test_processed)