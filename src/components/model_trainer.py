# this file trains the model and returns the best model r2_score
from src.logger import logging
from src.exception import CustomException
from src.utils import train_evaluate_models, train_evaluate_best_model, save_object

from dataclasses import dataclass
import os
import sys

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
    )
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import xgboost

from sklearn.metrics import mean_squared_error, r2_score



@dataclass
class ModelTrainerConfig:
    model_filepath=os.path.join("artifacts","model.pkl")
    r2_threshold=0.6


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initialize_training(self,train_df_transformed,test_df_transformed):
        """
        Requires transformed train and test datasets with independent and dependent features, splits the dataset, trains multiple models, evaluate them, save the best model and return the name, rmse and r2_score of the best model.
        """
        try:
            logging.info("Going to split the data and define models so we can train them.")

            # separating dependent and independent features
            x_train,y_train,x_test,y_test=(
                train_df_transformed[:, :-1],
                train_df_transformed[:, -1],
                test_df_transformed[:, :-1],
                test_df_transformed[:, -1]
            )

            # defining models
            models={
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "XgBoostRegression": xgboost.XGBRegressor(),
                "LinearRegression": LinearRegression(),
                "KNeighborsRegressor": KNeighborsRegressor()
            }

            # Going to train and evaluate the models
            logging.info("Going to train and evaluate the models")
            best_modelname = train_evaluate_models(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, models=models)

            
            # logging
            logging.info(f"Training and Evaluation completed. Best model: {best_modelname}")


            ### retraining and saving the best model
            logging.info(f"Retraining the best model and saving it.")
            
            best_modelojb=models[best_modelname]

            model, rmse, r2=train_evaluate_best_model(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, model_obj=best_modelojb)

            # if r2 is less than r2-threshold raising exception (as this is a not a good model)
            if r2<self.model_trainer_config.r2_threshold:
                raise CustomException(f"Best model r2 is: {r2}, rmse: {rmse}")
            
            logging.info(f"Best model: {best_modelname}. RMSE: {rmse}, R2: {r2}")

            ## saving model
            save_object(self.model_trainer_config.model_filepath, model)

            return (
                best_modelname,
                rmse,
                r2
            )

        except Exception as e:
            raise CustomException(e, sys)