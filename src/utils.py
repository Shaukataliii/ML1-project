# This file would utility code that will be called from other files.
from src.logger import logging
from src.exception import CustomException

import os
import sys
import numpy as np

# import dill
import pickle
import sklearn
from sklearn.metrics import mean_squared_error, r2_score



def save_object(filepath, object):
    """
    Requires preprocessor_filepath and the proprocessor object and dumps/saves it into the provided path using pickle.
    """
    try:
        # getting directory name from the filepath and making it
        dir_name=os.path.dirname(filepath)
        os.makedirs(dir_name, exist_ok=True)

        # saving the preprocessor
        with open(filepath,"wb") as file:
            pickle.dump(object,file)

        # logging
            logging.info(f"Object saved. Path: {filepath}")

    except Exception as e:
        raise CustomException(e, sys)


def train_evaluate_models(x_train, x_test, y_train, y_test, models: dict):
    """
    Requires transformed x_train, x_test, y_train, y_test, models: dict (containing modelname along with its object), trains the models, evaluate them and return the name of the best model. Raises exceptions when things are not right.
    """
    try:
        model_rmse={}
        model_r2={}

        for model_name, model_obj in models.items():

            ### training & testing
            # training 
            train_y_pred=model_obj.fit(x_train,y_train)
            # testing
            test_y_pred=model_obj.predict(x_test)

            ### evaluation
            # rmse calculation
            rmse=np.sqrt(mean_squared_error(y_test,test_y_pred))
            # r2-score calculation
            r2=r2_score(y_test,test_y_pred)

            # saving results
            model_rmse[model_name]=rmse
            model_r2[model_name]=r2


        # finding the best model
        # best by rmse
        best_by_rmse,best_rmse=sorted(model_rmse.items(), key=lambda item: item[1], reverse=False)[0]
        # best by r2
        best_by_r2,best_r2=sorted(model_r2.items(), key=lambda item: item[1], reverse=True)[0]

        if best_by_r2 == best_by_rmse:

            return best_by_rmse

        else:
            raise CustomException(f"Best model by rmse and by r2 are not same. By rmse:- {best_by_rmse}: {best_rmse}, By r2:- {best_by_r2}: {best_r2}")

    except Exception as e:
        raise CustomException(e, sys)
    

def train_evaluate_best_model(x_train, x_test, y_train, y_test, model_obj: dict):
    """
    Requires transformed x_train, x_test, y_train, y_test, model: dict (containing modelname along with its object), trains the model, evaluate it and return the model, rmse and r2_score.
    """
    try:
        ### training & testing
        # training
        model=model_obj
        train_y_pred=model.fit(x_train,y_train)
        # testing
        test_y_pred=model.predict(x_test)
        
        ### evaluation
        # rmse calculation
        rmse=np.sqrt(mean_squared_error(y_test,test_y_pred))
        # r2-score calculation
        r2=r2_score(y_test,test_y_pred)

        return (
            model,
            rmse,
            r2
        )
    
    except Exception as e:
        raise CustomException(e, sys)