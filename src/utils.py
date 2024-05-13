# This file would utility code that will be called from other files.
from src.logger import logging
from src.exception import CustomException

import os
import sys

# import dill
import pickle
import sklearn

def save_preprocessor_obj(preprocessor_filepath, preprocessor_obj):
    """
    Requires preprocessor_filepath and the proprocessor object and dumps/saves it into the provided path using pickle.
    """
    try:
        # getting directory name from the filepath and making it
        dir_name=os.path.dirname(preprocessor_filepath)
        os.makedirs(dir_name, exist_ok=True)

        # saving the preprocessor
        with open(preprocessor_filepath,"wb") as file:
            pickle.dump(preprocessor_obj,file)

        # logging
        logging.info(f"Preprocessor saved. Path: {preprocessor_filepath}")


    except Exception as e:
        raise CustomException(e, sys)
