# This file set-ups the logging functionailiy of the packages. Every time the file runs it creates a new file using the current datetime (in specified format) and writes the provided log into it. The files are created inside the logs folder. The code also checks if the directory is present or not. If not present then creates it.

import logging
import sys
import os
from datetime import datetime


################# preparing variables
# specifying log file and logs directory names
LOG_FILE_NAME=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_directory_name="logs"

# making the logs directory if it doesn't already exist
logs_directory_path=os.path.join(os.getcwd(), logs_directory_name)
os.makedirs(logs_directory_path, exist_ok=True)

# making the logs file path
LOG_FILE_PATH=f"{os.path.join(logs_directory_path,LOG_FILE_NAME)}"


####################3 changing the basicConfiguration of logger to work for our custom package
# automatically creates the LOG_FILE_PATH, if not present.
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


############### testing the logging
# if __name__=="__main__":
#     logging.log(level=logging.INFO, msg="First log")