import exception
from src.logger import logging
import sys



def create_error_message(error, error_details:sys):
    """
    This function requires an error message and a sys object that contains the error_details. It extracts the details i.e. filename, lineno from the error_details and uses the error to create a properly formamtted error message and then returns it.
    """
    # gathering information
    _,_,exec_tb=error_details.exc_info()
    filename=exec_tb.tb_frame.f_code.co_filename
    lineno=exec_tb.tb_lineno

    # formatting error message
    error_message=f"An exception occured in the code in file: {filename}, on line number: {lineno}, error message: {error}"

    return error_message



# create custsom exception handling
class CustomException(Exception):
    # work for raising the exception
    def __init__(self, error_message, error_details:sys):
        super().__init__(error_message)
        # preparing the error message for logging
        self.error_message=create_error_message(error_message, error_details)

    # used to displaying/printing the exception
    def __str__(self):
        return self.error_message


# testing code
# if __name__=="__main__":
#     try:
#         # some code that might raise an exception
#         by
#     except Exception as e:
#         print(CustomException(str(e), sys))