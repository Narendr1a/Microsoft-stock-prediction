import sys
import os

def error_message_detail(error, error_detail):
    _, _, exc_tb = error_detail.exc_info()

    filename = exc_tb.tc_frame.f_code.co_filename

    error_message = f"Error occured on python script name [{filename}] line number[{exc_tb.tb_lineno}] error message[{str(error)}]"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail = error_detail)
    def __str__(self):
        return self.error_message

