
import agents.parsers
from agents.parsers.json import JsonParser
from agents.parsers import LLMParser, ParseFailed

import templates.translation
import oaapi

import logging

def setup_root_logger(*, console=True, level=logging.DEBUG, filename=None):

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)
    
    formatter = logging.Formatter('[%(asctime)s][%(name)s - %(levelname)s] %(message)s')
    
    if (filename is not None):
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(level)  # Set the log level for the file handler
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    if (console):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    return root_logger

class MaxRetriesExceeded(Exception):
    def __init__(self, exception_list):
        super().__init__("MaxRetriesExceeded occurred with the following exceptions:")
        self.exception_list = exception_list

    def __str__(self):
        exception_str = "\n".join([str(exc) for exc in self.exception_list])
        return f"{super().__str__()}\n{exception_str}"

def is_exception_matched(e, exception_types):
    for exception_type in exception_types:
        try:
            if isinstance(e, exception_type):
                return True
        except TypeError:
            pass  # Some exception types cannot be directly compared using isinstance
    return False

def retry_until_succeed(func, exception_types=None, max_retries=3):
    exceptions = []
    for i in range(max_retries):
        try:
            res = func()
            return res
        except Exception as e:
            if (exception_types is not None):
                # only retry when match
                if (is_exception_matched(e, exception_types)):
                    exceptions.append(e)
                else:
                    raise
            else:
                # no limit
                exceptions.append(e)
    raise MaxRetriesExceeded(exceptions)
    

def parse_until_succeed(func, parser: LLMParser, max_retries=3):
    exceptions = []
    for i in range(max_retries):
        text = func()
        try:
            res = parser.parse(text)
            return res
        except ParseFailed as e:
            exceptions.append(e)
    raise MaxRetriesExceeded(exceptions)




def translate(text, from_lang, to_lang, additional_info, max_retries=3):
    ''' from_lang and to_lang can be in natural language '''
    template = templates.translation.translate(text, from_lang, to_lang, additional_info)
    parser = JsonParser()
    def chat_func():
        return oaapi.ask_once(
            'You are a helpful assistant.', 
            template)
    return parse_until_succeed(chat_func, parser, max_retries)


if __name__ == '__main__':
    print(
        translate(
            'Where are the objects located on the soil segment?',
            from_lang='English',
            to_lang='Chinese',
            additional_info='''Note that we are in context of remote sensing image question answering. Here are explanations of some words under this context:
segmentation: 一般指语义分割
segment: 一般指语义分割中，被分割出来的部分
object: 物体
''',
            max_retries=3,
            )
    )

