
import parsers
import parsers.json
from parsers import LLMParser, ParseFailed

import templates.translation
import oaapi

class MaxRetriesExceeded(Exception):
    def __init__(self, exception_list):
        super().__init__("MaxRetriesExceeded occurred with the following exceptions:")
        self.exception_list = exception_list

    def __str__(self):
        exception_str = "\n".join([str(exc) for exc in self.exception_list])
        return f"{super().__str__()}\n{exception_str}"


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
    parser = parsers.json.JsonParser()
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
