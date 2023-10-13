
import re, json

from .base import LLMParser, ParseFailed

def extract_text_between_braces(input_string):
    start_index = input_string.find('{')
    end_index = input_string.rfind('}')
    
    if start_index == -1 or end_index == -1 or end_index <= start_index:
        return None  # No opening or closing braces found, or closing brace appears before opening brace
    
    extracted_text = input_string[start_index:end_index + 1]
    return extracted_text

class JsonParser(LLMParser):

    def __init__(self):
        super().__init__()
    
    def parse(self, text: str) -> str:
        # if has braces
        t = extract_text_between_braces(text)
        if (t is None):
            raise ParseFailed(f'Error Parsing JSON: No braces: {text}')
        # if is json format
        try:
            json_obj = json.loads(t)
        except json.JSONDecodeError:
            raise ParseFailed(f'Error Parsing JSON: Not a json string: {t}')
        return json_obj



