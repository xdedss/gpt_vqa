
import re, json

from .base import LLMParser, ParseFailed

def find_last_valid_json(input_string):
    # Find the last occurrence of ']' or '}'
    last_index = max(input_string.rfind(']'), input_string.rfind('}'))

    # If neither ']' nor '}' is found, return None
    if last_index == -1:
        return None

    # Extract the potential JSON string
    for i in range(last_index):
        if (input_string[i] in ['[', '{']):
            potential_json = input_string[i:last_index + 1]

            # Try to parse the string as JSON
            try:
                json_object = json.loads(potential_json)
                return json_object  # Return the JSON object if parsing is successful
            except json.JSONDecodeError:
                pass
    return None  # Return None if parsing fails

class LastJsonParser(LLMParser):
    ''' find the last valid json obj or array in text '''

    def __init__(self):
        super().__init__()
    
    def parse(self, text: str) -> str:
        res = find_last_valid_json(text)
        if (res is None):
            raise ParseFailed(f'Error Parsing JSON: invalid text')
        return res


def extract_text_between_braces(input_string):
    start_index = min(input_string.find('{'), input_string.find('['))
    end_index = max(input_string.rfind('}'), input_string.rfind(']'))
    
    if start_index == -1 or end_index == -1 or end_index <= start_index:
        return None  # No opening or closing braces found, or closing brace appears before opening brace
    
    extracted_text = input_string[start_index:end_index + 1]
    return extracted_text

class JsonParser(LLMParser):

    def __init__(self):
        super().__init__()
    
    def parse(self, text: str) -> dict:
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



