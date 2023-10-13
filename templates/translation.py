

import time, sys, os, math, random
import numpy as np

def translate(text, from_lang, to_lang, additional_info=''):
    json_format = '{ "result": "translation result" }'
    return f'''You are an AI assistant that performs translation from {from_lang} to {to_lang}. Your response should strictly follow json format {json_format}
{additional_info}
Here is the user input:

{text}
'''
