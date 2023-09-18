
import datetime
import json
import os
import pickle
import random
import re
import subprocess
import time

import numpy as np

import oaapi


def util_log(msg):
    with open('log.txt', 'a') as f:
        f.write(f'[{time.strftime("%Y%m%d-%H%M%S")}] {msg}\n')

def execute_shell_command(command):
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
        return output.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e.output}")
        return None

def is_simple_number_array(arr):
    if (isinstance(arr, list) and len(arr) < 5):
        for element in arr:
            if not isinstance(element, (int, float)):
                return False
        return True
    else:
        return False

def format_json(json_obj, indent=4, level=0):
    '''
    json to string like this:
    
    labels = {
        'objects': {
            'persons': [
                [20, 30, 10, 10], # center x, y, w, h
                ...
            ],
            'cars': [
                ...
            ]
        },
        'segmentations': {
            'grass': ..., # np.array of bool
        }
    }
    '''
    result = ""
    indents = " " * indent * level
    indents_inner = " " * indent * (level + 1)

    if isinstance(json_obj, dict):
        result += "{" + '\n'
        for i, (key, value) in enumerate(json_obj.items()):
            result += indents_inner + repr(key) + ": "
            if isinstance(value, (dict, list)):
                result += format_json(value, indent, level + 1) + "," + '\n'
            elif isinstance(value, (int, float)):
                result += repr(value) + "," + '\n'
            else:
                result += format_json(value, indent, level + 1) + "," + '\n'
        result += indents + "}"
    elif isinstance(json_obj, list):
        if (is_simple_number_array(json_obj)):
            result = repr(json_obj)
        else:
            result += "[" + '\n'
            random.shuffle(json_obj)
            for i, item in enumerate(json_obj):
                result += indents_inner
                if (i == 0):
                    if isinstance(item, (dict, list)):
                        result += format_json(item, indent, level + 1) + "," + '\n'
                    else:
                            result += repr(item) + "," + '\n'
                else:
                    result += f'..., # {len(json_obj)} items in total\n'
                    break
            result += indents + "]"
    elif (isinstance(json_obj, np.ndarray)):
        result += "..." # + " # numpy array of " + str(json_obj.dtype)
    else:
        result += "..." # + " # " + type(json_obj).__name__

    return result

def extract_code_blocks(markdown_text):
    code_blocks = []
    pattern = r'```.*?\n([\s\S]*?)\n```'

    matches = re.finditer(pattern, markdown_text, re.MULTILINE)
    for match in matches:
        code_block = match.group(1)
        code_blocks.append(code_block)

    return code_blocks



def write_code_to_answer(question, ann_obj, *, has_seg=True):
    # replace the annotations with necessary ellipsis
    seg_related = '' if not has_seg else '''Segmentation masks are np arrays of bool. '''
    prompt = f'''here is a python annotation object we already have:

labels = {format_json(ann_obj)}

{seg_related}Bounding boxes are in format of top left x, y, w, h. The lists are unsorted. Please add python code below that can print the answer to the following question:

{question}

For yes/no questions, print True/False. For other questions, print the answer directly without format.
'''
    util_log('=============== write code to answer ===============')
    util_log(prompt)
    res = oaapi.ask_once('You are a helpful assistant.', prompt)
    util_log(res)
    codes = (extract_code_blocks(res))
    if (len(codes) == 0):
        # failed
        return None, None
    with open('temp.py', 'w') as f:
        f.write('''
import pickle
import numpy as np
import cv2
from numpy import any, all

with open('temp.pkl', 'rb') as f:
    labels = pickle.load(f)
''')
        for i, code in enumerate(codes):
            f.write(f'\n# ===================== BLOCK {i} ===============\n')
            for line in code.split('\n'):
                if (line.strip().startswith('labels =')):
                    # do not modify my labels!
                    line = line.replace('labels', 'labels_')
                f.write(line)
                f.write('\n')
    with open('temp.pkl', 'wb') as f:
        pickle.dump(ann_obj, f)
    ans = execute_shell_command('python temp.py')
    util_log(ans)

    with open('temp.py', 'r') as f:
        full_code = f.read()
    return full_code, ans



def make_questions(ann_obj, *, has_seg=True, override_examples=None):
    seg_related = '' if not has_seg else '''Area question:
What is the area of all houses in pixels?
What is the proportion of waterbody area?'''
    examples = f'''Counting questions:
How many persons are in the image?
How many vehicles are there on the road?
Existence questions:
Are there any cars in the image?
Are there houses on the dirt?
Location questions:
Where is the cat located in the image?
Where is the largest house?
Size questions:
What is the size of the smallest car object?
What is the size of the cat object?
How tall is the person on the right?
{seg_related}'''
    if (override_examples is not None):
        examples = override_examples
    prompt = f'''given objects and segmentation labels of an image, propose 10 questions that can be answered with a single sentence, without any other additional information.

example of questions:
{examples}

example of labels:
    labels = {format_json(ann_obj)}

Note that the lists are unordered. Avoid questions reguarding object's index in list. Please make clean and brief response.
'''
    util_log('=============== make_questions ===============')
    util_log(prompt)
    res = oaapi.ask_once('You are a helpful assistant.', prompt)
    util_log(res)
    return res



def extract_info(question, program_output):
    prompt = f'''Extract the answer of question "{question}" from given information below. Answer None if there is no relavent information. Do not write any explainations and only tell the answer.

information:
{program_output}
'''
    util_log('=============== extract_info ===============')
    util_log(prompt)
    res = oaapi.ask_once('You are a helpful assistant.', prompt)
    util_log(res)
    return res




def rephrase(text):
    prompt = f'''Rephrase: {text}'''
    util_log('=============== rephrase ===============')
    util_log(prompt)
    res = oaapi.ask_once('You are a helpful assistant.', prompt)
    util_log(res)
    return res



def rephrase_answer(question, answer):
    prompt = f'''Rephrase the answer in natural language. Brief answer.

Q: {question}
{answer}
'''
    util_log('=============== rephrase answer ===============')
    util_log(prompt)
    res = oaapi.ask_once('You are a helpful assistant.', prompt)
    util_log(res)
    return res


if __name__ == '__main__':
    

    
    labels = {
        'objects': {
            'persons': [
                [20, 30, 10, 123], # center x, y, w, h
                [100, 30, 10, 22], # center x, y, w, h
                [40, 30, 10, 51], # center x, y, w, h
            ],
            'cars': [
                [20, 30, 10, 10], # center x, y, w, h
                [20, 30, 10, 10], # center x, y, w, h
                [20, 30, 10, 10], # center x, y, w, h
            ],
            'cat': [
                [60, 70, 5, 5],
            ],
        },
        'stuff segmentations': {
            'grass': np.ones((128, 128), dtype=np.bool8),
            'dirt': np.ones((128, 128), dtype=np.bool8),
        }
    }

    # import bsbutil
    
    # json_dir = 'D:\\datasets\\bsb_dataset\\annotations'

    # with open(os.path.join(json_dir, 'panoptic_val.json'), 'r') as f:
    #     pan_json = json.load(f)
    # labels = (bsbutil.gather_data(pan_json, os.path.join(json_dir, '..', 'panoptic_val'), 1))

    print(format_json(labels))
    
    # code, ans = write_code_to_answer('How many vehicles are there in the image?', labels)
    # print(ans)
    
    print(make_questions(labels))

