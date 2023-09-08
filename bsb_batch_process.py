
import json
import os
import pickle
import random
import re
import time
import traceback

import cv2
import numpy as np
import openai.error

import bsbutil
import pantest

def make_question_until_success(labels):    
    questions = None
    while (True):
        try:
            questions = pantest.make_questions(labels).strip().split('\n')
            if (questions is not None and len(questions) == 10):
                break
            else:
                print('[bad questions]', questions)
        except openai.error.ServiceUnavailableError:
            traceback.print_exc()
            time.sleep(2)
        except Exception:
            traceback.print_exc()
            time.sleep(1)
    return questions

def write_code_to_answer_retry(labels, q, retry_tokens=10):
    ans = None
    while retry_tokens > 0:
        try:
            code, ans = pantest.write_code_to_answer(q, labels)
            if (ans is None):
                retry_tokens -= 2
            elif (ans.strip() == ''):
                retry_tokens -= 2
            else:
                break # good, exit
        except openai.error.ServiceUnavailableError:
            traceback.print_exc()
            time.sleep(2)
        except Exception as e:
            traceback.print_exc()
            retry_tokens -= 5 # this question might be bad
        print('retry...')
    if (ans is not None and ans.strip() == ''):
        ans = None
    return code, ans

def refine_answers_retry(q, a, retry_tokens=10):
    ans = None
    while retry_tokens > 0:
        try:
            ans = pantest.rephrase_answer(q, a)
            if (ans is None):
                retry_tokens -= 2
            elif (ans.strip() == ''):
                retry_tokens -= 2
            else:
                break # good, exit
        except openai.error.ServiceUnavailableError:
            traceback.print_exc()
            time.sleep(2)
        except Exception as e:
            traceback.print_exc()
            retry_tokens -= 5 # this question might be bad
        print('retry...')
    if (ans.strip() == ''):
        ans = None
    return ans

def main():
    split = 'val'

    output_dir = f'annotated_{split}'
    os.makedirs(output_dir, exist_ok=True)

    json_dir = 'D:\\datasets\\bsb_dataset\\annotations'
    with open(os.path.join(json_dir, f'panoptic_{split}.json'), 'r') as f:
        ann_json = json.load(f)
    label_dir = os.path.join(json_dir, '..', f'panoptic_{split}')

    for image in ann_json['images'][:]:
        image_id = image['id']

        output_json_file_name = os.path.join(output_dir, f'{image_id:04d}.json')
        output_pkl_file_name = os.path.join(output_dir, f'{image_id:04d}.pkl')
        if (os.path.isfile(output_json_file_name)):
            # skip
            print(f'skipping {image_id}')
            continue
        
        print(f'=== processing {image_id} ===')
        labels = bsbutil.gather_data(ann_json, label_dir, image_id)
        labels_unrelated = bsbutil.gather_data(ann_json, label_dir, random.choice(ann_json['images'])['id'])

        q_code_a = []
        questions = make_question_until_success(labels)
        questions_unrelated = make_question_until_success(labels_unrelated)

        random.shuffle(questions)
        random.shuffle(questions_unrelated)
        for q in questions[:3] + questions_unrelated[:3]:
            q = re.sub('^\d+\.', '', q).strip()
            print('Q:', q)
            code, ans = write_code_to_answer_retry(labels, q)
            print(ans)
            if (ans is not None):
                q_code_a.append([q, code, ans])
        
        # refine answers
        q_code_a_refined = []
        for q, code, a in q_code_a:
            print(f'Refine: {q}\n{a}')
            aa = refine_answers_retry(q, a)
            print(aa)
            if (aa is not None):
                q_code_a_refined.append([q, code, a, aa])

        json_part = []
        pkl_part = {'labels': labels, 'codes': []}
        for q, code, a, aa in q_code_a_refined:
            json_part.append({
                'question': q,
                'brief-answer': a,
                'answer': aa,
            })
            pkl_part['codes'].append(code)

        with open(output_json_file_name, 'w') as f:
            json.dump(json_part, f, indent=4)
        with open(output_pkl_file_name, 'wb') as f:
            pickle.dump(pkl_part, f)
        

if __name__ == '__main__':
    main()
