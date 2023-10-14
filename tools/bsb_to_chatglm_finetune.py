



# 1. to chinese
# 2. convert format, copy images
# {"img": "fewshot-data/2p.png", "prompt": "这张图片的背景里有什么内容？", "label": "这张图片的背景是蒙蒙细雨。"},
# {"img": "fewshot-data/pig.png", "prompt": "这张图片的背景里有什么内容？", "label": "这张图片的背景是是虚化的。"},
# {"img": "fewshot-data/meme.png", "prompt": "这张图片的背景里有什么内容？", "label": "这张图片的背景是蓝色的木质地板。"},

import os, sys, json, time, math, shutil
import numpy as np
import glob

sys.path.append('./')
import baiduapi

# TODO: make two stepss decoupled
def make_json(input_dir, expected_image_root, out_file):
    json_input_files = glob.glob(os.path.join(input_dir, '*.json'))
    json_input_files.sort()
    print(f'converting {len(json_input_files)} json files')
    res_json_obj = []
    if (os.path.isfile(out_file)):
        shutil.copy(out_file, out_file + '.bak')
        try:
            with open(out_file, 'r', encoding='utf-8') as f:
                res_json_obj = json.load(f)
        except Exception as e:
            print(e)
            print('Failed to load existing file')
    
    for json_input_file in json_input_files:
        with open(json_input_file, 'r') as f:
            json_obj = json.load(f)
        img_id = int(os.path.splitext(os.path.basename(json_input_file))[0])
        if (img_id <= 48):
            continue # temp: skip what we already have
        print('processing ', img_id)
        # src format:
        # [
        #     {
        #         "question": "Are there any Vehicles on the Street?",
        #         "brief-answer": "True",
        #         "answer": "Yes, there are vehicles on the street."
        #     },
        assert (type(json_obj) is list), json_obj
        # pick a question
        for i in range(len(json_obj)):
            qa = json_obj[i]
            q = qa['question']
            a = qa['answer']
            q_chinese = baiduapi.translate(q, 'en', 'zh')
            time.sleep(0.1) # take it easy
            a_chinese = baiduapi.translate(a, 'en', 'zh')
            time.sleep(0.1)
            res_json_obj.append({
                "img": (f'{expected_image_root}/{img_id}.png'), 
                "prompt": q_chinese, 
                "label": a_chinese,
            })
            print(q_chinese, a_chinese)
            # print(res_json_obj)

        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(res_json_obj, f, indent=4, ensure_ascii=False)

        # time.sleedp(100)



make_json('annotated_val', expected_image_root='/hdd0/lzr/bsb_dataset/image_val_png', out_file='finetune_glm.json')


