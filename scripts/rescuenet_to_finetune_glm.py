

# in sample
'''
{"image": "D:/LZR/Downloads/documents/RescuNet/val-org-img\\10781.jpg",
 "label": "D:/LZR/Downloads/documents/RescuNet/val-label-img\\10781_lab.png",
   "instruction": "Produce a segmented map that marks the building with total destruction in this photo.",
   "answer": "", 
   "criteria": "called_segmentation_5", 
   "type": "seg"
   }

'''

# out sample
'''
{
    "img": "/hdd0/lzr/bsb_dataset/image_val_png/47.png",
    "prompt": "物体面积的比例是多少？",
    "label": "物体面积约占总面积的63.03%。"
}
'''

input_file = 'rescuenet_regen/rescuenet_agent_train_part.json'
output_file = 'finetune_glm_rescuenet_train_0.2.json'

import os
import json

import random

res_array = []

with open(input_file, 'r') as f_in:
    objects = [json.loads(line) for line in f_in.readlines() if line.strip() != '']
    for o in objects:
        if (random.random() > 0.2):
            continue
        type = o['type']
        if (type not in ['existence', 'counting', 'connectivity']):
            continue
        img_path_local = o['image']
        img_path_remote = f'/hdd0/lzr/RescueNet/train/org-img/' + os.path.basename(img_path_local)
        prompt = o['instruction']
        ans = o['answer']
        res_array.append({
            "img": img_path_remote,
            "prompt": prompt,
            "label": ans
        })
    
print(len(res_array))

with open(output_file, 'w') as f_out:
    json.dump(res_array, f_out, indent=4, ensure_ascii=False)

