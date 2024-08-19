
import sys
import os
import json
import random

import tqdm
from PIL import Image
import numpy as np

sys.path.append('./')
from agents.environment.tools import count_connected_components, astar


# dataset format:
'''
[
    {
        "image": "path/to/image",
        "label": "path/to/label/image",
        "instruction": "xxx",
        "answer": "xxx",
        "criteria": "called_segmentation", # this is for basic tasks
        "type": "question type",
    }
]
'''

building_desc = [
    ['Background'],
    ['Water'],
    ['Building No Damage'],
    ['Building Minor Damage'],
    ['Building Major Damage'],
    ['Building Total Destruction'],
    ['Vehicle'],
    ['Road-Clear'],
    ['Road-Blocked'],
    ['Tree'],
    ['Pool']
]

class_casual_names = [
    'water',
    'building without damage',
    'building with minor damage',
    'building with major damage',
    'building with total destruction',
    'vehicle', # note: class names on github is reversed
    'clear road',
    'blocked road',
    'tree',
    'pool',
]

def make_segmentation(image_path, target_path, output_file, is_val):
    seg_instruction_templates = [
        "{verb} a segmentation map that highlights {class_name} in this image.",
        "{verb} a segmentation map showing the location of {class_name} in this picture.",
        "{verb} a map that segments and identifies {class_name} within this image.",
        "{verb} a visual segmentation highlighting {class_name} in the image.",
        "{verb} a segmentation map pinpointing the {class_name} in this picture.",
        "{verb} a segmented map that marks the {class_name} in this photo.",
        "{verb} a map that segments and indicates {class_name} in the given image.",
        "{verb} a segmentation map focusing on {class_name} within this picture.",
        "{verb} a segmentation map that identifies and highlights {class_name} in this image."
    ]
    if is_val:
        seg_instruction_templates = seg_instruction_templates[-2:]
    else:
        seg_instruction_templates = seg_instruction_templates[:-2]
    
    for seg_instruction in random.sample(seg_instruction_templates, k = min(len(seg_instruction_templates), 3)):
        for i, class_name in random.sample(list(enumerate(class_casual_names)), k = 3):
            class_i = i + 1
            for verb in [random.choice(['Create', 'Generate', 'Produce'])]:
                instruction = seg_instruction.format(verb=verb, class_name=class_name)
                output_file.write(json.dumps({
                    "image": image_path,
                    "label": target_path,
                    "instruction": instruction,
                    "answer": "",
                    "criteria": f"called_segmentation_{class_i}",
                    "type": "seg",
                }) + '\n')

def make_existence(image_path, target_path, target_np, output_file, is_val):
    exist_templates = [
        "Can you see any {class_name} in this picture?",
        "Are there any {class_name} visible in this image?",
        "Does this photo contain any {class_name}?",
        "Is this image showing any {class_name}?",
        "Are there {class_name} present in this picture?",
        "Do you notice any {class_name} in this image?",
        "Is there a presence of {class_name} in this photo?",
        "Are {class_name} depicted in this image?",
        "Can {class_name} be found in this picture?",
        "Does the image include any {class_name}?"
    ]
    if is_val:
        exist_templates = exist_templates[-2:]
    else:
        exist_templates = exist_templates[:-2]
    
    for template in random.sample(exist_templates, k = min(len(exist_templates), 3)):
        for i, class_name in random.sample(list(enumerate(class_casual_names)), k = 3):
            class_i = i + 1
            gt = np.sum(target_np == class_i) > 0
            instruction = template.format(class_name=class_name)
            output_file.write(json.dumps({
                "image": image_path,
                "label": target_path,
                "instruction": instruction,
                "answer": "yes" if gt else "no",
                "criteria": None,
                "type": "existence",
            }) + '\n')

def make_area(image_path, target_path, target_np, output_file, is_val, sqmeter_per_px):
    area_templates = [
        "How much area does {class_name} cover in the image?",
        "Can you determine the area of {class_name} in the photograph?",
        "What's the total area of {class_name} displayed in the picture?",
        "Could you calculate the area of {class_name} in the image?",
        "How much space does {class_name} take up in the picture?",
        "What is the measurement of the area of {class_name} in the image?",
        "How large is the area of {class_name} in the photo?",
        "What is the size of the area that {class_name} covers in the picture?"
    ]
    if is_val:
        area_templates = area_templates[-2:]
    else:
        area_templates = area_templates[:-2]
    
    for template in random.sample(area_templates, k = min(len(area_templates), 3)):
        for i, class_name in random.sample(list(enumerate(class_casual_names)), k = 3):
            class_i = i + 1
            if class_i not in (1, 2, 3, 4, 5, 7, 8, 9):
                continue
            gt = np.sum(target_np == class_i) * sqmeter_per_px
            instruction = template.format(class_name=class_name)
            output_file.write(json.dumps({
                "image": image_path,
                "label": target_path,
                "instruction": instruction,
                "answer": f"{gt:.2f}",
                "criteria": None,
                "type": "area",
            }) + '\n')

def make_counting(image_path, target_path, target_np, output_file, is_val):
    templates = [
        "What is the count of {class_name} in this picture?",
        "Can you identify the number of {class_name} shown in this image?",
        "How many {class_name} can be seen in this photo?",
        "Could you tell me the total number of {class_name} in this image?",
        "How many {class_name} are visible in this picture?",
        "What is the total {class_name} count in this image?",
        "Can you determine the number of {class_name} in this photograph?",
        "How many {class_name} are depicted in this image?",
    ]
    if is_val:
        templates = templates[-2:]
    else:
        templates = templates[:-2]
    
    for template in random.sample(templates, k = min(len(templates), 3)):
        for i, class_name in random.sample(list(enumerate(class_casual_names)), k = 3):
            class_i = i + 1
            if class_i not in (2, 3, 4, 5, 6, 9, 10): # fix label mismatch
                continue
            gt = count_connected_components(target_np == class_i)
            instruction = template.format(class_name=class_name)
            output_file.write(json.dumps({
                "image": image_path,
                "label": target_path,
                "instruction": instruction,
                "answer": str(gt),
                "criteria": None,
                "type": "counting",
            }) + '\n')

def make_connectivity(image_path, target_path, target_np, output_file, is_val):
    templates = [
        "Is there a direct path from {A} to {B}?",
        "Is there an unobstructed route from {A} to {B}?",
        "Is there an obvious route from {A} to {B}?",
        "Is the path from {A} to {B} clear?",
        "Is there an easy passage from {A} to {B}?",
        "Is the road from {A} to {B} clear?"
    ]
    if is_val:
        templates = templates[-2:]
    else:
        templates = templates[:-2]
    
    for template in templates:
        clear_road_i = 7 # 6->7 fixed label mismatch
        if (np.sum(target_np == clear_road_i) == 0):
            # no road available
            continue

        for i in range(3):
            candidates = [(x, y) for y, x in zip(*np.where(target_np == clear_road_i))]
            (x1, y1), (x2, y2) = random.sample(candidates, k=2)

            path = astar(target_np == clear_road_i, x1, y1, x2, y2)
            if (len(path) == 0):
                # not connected
                gt = "no"
            else:
                gt = "yes"
            instruction = template.format(A=(x1, y1), B=(x2, y2))
            output_file.write(json.dumps({
                "image": image_path,
                "label": target_path,
                "instruction": instruction,
                "answer": gt,
                "criteria": None,
                "type": "connectivity",
            }) + '\n')



def make_dataset(images_dir, targets_dir, dst_path, is_val):
    
    images_list = os.listdir(images_dir)
    images_list.sort()
    # images_list = images_list[:100]
    images_ids = [path.replace('.jpg', '') for path in images_list]
    targets_list_expected = [f'{id}_lab.png' for id in images_ids]

    for target_fname in targets_list_expected:
        assert os.path.isfile(os.path.join(targets_dir, target_fname))
    
    with open(dst_path, 'w', encoding='utf-8') as f:
        for image_id, image_fname, target_fname in zip(tqdm.tqdm(images_ids), images_list, targets_list_expected):
            # print(image_fname, target_fname)
            # analyze label
            image_path = os.path.join(images_dir, image_fname)
            target_path = os.path.join(targets_dir, target_fname)

            image = Image.open(image_path).convert('RGB')
            target = Image.open(target_path)
            target_np = np.array(target)

            make_segmentation(image_path, target_path, f, is_val)
            f.flush()
            make_existence(image_path, target_path, target_np, f, is_val)
            f.flush()
            make_counting(image_path, target_path, target_np, f, is_val)
            f.flush()
            make_connectivity(image_path, target_path, target_np, f, is_val)
            f.flush()
            make_area(image_path, target_path, target_np, f, is_val, 0.02 ** 2)
            f.flush()

if __name__ == '__main__':
    make_dataset(
        "D:/LZR/Downloads/documents/RescuNet/val-org-img", 
        "D:/LZR/Downloads/documents/RescuNet/val-label-img", 
        "rescuenet_regen_plus/rescuenet_agent_val.jsonl", is_val=True)
    make_dataset(
        "D:/LZR/Downloads/documents/RescuNet/train-org-img", 
        "D:/LZR/Downloads/documents/RescuNet/train-label-img", 
        "rescuenet_regen_plus/rescuenet_agent_train.jsonl", is_val=False)
