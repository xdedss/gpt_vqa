
import json
import os
import time

import cv2
import numpy as np

def find_ann_of_image_id(ann_json, image_id):
    annotations = ann_json['annotations']
    for annotation in annotations:
        if (annotation['image_id'] == image_id):
            return annotation
    return None

aircraft_names = ['SU-35','C-130','C-17','C-5','F-16','TU-160',
'E-3','B-52','P-3C','B-1B','E-8','TU-22','F-15','KC-135',
'F-22','FA-18','TU-95','KC-10','SU-34','SU-24']

class_names = {i:an for i,an in enumerate(aircraft_names)}

def gather_data(pan_json, image_dir, image_id):
    # build objects
    objects = dict()
    segmentations = dict()
    ann = find_ann_of_image_id(pan_json, image_id)
    if (ann is None):
        print('bad annotation or image id', image_id)
        return None
    img = cv2.imread(os.path.join(image_dir, ann['file_name']))
    if (img is None):
        print('bad image', ann['file_name'])
        return None
    b, g, r = img.transpose(2, 0, 1)
    for seg in ann['segments_info']:
        cat_id = seg['category_id']
        cat_name = class_names[cat_id]
        id = seg['id']
        bbox = seg['bbox']
        # things
        if (not cat_name in objects):
            objects[cat_name] = []
        objects[cat_name].append({
            'bbox': bbox,
        })
    
    res = {
        'objects': objects,
    }
    return res

if __name__ == '__main__':

    json_dir = 'E:\\LZR\\Storage\\Source\\Dataset\\airvic'

    # with open(os.path.join(json_dir, 'instance_val.json'), 'r') as f:
    #     instance_json = json.load(f)
    with open(os.path.join(json_dir, 'panoptic_mar20_final_val.json'), 'r') as f:
        pan_json = json.load(f)
    print(gather_data(pan_json, os.path.join(json_dir, 'JPEGImages'), 0))


