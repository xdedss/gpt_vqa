
import json
import os
import time

import cv2
import numpy as np

import logging

logger = logging.getLogger(__name__)

def log(message):
    logger.info(message)

def find_ann_of_image_id(ann_json, image_id):
    annotations = ann_json['annotations']
    for annotation in annotations:
        if (annotation['image_id'] == image_id):
            return annotation
    return None


class_names = {
    0: "Background",
    1: "Street",
    2: "Soil", # Permeable Area
    3: "Waterbody", # Lakes
    4: "Swimming pool",
    5: "Harbor",
    6: "Vehicle",
    7: "Boat",
    8: "Sports Court",
    9: "Soccer Field",
    10: "Commercial Building",
    11: "Residential Building",
    12: "Commercial Building Block",
    13: "House",
    14: "Small House" # Small Construction
}
def gather_data(pan_json, pan_image_dir, image_id):
    # build objects
    objects = dict()
    segmentations = dict()
    ann = find_ann_of_image_id(pan_json, image_id)
    if (ann is None):
        log(f'bad annotation or image id, {image_id}')
        return None
    label_img = cv2.imread(os.path.join(pan_image_dir, ann['file_name']))
    if (label_img is None):
        log(f'bad label image {ann["file_name"]}')
        return None
    b, g, r = label_img.transpose(2, 0, 1)
    ids = r + g * 256 + b * (256 ** 2)
    for seg in ann['segments_info']:
        cat_id = seg['category_id']
        cat_name = class_names[cat_id]
        id = seg['id']
        bbox = seg['bbox']
        if (cat_id >= 4):
            # things
            if (not cat_name in objects):
                objects[cat_name] = []
            objects[cat_name].append({
                'mask': id == ids,
                'bbox': bbox,
            })
        else:
            # stuff
            if (not cat_name in segmentations):
                segmentations[cat_name] = np.zeros(ids.shape, dtype=np.bool8)
            segmentations[cat_name] |= (ids == id)
    
    res = {
        'objects': objects,
        'stuff segmentations': segmentations,
    }
    return res


def gather_data_separate(pan_json, pan_image_dir, image_id):
    ''' get data, but for segmentation and object detection tasks separetely '''
    # build objects
    objects = []
    segmentations = dict()
    ann = find_ann_of_image_id(pan_json, image_id)
    if (ann is None):
        log(f'bad annotation or image id, {image_id}')
        return None
    label_img = cv2.imread(os.path.join(pan_image_dir, ann['file_name']))
    if (label_img is None):
        log(f'bad label image {ann["file_name"]}')
        return None
    b, g, r = label_img.transpose(2, 0, 1)
    ids = r + g * 256 + b * (256 ** 2)
    for seg in ann['segments_info']:
        cat_id = seg['category_id']
        cat_name = class_names[cat_id]
        id = seg['id']
        bbox = seg['bbox']
        if (cat_id >= 4):
            # things
            objects.append({
                'label': cat_name,
                'bbox': bbox,
            })
            # add to mask
            if (not cat_name in segmentations):
                segmentations[cat_name] = np.zeros(ids.shape, dtype=np.bool8)
            segmentations[cat_name] |= (ids == id)
        else:
            # stuff
            if (not cat_name in segmentations):
                segmentations[cat_name] = np.zeros(ids.shape, dtype=np.bool8)
            segmentations[cat_name] |= (ids == id)
    
    res = {
        'det': objects,
        'seg': segmentations,
    }
    return res

if __name__ == '__main__':

    json_dir = 'E:\\LZR\\Storage\\Source\\Dataset\\bsb_dataset\\annotations'

    # with open(os.path.join(json_dir, 'instance_val.json'), 'r') as f:
    #     instance_json = json.load(f)
    with open(os.path.join(json_dir, 'panoptic_val.json'), 'r') as f:
        pan_json = json.load(f)
    print(gather_data(pan_json, os.path.join(json_dir, '..', 'panoptic_val'), 1))


