
import sys
import os
import io
import json
import random
import glob
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import cv2

import tqdm
from PIL import Image
import numpy as np
from scipy.ndimage import label


# for reference
# class_casual_names = [
#     'water', 1
#     'building without damage', 2
#     'building with minor damage', 3
#     'building with major damage', 4
#     'building with total destruction', 5
#     'vehicle', 6 # note: class names on github is reversed
#     'clear road', 7
#     'blocked road', 8
#     'tree', 9
#     'pool', 10
# ]
OBJECT_CLASS_IDS = [
    2, 3, 4, 5, 6, 10
]

def make_obj_det(segmentation_map, output_json_path):
    res = []
    for class_id in OBJECT_CLASS_IDS:
        # only convert those foreground objects
        instance_id_map, num_instances = label(segmentation_map == class_id)
        for instance_id in range(1, num_instances + 1):
            obj_mask = instance_id_map == instance_id
            ys, xs = (np.where(obj_mask))
            ymax = np.max(ys)
            ymin = np.min(ys)
            xmax = np.max(xs)
            xmin = np.min(xs)
            # in x, y, w, h
            x = int(xmin)
            y = int(ymin)
            w = int(xmax - xmin)
            h = int(ymax - ymin)
            if (w < 10 and h < 10):
                continue # too small
            res.append({
                'bbox': [x, y, w, h],
                'type': class_id,
            })
    with open(output_json_path, 'w') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)


def process_one_image(args):
    input_path, output_dir = args
    raw_fname = os.path.splitext(os.path.basename(input_path))[0]
    # label_img = np.array(Image.open(input_path))
    label_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    output_path = os.path.join(output_dir, f'{raw_fname}.json')
    make_obj_det(label_img, output_path)

def subproc_initializer():
    # this is to make sure that ctrl-c terminates gracefully
    import signal
    signal.signal(signal.SIGINT, lambda: None)

def make_directory(input_dir, output_dir):
    # convert every *_lab.png to *.json
    os.makedirs(output_dir, exist_ok=True)

    task_args = []
    for input_path in glob.glob(os.path.join(input_dir, '*_lab.png')):
        task_args.append((input_path, output_dir))
        
    with multiprocessing.Pool(8, subproc_initializer) as pool:
        try:
            list( # list() forces the iterator to be fully evaluated
                tqdm.tqdm(
                    pool.imap(process_one_image, task_args),
                    total=len(task_args),
                )
            )
        except KeyboardInterrupt as e:
            print('keyboard interrupt, breaking current progress')
            pass


if __name__ == '__main__':
    # label_img = np.array(Image.open(r'D:\LZR\Downloads\documents\RescuNet\test-label-img\10844_lab.png'))
    # make_obj_det(label_img, 'test_gen_det.json')

    make_directory("D:/LZR/Downloads/documents/RescuNet/val-label-img", "D:/LZR/Downloads/documents/RescuNet/val-label-det")
    make_directory("D:/LZR/Downloads/documents/RescuNet/train-label-img", "D:/LZR/Downloads/documents/RescuNet/train-label-det")
    pass
