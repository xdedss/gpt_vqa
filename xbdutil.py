
import json
import os
import time

import cv2
import numpy as np

import logging

logger = logging.getLogger(__name__)

def log(message):
    logger.info(message)


class_names = {
    0: "Background",
    1: "Undamaged buildings",
    2: "Slightly damaged buildings",
    3: "Severely damaged buildings",
    4: "Completely destroyed buildings",
}
def get_label_dicts(pre_disaster_label, post_disaster_label):
    ''' get data, xbd '''

    assert os.path.isfile(pre_disaster_label)
    assert os.path.isfile(post_disaster_label)
    label_img_pre = cv2.imread(pre_disaster_label, cv2.IMREAD_GRAYSCALE)
    label_img_post = cv2.imread(post_disaster_label, cv2.IMREAD_GRAYSCALE)
    print(label_img_pre.sum(), label_img_post.sum())
    
    def label_map_to_dict(label_img):
        segmentations = dict()
        
        for label_i in np.unique(label_img):
            label_class_name = class_names[label_i]
            segmentations[label_class_name] = (label_img == label_i).astype(np.bool8)
        return segmentations
    
    res = {
        'post_seg': label_map_to_dict(label_img_post),
        'pre_seg': label_map_to_dict(label_img_pre),
    }
    return res

if __name__ == '__main__':

    label_dir = 'D:\\LZR\\MyFiles\\xbd\\test\\targets'

    data = get_label_dicts(
        os.path.join(label_dir, 'guatemala-volcano_00000005_pre_disaster_target.png'),
        os.path.join(label_dir, 'guatemala-volcano_00000005_post_disaster_target.png'),

    )

    print(data)

