
from yacs.config import CfgNode as CN
import simple_agent
from simple_agent import SimpleAgent    
from agents import AgentBase
from agents.environment import Tool, ToolError, Resource
from agents.environment.tools import ImageMetaTool, PythonTool
from agents.environment.resources import ImageResource, JsonResource, MasksResource
from agents.parsers.json import LastJsonParser

import os, json

import bsbutil

if __name__ == '__main__':
    image_id = 1

    json_dir = 'E:\\LZR\\Storage\\Source\\Dataset\\bsb_dataset\\annotations'
    with open(os.path.join(json_dir, 'panoptic_val.json'), 'r') as f:
        pan_json = json.load(f)
    dataset_gt = bsbutil.gather_data_separate(pan_json, os.path.join(json_dir, '..', 'panoptic_val'), image_id)
    
    unique_obj_labels = list(set([o['label'] for o in dataset_gt['det']]))
    unique_seg_labels = list(dataset_gt['seg'].keys())

    a = SimpleAgent(CN())
    a.add_tool('semantic_segmentation', ImageMetaTool('seg', f'this tool takes an image, performs semantic segmentation, and returns masks with following labels: {json.dumps(unique_seg_labels)}', output_type='masks'))
    a.add_tool('object_detection', ImageMetaTool('det', f'this tool takes an image, performs object detection, and returns object bounding boxes in xywh format. The categories of objects that will be detected are {json.dumps(unique_obj_labels)}'))
    # a.add_tool('python', PythonTool())
    a.add_resource('input', ImageResource(None, meta={
        'seg': MasksResource(dataset_gt['seg']),
        'det': JsonResource(dataset_gt['det']),
    }))
    a.require_confirm = True
    a.chat('How many houses are there in the image?')


