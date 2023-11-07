
from yacs.config import CfgNode as CN
import simple_agent
from simple_agent import SimpleAgent
from simple_agent_feedback import SimpleAgent as SimpleAgentFeedback
from agents import AgentBase
from agents.environment import Tool, ToolError, Resource
from agents.environment.tools import ImageMetaTool, PythonTool
from agents.environment.resources import ImageResource, JsonResource, MasksResource
from agents.parsers.json import LastJsonParser
import llm_metrics

import os, json, random

import bsbutil
import database_sqlite

def get_answer(question, image_id, feedback=False, need_confirm=False):
    json_dir = 'E:\\LZR\\Storage\\Source\\Dataset\\bsb_dataset\\annotations'
    with open(os.path.join(json_dir, 'panoptic_val.json'), 'r') as f:
        pan_json = json.load(f)
    dataset_gt = bsbutil.gather_data_separate(pan_json, os.path.join(json_dir, '..', 'panoptic_val'), image_id)
    
    unique_obj_labels = list(set([o['label'] for o in dataset_gt['det']]))
    unique_seg_labels = list(dataset_gt['seg'].keys())

    if (feedback):
        a = SimpleAgentFeedback(CN())
    else:
        a = SimpleAgent(CN())
    a.add_tool('semantic_segmentation', ImageMetaTool('seg', f'this tool takes an image, performs semantic segmentation, and returns masks with following labels: {json.dumps(unique_seg_labels)}', output_type='masks'))
    a.add_tool('object_detection', ImageMetaTool('det', f'this tool takes an image, performs object detection, and returns object bounding boxes in xywh format. The categories of objects that will be detected are {json.dumps(unique_obj_labels)}'))
    # a.add_tool('python', PythonTool())
    a.add_resource('input', ImageResource(None, meta={
        'seg': MasksResource(dataset_gt['seg']),
        'det': JsonResource(dataset_gt['det']),
    }))
    a.require_confirm = need_confirm
    request = question
    if (feedback):
        res = a.chat_feedback(request)
    else:
        res = a.chat(request)
    return res

def count_det_label(data, label):
    count = 0
    for item in data:
        if item["label"] == label:
            count += 1
    return count

def evaluate(image_id, question, answer_gt, feedback=False, need_confirm=False):

    print(f'QUESTION:\n{question}\nANSWER GT:\n{answer_gt}')
    if (need_confirm):
        input('continue?')

    # call the agent
    answer = get_answer(question, image_id, feedback, need_confirm)

    # now use gpt metric
    correctness = llm_metrics.compare_question_answer_groundtruth(question, answer, answer_gt)

    return answer, correctness


def evaluate_all(split='val', feedback=False, need_confirm=False):
    db = database_sqlite.Database('simple_agent_bsb.db')
    json_dir = 'E:\\LZR\\Storage\\Source\\Dataset\\bsb_dataset\\annotations'
    label_dir = os.path.join(json_dir, '..', f'panoptic_{split}')
    with open(os.path.join(json_dir, 'panoptic_val.json'), 'r') as f:
        pan_json = json.load(f)

    for image in pan_json['images'][0:50]:
        image_id = image['id']
    
        dataset_gt = bsbutil.gather_data_separate(pan_json, label_dir, image_id)
        
        unique_obj_labels = list(set([o['label'] for o in dataset_gt['det']]))
        unique_seg_labels = list(dataset_gt['seg'].keys())

        for label in unique_obj_labels:
            print(image_id, label)
            

            question = f'How many {label} are there in the image?'
            answer_gt = count_det_label(dataset_gt['det'], label)
            
            try:
                answer, correctness = evaluate(image_id, question, answer_gt, feedback, need_confirm)
                if (correctness):
                    db.add_data(image_id, question, answer, answer_gt, 'correct', '')
                else:
                    db.add_data(image_id, question, answer, answer_gt, 'incorrect', '')
            except Exception as e:
                db.add_data(image_id, question, '', answer_gt, 'error', str(e))
            
    db.close()


if __name__ == '__main__':
    image_id = 50
    feedback = False

    # res = get_answer('How many houses are there in the image?', image_id, feedback)
    # print('=================================')
    # print(res)

    # correctness = evaluate(image_id, feedback, need_confirm=False)
    # print('=================================')
    # print(correctness)

    evaluate_all()




