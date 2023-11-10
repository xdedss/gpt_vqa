
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

import os, json, random, uuid
import logging

import requests
import urllib.parse

import bsbutil
import database_sqlite
import llm_utils

logger = logging.getLogger()

def get_answer_call_api(api_url, image_path, text) -> str:
    # Create a dictionary with the files to be sent
    files = {'image': open(image_path, 'rb')}
    
    # Provide the text as a separate parameter
    data = {'question': text}
    
    # Make the POST request to the API endpoint
    response = requests.post(api_url+f'?question={urllib.parse.quote(text)}', files=files, json=data)
    
    # Return the response content as a string
    return response.text


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

    logger.info(f'QUESTION:\n{question}\nANSWER GT:\n{answer_gt}')
    if (need_confirm):
        input('continue?')

    # call the agent
    # answer = get_answer(question, image_id, feedback, need_confirm)
    answer = get_answer_call_api(
        'http://127.0.0.1:8000/inference', 
        os.path.join('E:\\LZR\\Storage\\Source\\Dataset\\bsb_dataset\\image_val_png', f'{image_id}.png'),
        question)
    logger.info("visualglm answer")
    logger.info(answer)
    # import sys
    # sys.exit(0)

    # now use gpt metric
    correctness = llm_utils.retry_until_succeed(
        lambda: llm_metrics.compare_question_answer_groundtruth(question, answer, answer_gt)
    )

    return answer, correctness


def evaluate_all(*, start_index=0, end_index=None, db_path='default.db', split='val', feedback=False, need_confirm=False):
    db = database_sqlite.Database(db_path)
    json_dir = 'E:\\LZR\\Storage\\Source\\Dataset\\bsb_dataset\\annotations'
    label_dir = os.path.join(json_dir, '..', f'panoptic_{split}')
    with open(os.path.join(json_dir, f'panoptic_{split}.json'), 'r') as f:
        pan_json = json.load(f)

    if (end_index is None):
        end_index = len(pan_json['images'])
    for image in pan_json['images'][start_index:end_index]:
        image_id = image['id']
    
        dataset_gt = bsbutil.gather_data_separate(pan_json, label_dir, image_id)
        
        unique_obj_labels = list(set([o['label'] for o in dataset_gt['det']]))
        unique_seg_labels = list(dataset_gt['seg'].keys())

        for label in unique_obj_labels:
            
            meta_data = dict()
            uid = str(uuid.uuid4())
            meta_data['uuid'] = uid # for comparison with 
            logger.info(f'uuid: {uid}')

            logger.info((image_id, label))

            question = f'How many {label} are there in the image?'
            answer_gt = count_det_label(dataset_gt['det'], label)
            
            try:
                answer, correctness = evaluate(image_id, question, answer_gt, feedback, need_confirm)
                if (correctness):
                    db.add_data(image_id, question, answer, answer_gt, 'correct', json.dumps(meta_data, ensure_ascii=False))
                else:
                    db.add_data(image_id, question, answer, answer_gt, 'incorrect', json.dumps(meta_data, ensure_ascii=False))
            except Exception as e:
                import traceback
                traceback.print_exc()
                meta_data['error'] = str(e)
                logger.error(e)
                db.add_data(image_id, question, '', answer_gt, 'error', json.dumps(meta_data, ensure_ascii=False))
            
    db.close()


if __name__ == '__main__':
    image_id = 50
    feedback = False

    llm_utils.setup_root_logger(
        filename='simple_agent_bsb.log', 
        level=logging.INFO)

    # res = get_answer('How many houses are there in the image?', image_id, feedback)
    # print('=================================')
    # print(res)

    # correctness = evaluate(image_id, feedback, need_confirm=False)
    # print('=================================')
    # print(correctness)

    evaluate_all(
        start_index=50,
        end_index=51,
        feedback=feedback, 
        db_path='simple_agent_bsb.db')




