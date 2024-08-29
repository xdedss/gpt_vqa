
from yacs.config import CfgNode as CN
import simple_agent
from simple_agent import SimpleAgent
from simple_agent_feedback import SimpleAgent as SimpleAgentFeedback
from agents import AgentBase
from agents.environment import Tool, ToolError, Resource
from agents.environment.tools import ImageMetaTool, PythonTool, DetectionCounting, MaskArea, MaskCount, MaskPathFinding
from agents.environment.resources import ImageResource, JsonResource, MasksResource
from agents.parsers.json import LastJsonParser
import llm_metrics

import cv2
import tqdm

import os, json, random, uuid, time, base64
import logging

import requests
import urllib.parse

import database_sqlite
import llm_utils

logger = logging.getLogger()


def send_visualglm_api(image_path, question) -> str:
    # Encode the image in base64
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    # Prepare the JSON payload
    payload = {
        "image": encoded_image,
        "text": question,
        "history": []
    }
    
    data = json.dumps(payload)
    
    # Send the request to the server
    url = "http://127.0.0.1:8080"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=data)
    
    # Parse the response and return the "result"
    response_data = response.json()
    return response_data.get("result", ""), []

def get_answer_call_api(api_url, image_path, text) -> str:
    # Create a dictionary with the files to be sent
    files = {'image': open(image_path, 'rb')}
    
    # Provide the text as a separate parameter
    data = {'question': text}
    
    # Make the POST request to the API endpoint
    response = requests.post(api_url+f'?question={urllib.parse.quote(text)}', files=files, json=data)
    
    # Return the response content as a string
    return response.text, []


def get_answer(question, label_path, feedback=False, need_confirm=False):
    ''' returns ans, action_history '''

    label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    
    class_casual_names = [
        'water',
        'building without damage',
        'building with minor damage',
        'building with major damage',
        'building with total destruction',
        'vehicle',
        'clear road',
        'blocked road',
        'tree',
        'pool',
    ]
    # fix class name mismatch, this should cause no obvious problem

    mask_dict = dict()

    for i in range(len(class_casual_names)):
        binary_mask = label_img == (i + 1)
        class_name = class_casual_names[i]
        mask_dict[class_name] = binary_mask

    if (feedback):
        a = SimpleAgentFeedback(CN())
    else:
        a = SimpleAgent(CN())
    a.add_tool(
        'semantic_segmentation', 
        ImageMetaTool('seg', f'this tool takes an image, performs semantic segmentation, and returns masks with following labels: {json.dumps(class_casual_names)}', output_type='masks'))
    a.add_tool(
        'mask_area_calculation',
        MaskArea())
    a.add_tool(
        'mask_count',
        MaskCount())
    a.add_tool(
        'mask_path_finding',
        MaskPathFinding())
    
    # a.add_tool('python', PythonTool())
    a.add_resource('input', ImageResource(None, meta={
        'seg': MasksResource(mask_dict),
    }))
    a.require_confirm = need_confirm
    request = question
    if (feedback):
        res = a.chat_feedback(request)
    else:
        res = a.chat(request)
    return res, a.action_history_array

def count_det_label(data, label):
    count = 0
    for item in data:
        if item["label"] == label:
            count += 1
    return count

def evaluate(question, label_path, answer_gt, label_type, feedback=False, need_confirm=False):

    logger.info(f'QUESTION:\n{question}\nANSWER GT:\n{answer_gt}')
    if (need_confirm):
        input('continue?')

    # call the agent
    # answer, action_history = get_answer(question, label_path, feedback, need_confirm)

    answer, action_history = send_visualglm_api(
        # 'http://127.0.0.1:8000/inference', 
        label_path,
        question)
    logger.info("visualglm answer")
    logger.info(answer)

    if (label_type == 'seg'):
        # the agent must call segmentation
        has_seg = False
        for action in action_history:
            if (action.action['id'] == 'semantic_segmentation'):
                has_seg = True
                break
        correctness = has_seg
    else:
        # now use gpt metric
        correctness = llm_utils.retry_until_succeed(
            lambda: llm_metrics.compare_question_answer_groundtruth(question, answer, answer_gt)
        )

    return answer, correctness, action_history


def evaluate_all(dataset_jsonl: str, *, start_index=0, end_index=None, db_path='default_rescuenet.db', feedback=False, need_confirm=False):
    
    with open(dataset_jsonl, 'r') as f:
        valset_objects = [json.loads(s) for s in f.readlines() if s.strip() != '']
    
    import ray
    
    ray.init(num_cpus=4)

    @ray.remote(num_cpus=1)
    def process_valset_item(valset_obj):
        label_path = valset_obj['label']
        question = valset_obj['instruction']
        gt_answer = valset_obj['answer']
        label_type = valset_obj['type']

        image_id = 0 # unused

        meta_data = dict()

        meta_data['label_type'] = label_type
        meta_data['label_path'] = label_path
          
        try:
            answer, correctness, action_history = evaluate(question, label_path, gt_answer, label_type, feedback=feedback, need_confirm=need_confirm)
            meta_data['action_history'] = [
                {'action': action.action, 'action_result': action.action_result} for action in action_history
            ]
            db = database_sqlite.Database(db_path)
            if (correctness):
                db.add_data(image_id, question, answer, gt_answer, 'correct', json.dumps(meta_data, ensure_ascii=False))
            else:
                db.add_data(image_id, question, answer, gt_answer, 'incorrect', json.dumps(meta_data, ensure_ascii=False))
            db.close()
        except NameError:
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            meta_data['error'] = str(e)
            logger.error(e)
            db = database_sqlite.Database(db_path)
            db.add_data(image_id, question, '', gt_answer, 'error', json.dumps(meta_data, ensure_ascii=False))
            db.close()

    if (end_index is None):
        end_index = len(valset_objects)
    
    tasks = []
    for valset_obj in valset_objects[start_index:end_index]:
        tasks.append(process_valset_item.remote(valset_obj))

    with tqdm.tqdm(total=len(tasks)) as pbar:
        while tasks:
            done, tasks = ray.wait(tasks, num_returns=1)
            pbar.update(len(done))
            time.sleep(0.1)


if __name__ == '__main__':

    feedback = False
    llm_utils.setup_root_logger(
        filename='simple_agent_rescuenet_visualglm_600.log', 
        level=logging.INFO)

    # selected_id = [23]
    # with open('rescuenet_agent_val_tiny.jsonl', 'r') as f:
    #     valset_objects = [json.loads(s) for s in f.readlines() if s.strip() != '']

    # for id in selected_id:
    #     obj = valset_objects[id]

    #     label_path = obj['label']
    #     question = obj['instruction']
    #     gt_answer = obj['answer']
    #     label_type = obj['type']
    #     logger.info('-------- label info ---------')
    #     logger.info(f'path: {label_path}')
    #     logger.info(f'question: {question}')
    #     logger.info(f'answer: {gt_answer}')
    #     logger.info(f'label_type: {label_type}')
    #     logger.info('-------- label info ---------')

    #     res, eval_correctness, action_history = evaluate(question, label_path, gt_answer, label_type, feedback, need_confirm=True)

    #     # res = get_answer(question, label_path, feedback, need_confirm=True)
    #     logger.info('=================================')
    #     logger.info(res)
    #     logger.info(f'CORRECT: {eval_correctness}')


    # res = send_visualglm_api(
    #     'D:/LZR/Downloads/documents/RescuNet/val-org-img\\10842.jpg',
    #     'Are there any water visible in this image?',
    # )
    # print(res)
    # xx

    evaluate_all(
        'rescuenet_with_corrected_label/rescuenet_agent_val_small_960.jsonl',
        start_index=0,
        end_index=None,
        feedback=feedback, 
        db_path='simple_agent_rescuenet_valset_small_960_visualglm_tuned600.db')




