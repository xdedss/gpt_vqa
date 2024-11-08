
from yacs.config import CfgNode as CN
import simple_agent
from simple_agent import SimpleAgent
from simple_agent_tuned import SimpleAgent as SimpleAgentTuned
from simple_agent_feedback import SimpleAgent as SimpleAgentFeedback
from agents import AgentBase
from agents.environment import Tool, ToolError, Resource
from agents.environment.tools import ImageMetaTool, PythonTool, DetectionCounting, MaskArea, MaskCount, MaskPathFinding
from agents.environment.resources import ImageResource, JsonResource, MasksResource
from agents.parsers.json import LastJsonParser
import llm_metrics

import cv2
import tqdm
    
import ray

import os, json, random, uuid, time, base64
import logging

import requests
import urllib.parse

import database_sqlite
import llm_utils

logger = logging.getLogger()


def convert_to_org_image_path(input_path):
    # Parse the path
    directory, file_name = os.path.split(input_path)
    parent_dir, label_folder = os.path.split(directory)
    
    # Ensure the label folder matches the '-label-img' structure
    if "label-img" in label_folder:
        yyy = label_folder.split("-label-img")[0] # val/train
        
        # Form the new directory path and file path for output
        src_dir = os.path.join(parent_dir, f"{yyy}-org-img")
        src_fname = file_name.replace("_lab.png", ".jpg")
        full_path = os.path.join(src_dir, src_fname)
        return full_path
    else:
        print("Input path does not match the expected structure.")
        raise ValueError(input_path)

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
    url = "http://172.17.135.64:8081"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=data)
    
    # Parse the response and return the "result"
    response_data = response.json()
    return response_data.get("result", ""), []

def send_geochat_api(image_path, question):
    # Read and encode the image in base64
    with open(image_path, "rb") as image_file:
        data = image_file.read()
        encoded_image = base64.b64encode(data).decode("utf-8")
    print("image_data:", data[:100], len(data))
    
    # Prepare the JSON payload
    payload = {
        "image": encoded_image,
        "text": question,
    }
    
    data = json.dumps(payload)
    
    # Send the request to the server
    url = "http://172.17.135.64:8080"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=data)
    
    # Parse the response and return the "result"
    response_data = response.json()
    return response_data.get("result", "invalid"), []

def get_answer_call_api(api_url, image_path, text) -> str:
    # Create a dictionary with the files to be sent
    files = {'image': open(image_path, 'rb')}
    
    # Provide the text as a separate parameter
    data = {'question': text}
    
    # Make the POST request to the API endpoint
    response = requests.post(api_url+f'?question={urllib.parse.quote(text)}', files=files, json=data)
    
    # Return the response content as a string
    return response.text, []


def get_answer(question, label_path, det_label_path, feedback=False, need_confirm=False, model_name='gpt-3.5-turbo', disable_tool_desc=False, disable_aux_tool=False):
    ''' returns ans, action_history '''

    label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    with open(det_label_path, 'r') as f:
        det_label = json.load(f)
    
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
    
    obj_casual_names = [
        'building without damage',
        'building with minor damage',
        'building with major damage',
        'building with total destruction',
        'vehicle',
        'pool',
    ]

    mask_dict = dict()

    for i in range(len(class_casual_names)):
        binary_mask = label_img == (i + 1)
        class_name = class_casual_names[i]
        mask_dict[class_name] = binary_mask

    for object in det_label:
        # convert to casual names
        # type counts from 0=bg 1=water, ...
        object['label'] = class_casual_names[object['type'] - 1]

    agent_cfg = CN()
    agent_cfg.model_name = model_name
    agent_cfg.tool_desc = not disable_tool_desc
    # agent_cfg.model_name = "meta-llama/Llama-3.1-8B-Instruct"# model_name
    # agent_cfg.base_url = "http://172.17.135.64:8000/v1"
    # agent_cfg.api_key = "aa"
    if (feedback):
        a = SimpleAgentFeedback(agent_cfg)
    else:
        a = SimpleAgent(agent_cfg)

        agent_cfg.api_key = "aa"
        agent_cfg.base_url = 'http://localhost:8001/v1'
        a = SimpleAgentTuned(agent_cfg)
    a.add_tool(
        'semantic_segmentation', 
        ImageMetaTool(
            'seg', 
            f'this tool takes an image, performs semantic segmentation, and returns masks with following labels: {json.dumps(class_casual_names)}', output_type='masks'))
    a.add_tool(
        'object_detection', 
        ImageMetaTool(
            'det', 
            f'this tool takes an image, performs object detection, and returns an array of objects, possibly of the following types: {json.dumps(obj_casual_names)}', output_type='json'))
    
    if (not disable_aux_tool):
        a.add_tool(
            'mask_area_calculation',
            MaskArea(sqmeter_per_px=0.02**2))
        # a.add_tool(
        #     'mask_count',
        #     MaskCount())
        a.add_tool(
            'object_count',
            DetectionCounting(),
        )
        a.add_tool(
            'mask_path_finding',
            MaskPathFinding())
    
    # a.add_tool('python', PythonTool())
    a.add_resource('input', ImageResource(None, meta={
        'seg': MasksResource(mask_dict),
        'det': JsonResource(det_label)
    }))
    a.require_confirm = need_confirm
    request = question
    if (feedback):
        res = a.chat_feedback(request)
    else:
        res = a.chat(request)
    return res, a.action_history_array, a.history

def count_det_label(data, label):
    count = 0
    for item in data:
        if item["label"] == label:
            count += 1
    return count

def evaluate(question, label_path, det_label_path, answer_gt, gt_plan, label_type, feedback=False, need_confirm=False, model_name='gpt-3.5-turbo', disable_tool_desc=False, disable_aux_tool=False):

    logger.info(f'QUESTION:\n{question}\nANSWER GT:\n{answer_gt}')
    if (need_confirm):
        input('continue?')

    action_history = []
    chat_history = dict()
    # # call the agent
    # answer, action_history, chat_history = get_answer(
    #     question, 
    #     label_path, 
    #     det_label_path, 
    #     feedback, 
    #     need_confirm, 
    #     model_name=model_name, 
    #     disable_tool_desc=disable_tool_desc,
    #     disable_aux_tool=disable_aux_tool,
    #     )

    # answer, action_history = send_visualglm_api(
    #     # 'http://127.0.0.1:8000/inference', 
    #     label_path,
    #     question)
    # logger.info("visualglm answer")

    # geochat
    # answer, action_history = send_geochat_api(
    #     label_path,
    #     question)
    # logger.info("geochat answer")

    # raw vision api
    import otherapi, oaapi
    answer = otherapi.send_claude(
        'You are a helpful assistant, answer questions with yes/no or a number.',
        question,
        convert_to_org_image_path(label_path)
    )


    logger.info(answer)

    # check what get called
    has_seg = False
    has_det = False
    for action in action_history:
        if (action.action['id'] == 'semantic_segmentation'):
            has_seg = True
            break
    for action in action_history:
        if (action.action['id'] == 'object_detection'):
            has_det = True
            break
    plan_criteria = {
        'det': has_det,
        'seg': has_seg,
        'seg_or_det': has_det or has_seg,
    }
    plan_correct = True
    for plan_item in gt_plan:
        plan_correct = plan_correct and plan_criteria[plan_item]


    if (label_type in ['seg', 'det']):
        # the agent must call segmentation
        ans_correct = "invalid"
    else:
        # now use gpt metric
        ans_correct = llm_utils.retry_until_succeed(
            lambda: llm_metrics.compare_question_answer_groundtruth(question, answer, answer_gt)
        )
    
    correctness = {
        'ans': ans_correct,
        'plan': plan_correct,
    }

    return answer, correctness, action_history, chat_history


def evaluate_all(dataset_jsonl: str, *, start_index=0, end_index=None, db_path='default_rescuenet.db', feedback=False, need_confirm=False, model_name='gpt-3.5-turbo', disable_tool_desc=False, disable_aux_tool=False):
    
    with open(dataset_jsonl, 'r') as f:
        valset_objects = [json.loads(s) for s in f.readlines() if s.strip() != '']

    # @ray.remote(num_cpus=1)
    def process_valset_item(valset_obj):
        label_path = valset_obj['label']
        det_label_path = valset_obj['det_label']
        question = valset_obj['instruction']
        gt_answer = valset_obj['answer']
        label_type = valset_obj['type']
        gt_plan = valset_obj['plan']

        image_id = 0 # unused

        meta_data = dict()

        meta_data['label_type'] = label_type
        meta_data['label_path'] = label_path
          
        try:
            answer, correctness, action_history, chat_history = evaluate(
                question, 
                label_path, 
                det_label_path, 
                gt_answer, 
                gt_plan, 
                label_type, 
                feedback=feedback, 
                need_confirm=need_confirm,
                model_name=model_name,
                disable_tool_desc=disable_tool_desc,
                disable_aux_tool=disable_aux_tool,
                )
            meta_data['action_history'] = [
                {'action': action.action, 'action_result': action.action_result} for action in action_history
            ]
            meta_data['chat_history'] = chat_history
            db = database_sqlite.Database(db_path)
            db.add_data(
                image_id, question, answer, gt_answer, 
                json.dumps(correctness, ensure_ascii=False), 
                json.dumps(meta_data, ensure_ascii=False))
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
        process_valset_item(valset_obj)
        # tasks.append(process_valset_item.remote(valset_obj))

    # with tqdm.tqdm(total=len(tasks)) as pbar:
    #     while tasks:
    #         done, tasks = ray.wait(tasks, num_returns=1)
    #         pbar.update(len(done))
    #         time.sleep(0.1)


def free_test():
    import sys
    sys.path.append('./')
    import llm_utils
    llm_utils.setup_root_logger(
        filename='free_test.log', 
        level=logging.WARNING)

    # selected_id = [84]
    selected_id = [53]
    # question_novel = 'How many buildings are there and what level of damage do they have?'
    question_novel = 'What is the average area of destroyed houses?'
    with open('rescuenet_regen_plus_det/rescuenet_agent_val_1k_real.jsonl', 'r') as f:
        valset_objects = [json.loads(s) for s in f.readlines() if s.strip() != '']

    for id in selected_id:
        obj = valset_objects[id]

        label_path = obj['label']
        det_label_path = obj['det_label']
        question = obj['instruction']
        gt_answer = obj['answer']
        label_type = obj['type']
        logger.info('-------- label info ---------')
        logger.info(f'path: {label_path}')
        logger.info(f'question: {question}')
        logger.info(f'answer: {gt_answer}')
        logger.info(f'label_type: {label_type}')
        logger.info('-------- label info ---------')

        res = get_answer(question_novel, label_path, det_label_path, model_name='gpt-4o-mini', need_confirm=False)
        print(res)


if __name__ == '__main__':

    feedback = False
    
    def logger_setup():
        import sys
        sys.path.append('./')
        import llm_utils
        llm_utils.setup_root_logger(
            filename='fff.log', 
            level=logging.WARNING)
    
    logger_setup()

    ray.init(num_cpus=4, runtime_env={"worker_process_setup_hook": logger_setup})

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
        'rescuenet_regen_plus_det/rescuenet_agent_val_1k_real_tier2.jsonl',
        start_index=0,
        end_index=None,
        feedback=feedback, 
        db_path='with_det_1k_real_tier2.db')




