
import time, sys, os, math, random
import numpy as np

import templates.hugging
import oaapi
import json


def test_planning():
    demo = '''
Can you tell me how many objects in e1.jpg?

[{"task": "object-detection", "id": 0, "dep": [-1], "args": {"image": "e1.jpg" }}]

In e2.jpg, what’s the animal and what’s it doing?

[{"task": "image-to-text", "id": 0, "dep":[-1], "args": {"image": "e2.jpg" }}, {"task":"image-cls", "id": 1, "dep": [-1], "args": {"image": "e2.jpg" }}, {"task":"object-detection", "id": 2, "dep": [-1], "args": {"image": "e2.jpg" }}, {"task": "visual-quesrion-answering", "id": 3, "dep":[-1], "args": {"text": "what’s the animal doing?", "image": "e2.jpg" }}]

First generate a HED image of e3.jpg, then based on the HED image and a text “a girl reading a book”, create a new image as a response.

[{"task": "pose-detection", "id": 0, "dep": [-1], "args": {"image": "e3.jpg" }}, {"task": "pose-text-to-image", "id": 1, "dep": [0], "args": {"text": "a girl reading a book", "image": "<resource>-0" }}]

'''
    task_list = '''
image-to-text: 
image-cls
semantic-segmentation
object-detection
pose-detection
pose-text-to-image
visual-quesrion-answering
'''
    chat_logs = 'None'

    res = templates.hugging.planning(task_list, demo, chat_logs,
        user_input='can you tell me what animal is in input.jpg?')
    print(res)

    ret = oaapi.ask_once(system='You are a helpful assistant.', user_question=res)
    print(ret)

    return ret

def test_generation():
    planning = '[{"task": "object-detection", "id": 0, "dep": [-1], "args": {"image": "e1.jpg" }}]'

    res = templates.hugging.response_gen(
        user_input='Can you tell me how many objects in e1.jpg?', 
        tasks=planning,
        model_assignment='task0: {"id": "yolov5", "reason": "it is well known and performs good on object detection"}',
        predictions='task0: [{"class": "person", "bbox", [100, 121, 34, 50]}, {"class": "car", "bbox", [100, 121, 34, 50]}]')
    
    print(res)

    ret = oaapi.ask_once(system='You are a helpful assistant.', user_question=res)
    print(ret)

    return ret


if __name__ == '__main__':
    responds = []
    for i in range(64):
        responds.append(test_planning())
        with open('res.json', 'w') as f:
            json.dump(responds, f)
    # test_generation()
