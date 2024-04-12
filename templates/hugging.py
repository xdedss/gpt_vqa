
import time, sys, os, math, random
import numpy as np


def planning(
    available_task_list,
    demonstrations,
    chat_logs,
    user_input,
):
    
    format_example = '[{"task": task, "id", task_id, "dep": dependency_task_ids, "args": {"text": text, "image": URL, "audio": URL, "video": URL}}]'
    text = f'''
You are an AI assistant that performs task parsing on user input, generating a list of tasks with the following format: {format_example}. The "dep" field denotes the id of the previous task which generates a new resource upon which the current task relies. The tag "<resource>-task_id" represents the generated text, image, audio, or video from the dependency task with the corresponding task_id. The task must be selected from the following options: 

{ available_task_list }

Please note that there exists a logical connections and order between the tasks. In case the user input cannot be parsed, an empty JSON response should be provided. Here are several cases for your reference: 

{ demonstrations }

To assist with task planning, the chat history is available as follows, where you can trace the user-mentioned resources and incorporate them into the task planning stage.

{ chat_logs }

User input: {user_input}

'''
    return text



def response_gen(
    user_input,
    tasks,
    model_assignment,
    predictions,
):
    text = f'''
With the input and the inference results, you as an AI assistant need to describe the process and results. The previous stages can be formed as:

User Input: { user_input }

Task Planning: 

{ tasks }

Model Selection: 

{ model_assignment }

ask Execution: 

{ predictions }

You must first answer the user’s request in a straightforward manner. Then describe the task process and show your analysis and model inference results to the user in the first person. If inference results contain a file path, must tell the user the complete file path. If there is nothing in the results, please tell me you can’t make it.
'''
    return text




if __name__ == '__main__':
    demo = '''
Can you tell me how many objects in e1.jpg?

[{"task": "object-detection", "id": 0, "dep": [-1], "args": {"image": "e1.jpg" }}]

In e2.jpg, what’s the animal and what’s it doing?

[{"task": "image-to-text", "id": 0, "dep":[-1], "args": {"image": "e2.jpg" }}, {"task":"image-cls", "id": 1, "dep": [-1], "args": {"image": "e2.jpg" }}, {"task":"object-detection", "id": 2, "dep": [-1], "args": {"image": "e2.jpg" }}, {"task": "visual-quesrion-answering", "id": 3, "dep":[-1], "args": {"text": "what’s the animal doing?", "image": "e2.jpg" }}]

First generate a HED image of e3.jpg, then based on the HED image and a text “a girl reading a book”, create a new image as a response.

[{"task": "pose-detection", "id": 0, "dep": [-1], "args": {"image": "e3.jpg" }}, {"task": "pose-text-to-image", "id": 1, "dep": [0], "args": {"text": "a girl reading a book", "image": "<resource>-0" }}]

'''
    task_list = '''
image-to-text
image-cls
semantic-segmentation
object-detection
pose-detection
pose-text-to-image
visual-quesrion-answering
'''
    chat_logs = 'None'

    res = planning(task_list, demo, chat_logs, "user input")
    print(res)

    


