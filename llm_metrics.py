
import oaapi
from agents.parsers.json import LastJsonParser
import datetime
import logging

logger = logging.getLogger(__name__)

def log(message):
    logger.info(message)
    # now = datetime.datetime.now()
    # print(f'[{now.strftime("%Y%m%d-%H%M%S")}][INFO] {message}')

# gt should be categorical, check if equal
def compare_question_answer_groundtruth(user_question, response, gt, need_confirm=False) -> bool:
    
    json_str = '{"result": true/false}'
    prompt = f'''
You are an AI assistant that assess the quality of response produced by an autonomous system in respond to given user request. The system should answer any question that the user asks at the end of the response. 

The user request is:

```
{user_question}
```

The system response to be assessed is:

```
{response}
```

The gound truth of the question is: {gt}

Compare the answer in the response to the ground truth. You are going to assess the following aspect of the answer: Does the response align with the ground truth?

Here are guidelines:
1. If the question is about counting and the answer denies the existence of the required object, we consider it equivalent to answering with number 0.
2. For questions about counting, the quantity in the answer must be exactly equal to the ground truth.

You should first explain your thought and your assessment. Finally you should summarize your assessment with strict json format: {json_str}
'''.strip()
    
    log('LLM metrics input')
    log(prompt)
    res = oaapi.ask_once('You are a helpful assistant.', prompt)
    log('LLM metrics output')
    log(res)

    json_obj = LastJsonParser().parse(res)
    return json_obj['result']

# gt should be a number
def compare_question_answer_groundtruth_numerical(user_question, response, gt, need_confirm=False):
    json_str = '{"result": true/false, "numerical_value_in_response": ...}'
    prompt = f'''
You are an AI assistant that assess the quality of response produced by an autonomous system in respond to given user request. The system should answer any question that the user asks at the end of the response. 

The user request is:

```
{user_question}
```

The system response to be assessed is:

```
{response}
```

The gound truth of the question is: {gt}

Firstly, you should extract the numerical answer. Then you will compare the answer in the response to the ground truth. You are going to assess the following aspect of the answer: Does the response align with the ground truth?

Here are guidelines:
1. If the question is about counting and the answer denies the existence of the required object, we consider it equivalent to answering with number 0.

You should first explain your thought and your assessment. Finally you should summarize your assessment with strict json format: {json_str}
'''.strip()
    
    log('LLM metrics input')
    log(prompt)
    res = oaapi.ask_once('You are a helpful assistant.', prompt)
    log('LLM metrics output')
    log(res)

    json_obj = LastJsonParser().parse(res)
    return json_obj


# gt should be a number
def extract_numerical_answer_fewshot(user_question, response, need_confirm=False):
    json_str = '{"result": true/false, "numerical_value_in_response": ...}'
    prompt = '''
You are an AI assistant that analyze the responses produced by an autonomous system in respond to given user request. The system should answer any question that the user asks at the end of the response. And you will extract the numerical answer in the answer, and convert it to a strict json format.

Extract answers:

Question: How many persons are there?
Answer: There are three persons in two cars.
Reasoning: The question asks about the number of cars, therefore the relevent number that answers the question would be 3.
Extraction: {"value": 3}

Question: How many houses are in the picture?
Answer: There are no house in the picture.
Reasoning: The answer states that there are no house, so the numerical answer to the question is 0.
Extraction: {"value": 0}

Question: What is 2 * 8?
Answer: 2 times 8 equals 21.
Reasoning: The answer that the system gives is 21. Although it is wrong, we should extract the number as is.
Extraction: {"value": 21}

Question: It is a good day.
Answer: Hello, how are you? I see 3 people in the image.
Reasoning: The question does not ask for any numerical value, so there is no relevant answer.
Extraction: {"value": null}

Question: How many tanks are there?
Answer: I can not count the number of tanks.
Reasoning: The answer fails to solve the question, so there is no answer.
Extraction: {"value": null}

'''.strip() + f'''

Question: {user_question}
Answer: {response}
Reasoning: 
'''
    
    log('LLM metrics input')
    log(prompt)
    res = oaapi.ask_once('You are a helpful assistant.', prompt)
    log('LLM metrics output')
    log(res)

    # prevent multiple continuations
    res = res.split('Question:')[0]

    json_obj = LastJsonParser().parse(res)
    return json_obj



# no gt is provided, check logical connection between q and a
def compare_question_answer(user_question, response, need_confirm=False):
    json_str = '{"result": true/false}'
    prompt = f'''
You are an AI assistant that assess the quality of responseproduced by an autonomous system in respond to given user question. The system should answer any question that the user asks at the end of the response. 

The user question is:

```
{user_question}
```

The answer to be assessed is:

```
{response}
```

Firstly, you should find the answer to user question among the response text. You are going to assess the following aspect of the answer: Does the response contain a valid answer to the user question?


Here are guidelines:
1. The answer is considered valid as long as it directly or indirectly answers the question.
2. The answer is considered invalid if it is irrelevant to the question.

Here are good examples that resolves the user's question:
User: How many people are there in the image? 
Response: There are no people in the image.
User: What is the number of people in the image?
Response: The number of people is 0.

Here are bad examples that does not resolve the user's question:
User: How many people are there in the image?
Response: There are 3 cars in the image.
User: What is the number of people in the image?
Response: There are no people in the image.

You should first explain your thought and your assessment. Finally you should summarize your assessment with strict json format: {json_str}
'''.strip()
    
    log('LLM metrics input')
    log(prompt)
    res = oaapi.ask_once('You are a helpful assistant.', prompt)
    log('LLM metrics output')
    log(res)

    json_obj = LastJsonParser().parse(res)
    return json_obj['result']



