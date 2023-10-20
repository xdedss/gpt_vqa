
import oaapi
from agents.parsers.json import JsonParser

# gt should be categorical, check if equal
def compare_question_answer_groundtruth(user_question, response, gt):
    
    json_str = '{"result": true/false}'
    prompt = f'''
You are an AI assistant that assess the quality of response produced by an autonomous system in respond to given user request. The system should answer any question that the user asks at the end of the response. 

The user request is:

```
{user_question}
```

The answer to be assessed is:

```
{response}
```

The gound truth of the question is: {gt}

You are going to assess the following aspect of the answer: Does the response align with the fact?


You should first explain your thought and your assessment. Finally you should summarize your assessment with strict json format: {json_str}
'''.strip()
    
    res = oaapi.ask_once('You are a helpful assistant.')
    json_obj = JsonParser().parse(res)
    return json_obj['result']

# gt should be a number
def compare_question_answer_groundtruth_numerical(user_question, response, gt):
    json_str = '{"result": true/false, "numerical_value_in_response": ...}'
    prompt = f'''
You are an AI assistant that assess the quality of response produced by an autonomous system in respond to given user request. The system should answer any question that the user asks at the end of the response. 

The user request is:

```
{user_question}
```

The answer to be assessed is:

```
{response}
```

The gound truth of the question is: {gt}

Firstly, you should extract the numerical answer. Then you will compare the answer in the response to the ground truth. You are going to assess the following aspect of the answer: Does the response align with the fact?

Here are guidelines:
1. If the question is about counting and the answer denies the existence of the required object, we consider it equivalent to answering with number 0.

You should first explain your thought and your assessment. Finally you should summarize your assessment with strict json format: {json_str}
'''.strip()

# no gt is provided, check logical connection between q and a
def compare_question_answer(user_question, response):
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
    
    res = oaapi.ask_once('You are a helpful assistant.')
    json_obj = JsonParser().parse(res)
    return json_obj['result']



