
from yacs.config import CfgNode as CN
import agents
from agents import AgentBase
from agents.environment import Tool, ToolError, Resource
from agents.environment.tools import ImageMetaTool, PythonTool, find_nearest_string
from agents.environment.resources import ImageResource, JsonResource, MasksResource
from agents.parsers.json import LastJsonParser

import json
import jsonschema

import oaapi


class SimpleAgent(AgentBase):

    CONTEXT = '''You are an AI assistant that use tools to solve the user's request. The first stage is planning, where you are provided with informations about the user input, tools available and resources available, and you should make a plan to use tools available to solve the user's request. The second stage is running, where the plan is executed and results are stored in resources database. The third stage is summarizing, where you make conclusion to the user's request based on previous stages.
'''.strip()
    
    require_confirm = False

    def __init__(self, cfg_node: CN):
        super().__init__(cfg_node)
    
    def ask_confirm(self):
        if (self.require_confirm):
            res = input('confirm?')
            if (res.lower() in ['n', 'no']):
                import sys
                self.logw('user cancelled action')
                sys.exit(0)
    
    def feedback_planning(self, input: str):

        tool_desc = ''
        for tool_id in self.tools:
            tool: Tool = self.tools[tool_id]
            inputs_str = json.dumps(tool.inputs)
            outputs_str = json.dumps(tool.outputs)
            tool_desc += f'''Tool ID: {tool_id}
Inputs:
{inputs_str}
Outputs:
{outputs_str}
Note:
{tool.description}

'''
        action_history_desc = ''
        for action_history in self.action_history_array:
            action_history_desc += f'action: {json.dumps(action_history.action)}\nresult: {json.dumps(action_history.action_result)}\n'
        if (action_history_desc.strip() == ''):
            action_history_desc = 'Currently no tool call has been performed'
        
        resource_desc = ''
        for resource_id in self.resources:
            resource: Resource = self.resources[resource_id]
            resource_desc += f'resources["{resource_id}"] is as follows:\nid: {resource_id}, type: {resource.type}\nDetails: {resource.detailed_desc()}\n'
        if (resource_desc.strip() == ''):
            resource_desc = 'There is no resource in database'
        
        json_example = '[{"id": "tool_id" ,"reason": "reason to perform this step", "inputs": {"input name": "resource id", ...}, "outputs": {"output name": "resource id", ...}}, ...]'

        prompt = f'''{self.CONTEXT}
Now you will perform the first stage: planning. The planning is done iteratively. In this iteration, you should first examine the results of tools you used in previous iterations if there are any. Then analyze the user's request and think about what remains to be done and break them into sequential steps. In each step you can use one tool, please clarify which tool you want to use, why use it, what is used as its input, and how do you save its output. Finally, you will summarize your plan for this iteration following strict json format like {json_example}.

Please follow the rules:
1. You can only use tools that are available, do not make up tools that do not exist.
2. You have the ability to analyze text or json by yourself, so you do not have to use tool for simple analysis.
3. The summarized json should only contain calls to tools.
4. The summarized json should not contain steps that are already done.
5. The output of each step will be saved into a database with the id you assign, and may be used as input of subsequent steps if you need.
6. The plan needs to gather necessary information to answer the user's request.
7. If you think all necessary steps are already done, you can summarize your plan with an empty list [].


Here is a list of available tools:

{tool_desc}

Here is sequential tool calls that you planned and performed:
{action_history_desc}

Here is the resources object that contains resources that are used as inputs or outputs of the tool calls:
{resource_desc}

Here is the user's request:
{input}

'''.strip()
        
        self.log(prompt)

        self.log('call api')
        self.ask_confirm()
        res = oaapi.ask_once('You are a helpful assistant', prompt, self.cfg.model_name)

        self.log('llm reply:')
        self.log(res)

        parser = LastJsonParser()
        res_json = parser.parse(res)
        jsonschema.validate(res_json, {
            "type": "array",
            "items": agents.ACTION_SCHEMA,
        })

        self.log('parsed JSON:')
        self.log(res_json)

        # replace id with the most similar ones
        for action in res_json:
            if action['id'] not in self.tools:
                nearest_d, nearest_id = find_nearest_string(action['id'], self.tools.keys())
                if (nearest_d < 8):
                    action['id'] = nearest_id
        # simply remove the tools that does not exist
        res_json = [action for action in res_json if action['id'] in self.tools]
        
        self.log('filtered JSON:')
        self.log(res_json)

        return res_json

    
    def summarize(self, input: str):
        
        tool_desc = ''
        for tool_id in self.tools:
            tool: Tool = self.tools[tool_id]
            inputs_str = json.dumps(tool.inputs)
            outputs_str = json.dumps(tool.outputs)
            tool_desc += f'''Tool ID: {tool_id}
Inputs:
{inputs_str}
Outputs:
{outputs_str}
Note:
{tool.description}

'''
        
        action_history_desc = ''
        for action_history in self.action_history_array:
            action_history_desc += f'action: {json.dumps(action_history.action)}\nresult: {json.dumps(action_history.action_result)}\n'
        if (action_history_desc.strip() == ''):
            action_history_desc = 'No tool call has been performed'

        resource_desc = ''
        for resource_id in self.resources:
            resource: Resource = self.resources[resource_id]
            resource_desc += f'resources["{resource_id}"] is as follows:\nid: {resource_id}, type: {resource.type}\nDetails: {resource.detailed_desc()}\n'
        if (resource_desc.strip() == ''):
            resource_desc = 'There is no resource in database'
        

        prompt = f'''{self.CONTEXT}
Now you will perform the third stage: summarizing. Firstly you will describe what tool you have used and how you used it in natural language. Next, you will summarize whether the plan to solve user's request is successful. Finally, you will respond to the user's request with a clear answer.

Please follow the rules:
1. The final answer should be directly answering the user's request without irrelevant informations.
2. If the user is asking "how many", provide an exact number.

Here is a list of tools and their descriptions:
{tool_desc}

Here is sequential tool calls that you planned and performed:
{action_history_desc}

Here is the resources object that contains resources that are used as inputs or outputs of the actions:
{resource_desc}

Here is the user's request:
{input}
'''.strip()
        
        self.log(prompt)
        self.log('call api')
        self.ask_confirm()
        res = oaapi.ask_once('You are a helpful assistant', prompt, self.cfg.model_name)

        self.log('llm reply:')
        self.log(res)

        return res




if __name__ == '__main__':
    a = SimpleAgent(CN())
    a.require_confirm = True
    a.add_tool('semantic_segmentation', ImageMetaTool('seg', 'this tool takes an image, performs semantic segmentation, and returns masks with following labels: ["Asphalt"]', output_type='masks'))
    a.add_tool('object_detection', ImageMetaTool('det', 'this tool takes an image, performs object detection, and returns object bounding boxes in xywh format. The categories of objects that will be detected are ["Car", "Plane"]'))
    # a.add_tool('python', PythonTool())
    a.add_resource('input', ImageResource(None, meta={
        'seg': MasksResource({'Asphalt': 'xxx'}),
        'det': JsonResource({'Car': [[1,2,16,14], [20,6,13,23]]}),
    }))
    a.chat_feedback('If there are cars in the image, perform segmentation on the image. If not, do not perform segmentation')
