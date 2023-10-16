
from yacs.config import CfgNode as CN
import agents
from agents import AgentBase
from agents.environment import Tool, ToolError, Resource
from agents.environment.tools import ImageMetaTool, PythonTool
from agents.environment.resources import ImageResource
from agents.parsers.json import LastJsonParser

import json
import jsonschema

import oaapi


class SimpleAgent(AgentBase):

    def __init__(self, cfg_node: CN):
        super().__init__(cfg_node)
    
    def one_time_planning(self, input: str):

        tool_desc = ''
        for tool_id in self.tools:
            tool: Tool = self.tools[tool_id]
            inputs_str = '\n'.join([f'  name: {input_name} type: {tool.inputs[input_name]}' for input_name in tool.inputs])
            outputs_str = '\n'.join([f'  name: {output_name} type: {tool.outputs[output_name]}' for output_name in tool.outputs])
            tool_desc += f'''Tool ID: {tool_id}
Inputs and types:
{inputs_str}
Outputs and types:
{outputs_str}
Note:
{tool.description}


'''
        resource_desc = ''
        for resource_id in self.resources:
            resource: Resource = self.resources[resource_id]
            resource_desc += f'id: {resource_id} type: {resource.type}\n'
        
        json_example = '[{"id": "tool_id", "inputs": {"input name": "resource id", ...}, "outputs": {"output name": "resource id", ...}}, ...]'

        prompt = f'''
You are an AI assistant that make plan to use tools available to solve the user's request. First you should analyze the user's request and break them into sequential steps. In each step you can use one tool, please clarify which tool you want to use, why use it, what is used as its input, and how do you save its output. Finally, you will summarize your plan with an array of actions following strict json format like {json_example}.

Please follow the rules:
1. You can only use tools that are available.
2. The output of each step will be saved into a database with the id you assign, and may be used as input of subsequent steps if you need.
3. The plan need to produce necessary information to answer the user's request.

Here is a list of available tools:

{tool_desc}

Here is a list of resources currently in database:
{resource_desc}

Here is the user's request:

{input}

'''.strip()
        
        self.log(prompt)

        self.log('call api')
        res = oaapi.ask_once('You are a helpful assistant', prompt)

        self.log(res)

        parser = LastJsonParser()
        res_json = parser.parse(res)
        jsonschema.validate(res_json, {
            "type": "array",
            "items": agents.ACTION_SCHEMA,
        })

        # simply remove the tools that does not exist
        res_json = [action for action in res_json if action['id'] in self.tools]

        return res_json

    
    def summarize(self, input: str, action_history):
        print(action_history)
        self.log(self.resources)



if __name__ == '__main__':
    a = SimpleAgent(CN())
    a.add_tool('semantic_segmentation', ImageMetaTool('seg', 'this tool takes an image, performs semantic segmentation, and returns masks'))
    a.add_tool('object_detection', ImageMetaTool('det', 'this tool takes an image, performs object detection, and returns object bounding boxes in xywh format'))
    # a.add_tool('python', PythonTool())
    a.add_resource('input', ImageResource(None, meta={
        'seg': {'mask', 'xxx'},
        'det': {'asdfasdf': [[1,2,3,4], [3,4,5,6]]},
    }))
    a.chat('run segmentation on this image.')
