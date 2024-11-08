
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
    
    def one_time_planning(self, input: str):

        # keep consistent with make_sft_dataset.py + training template
        prompt = f'''Planning:
{input}
### Answer:'''.strip()
        
        self.log(prompt)

        self.log('call api')
        self.ask_confirm()
        res = oaapi.completion_once(
            prompt, 
            self.cfg.base_url,
            self.cfg.model_name, 
            stop='\n',
            )

        self.log('llm reply:')
        self.log(res)

        self.history['chat'].append((prompt, res))

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

        action_history_json = [
            {'action': action.action, 'action_result': action.action_result} for action in self.action_history_array
        ]
        
        resource_desc = ''
        for resource_id in self.resources:
            resource: Resource = self.resources[resource_id]
            resource_desc += f'resources["{resource_id}"] is as follows:\n{resource.detailed_desc()}\n'
        if (resource_desc.strip() == ''):
            resource_desc = 'There is no resource in database'
        
        self.history['resource_desc'] = resource_desc

        prompt = f'''Summarize:
{input}
{json.dumps(action_history_json, ensure_ascii=False)}
{resource_desc}
### Answer:'''.strip()
        
        self.log(prompt)
        self.log('call api')
        self.ask_confirm()
        res = oaapi.completion_once(
            prompt, 
            self.cfg.base_url,
            self.cfg.model_name,
            stop='\n',
            )

        self.log('llm reply:')
        self.log(res)
        
        self.history['chat'].append((prompt, res))

        return res



if __name__ == '__main__':
    a = SimpleAgent(CN())
    a.add_tool('semantic_segmentation', ImageMetaTool('seg', 'this tool takes an image, performs semantic segmentation, and returns masks with following labels: ["Asphalt"]', output_type='masks'))
    a.add_tool('object_detection', ImageMetaTool('det', 'this tool takes an image, performs object detection, and returns object bounding boxes in xywh format. The categories of objects that will be detected are ["Car", "Plane"]'))
    # a.add_tool('python', PythonTool())
    a.add_resource('input', ImageResource(None, meta={
        'seg': MasksResource({'Asphalt': 'xxx'}),
        'det': JsonResource({'Car': [[1,2,16,14], [20,6,13,23]]}),
    }))
    a.chat('Draw a image of fish for me')
