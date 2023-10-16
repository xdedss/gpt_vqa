

from yacs.config import CfgNode as CN
import jsonschema
import datetime

from . import environment

# TODO: better way to write down schema
ACTION_SCHEMA = {
    'type': 'object',
    'properties': {
        'id': {'type': 'string'},
        'inputs': {'type': 'object'},
        'outputs': {'type': 'object'},
    },
    'required': ['id', 'inputs', 'outputs'],
}


class AgentBase():

    def __init__(self, cfg_node: CN):
        self.cfg = cfg_node
        self.resources = dict()
        self.tools = dict()
    
    def add_resource(self, id: str, resource):
        if id not in self.resources:
            self.resources[id] = resource
        else:
            raise ValueError(f'resource id {id} exists')

    def add_tool(self, id: str, tool: environment.Tool):
        if id not in self.tools:
            self.tools[id] = tool
        else:
            raise ValueError(f'action id {id} exists')

    def one_time_planning(self, input: str):
        raise NotImplementedError()
    
    def summarize(self, input: str, action_history):
        raise NotImplementedError()

    def run_action(self, action: dict) -> dict:
        ''' run an action description
        {'id': xxx, 'inputs': {'image': xxx}, 'outputs': {'mask': xxx, 'boxes': xxx}}
        return {"success": true/false, "reason": "xxx"}
        '''
        # check json
        try:
            jsonschema.validate(action, ACTION_SCHEMA)
        except jsonschema.ValidationError:
            return {"success": False, "reason": "action json format invalid(this should never happen)"}
        # check action exists
        id = action['id']
        inputs_designation = action['inputs']
        outputs_designation = action['outputs']
        if (id not in self.tools):
            return {"success": False, "reason": f"tool id {id} does not exist"}
        # check required inputs are designated
        actual_input_dict = dict()
        tool = self.tools[id]
        tool: environment.Tool
        for input_key in tool.inputs:
            input_type = tool.inputs[input_key]
            if (input_key not in inputs_designation):
                return {"success": False, "reason": f"input '{input_key}' of tool is not specified"}
            if (input_type == 'text'):
                input_text = inputs_designation[input_key]
                actual_input_dict[input_key] = input_text
            else:
                input_resource_id = inputs_designation[input_key]
                if (input_resource_id not in self.resources):
                    return {"success": False, "reason": f"resource id {input_resource_id} does not exist"}
                else:
                    actual_input_dict[input_key] = self.resources[input_resource_id]
        # check designated outputs exists in the tool
        for output_key in outputs_designation:
            if (output_key not in tool.outputs):
                return {"success": False, "reason": f"output '{output_key}' does not exist"}
            output_resource_id = outputs_designation[output_key]
            if (output_resource_id in self.resources):
                self.logw(f"resource id {output_resource_id} will be overridden")
        # run the tool
        try:
            actual_output_dict = tool.use(actual_input_dict)
        except environment.ToolError as e:
            return {"success": False, "reason": f"ToolError: {e.reason}"}
        except Exception as e:
            return {"success": False, "reason": f"Unexpected error using tool: {e}"}
        
        for output_key in outputs_designation:
            output_resource_id = outputs_designation[output_key]
            actual_output = actual_output_dict[output_key]
            self.resources[output_resource_id] = actual_output
        return {"success": True}
            


    def chat(self, input: str) -> str:

        # ================= planning =================
        # 1 time planning vs feedback
        # 1 time
        actions = self.one_time_planning(input)

        action_history = []
        for action in actions:
            # action: {'id': xxx, 'inputs': {'image': xxx}, 'outputs': {'mask': xxx, 'boxes': xxx}}
            # result should contain: success/fail, why fail
            result_description = self.run_action(action)
            action_history.append((action, result_description))
        
        # # feedback
        # action_history = []
        # while (True):
        #     action = self.feedback_planning(input, action_history)
        #     result_description = self.run_action(action)
        #     action_history.append((action, result_description))

        # ================= summarize ==============
        summary = self.summarize(input, action_history)

        return summary

            
    def loge(self, message):
        now = datetime.datetime.now()
        print(f'[{now.strftime("%Y%m%d-%H%M%S")}][ERROR] {message}')
    def logw(self, message):
        now = datetime.datetime.now()
        print(f'[{now.strftime("%Y%m%d-%H%M%S")}][WARN] {message}')
    def log(self, message):
        now = datetime.datetime.now()
        print(f'[{now.strftime("%Y%m%d-%H%M%S")}][INFO] {message}')

