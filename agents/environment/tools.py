

from . import Tool, ToolError
from .resources import ImageResource, JsonResource, MasksResource

import jsonschema

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from difflib import SequenceMatcher

import numpy as np
import cv2

from scipy.ndimage import label


DETECTION_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
      "label": {
        "type": "string",
        "description": "Label of the object"
      },
      "bbox": {
        "type": "array",
        "items": {
          "type": "integer"
        },
        "minItems": 4,
        "maxItems": 4,
        "description": "Bounding box coordinates [x, y, width, height]"
      }
    },
    "required": ["label", "bbox"]
}


def edit_distance(a, b):
    required_edits = [
        code
        for code in (
            SequenceMatcher(a=a, b=b, autojunk=False)
            .get_opcodes()
        )
        if code[0] != 'equal'
    ]
    edit_count = 0
    for edit_type, start1, end1, start2, end2 in required_edits:
        edit_count += max(end1 - start1, end2 - start2)
    return edit_count

def find_nearest_string(a, strings):
    ''' returns min_distance, min_s '''
    min_distance = np.inf
    min_s = None
    for s in strings:
        d = edit_distance(a, s)
        if (d < min_distance):
            min_s = s
            min_distance = d
    return min_distance, min_s


class ImageMetaTool(Tool):
    ''' this will retrieve certain info from image meta, without wrapping, which means you have to wrap it in image meta '''
    
    description = '(image meta tool)'

    inputs = [{
        'name': 'image_input',
        'type': 'image',
    }]
    outputs = [{
        'name': 'output',
        'type': 'json',
    }]

    def __init__(self, key, description, output_type='json') -> None:
        super().__init__()
        self.key = key
        self.description = description
        self.outputs = [{
            'name': 'output',
            'type': output_type,
        }]

    def use(self, inputs):
        image = inputs['image_input']
        if (not isinstance(image, ImageResource)):
            raise ToolError("input type must be image")
        if image.meta is None:
            raise ToolError("image meta does not exists")
        if (self.key not in image.meta):
            raise ToolError(f"image meta does not have {self.key}")
        return {
            'output': image.meta[self.key]
        }


class DummyTool(Tool):
    ''' for testing '''
    
    def __init__(self, description, inputs, outputs, output_func) -> None:
        super().__init__()
        self.description = description
        self.outputs = outputs
        self.inputs = inputs
        self.output_func = output_func
    
    def use(self, inputs):
        return self.output_func(inputs)



class DetectionCounting(Tool):
    ''' counts detection result '''
    
    description = 'This tool will count the number of objects of given label in given detection result. The first input "detection_info" should be the resource id of the detection result, the second input "label_to_count" should be the plain text of the label to count. The output should be the resource id to store the counting result.'

    inputs = [{
        'name': 'detection_info',
        'type': 'json',
    },{
        'name': 'label_to_count',
        'type': 'text',
    }]
    outputs = [{
        'name': 'count',
        'type': 'json',
    }]

    def __init__(self) -> None:
        super().__init__()
    
    
    def count_det_label(self, data, label):
        count = 0
        for item in data:
            if item["label"] == label:
                count += 1
        return count

    def use(self, inputs):
        det_info = inputs['detection_info']
        label = inputs['label_to_count']

        det_info: JsonResource

        jsonschema.validate(det_info.data, {
            'type': 'array',
            'items': DETECTION_ITEM_SCHEMA
        })
        
        
        return {
            'count': JsonResource({
                'count': self.count_det_label(det_info.data, label)
            })
        }


class MaskArea(Tool):
    ''' counts area '''
    
    description = 'This tool will calculate the area of a segmentation mask. The first input "mask_to_calc" should be the resource id of the mask to calculate area. The output should be the resource id to store the result, the second input "label_to_calc" should be the plain text of the label to calculate. The result contains information about the absolute area in square meters.'

    inputs = [{
        'name': 'mask_to_calc',
        'type': 'mask',
    },{
        'name': 'label_to_calc',
        'type': 'text',
    }]
    outputs = [{
        'name': 'area',
        'type': 'json',
    }]

    def __init__(self, sqmeter_per_px = 0.0004) -> None:
        super().__init__()
        self.sqmeter_per_px = sqmeter_per_px

    def use(self, inputs):
        mask = inputs['mask_to_calc']
        label = inputs['label_to_calc']

        assert isinstance(mask, MasksResource)
        mask = mask.data[label].astype(bool)
        raw_count = mask.sum()
        proportion = mask.mean()
        sq_meter = raw_count * self.sqmeter_per_px
        
        return {
            'area': JsonResource({
                'square_meter': sq_meter,
                # 'pixel_area': int(raw_count),
                # 'proportion': float(proportion),
            })
        }


class MaskCount(Tool):
    ''' counts instances '''
    
    description = 'This tool will count the number of instance given a mask. The first input "mask_to_count" should be the resource id of the mask to count. The output should be the resource id to store the result, the second input "label_to_count" should be the plain text of the label to count.'

    inputs = [{
        'name': 'mask_to_count',
        'type': 'mask',
    },{
        'name': 'label_to_count',
        'type': 'text',
    }]
    outputs = [{
        'name': 'count',
        'type': 'json',
    }]

    def __init__(self) -> None:
        super().__init__()
    

    def use(self, inputs):
        mask = inputs['mask_to_count']
        label = inputs['label_to_count']

        assert isinstance(mask, MasksResource)
        mask = mask.data[label].astype(bool)
        count = count_connected_components(mask)
        
        
        return {
            'count': JsonResource({
                'number_of_instances': int(count),
            })
        }

def count_connected_components(binary_mask):
    binary_mask = binary_mask.astype(bool)
    labeled_array, num_features = label(binary_mask)
    return num_features


class MaskPathFinding(Tool):
    ''' finds path '''
    
    description = 'This tool will identify if there exists a path between 2 given points in a given a mask. The first input "mask_to_find_path" should be the resource id of the mask to find path in. The output should be the resource id to store the path finding result, the second input "label_to_find_path" should be the plain text of the label to find path in. "start_xy" and "end_xy" are the [x, y] coordinate of start and end points. '

    inputs = [{
        'name': 'mask_to_find_path',
        'type': 'mask',
    },{
        'name': 'label_to_find_path',
        'type': 'text',
    },{
        'name': 'start_xy',
        'type': 'list',
    },{
        'name': 'end_xy',
        'type': 'list',
    }]
    outputs = [{
        'name': 'path_finding_result',
        'type': 'json',
    }]

    def __init__(self) -> None:
        super().__init__()
    

    def use(self, inputs):
        mask = inputs['mask_to_find_path']
        label = inputs['label_to_find_path']
        start_xy = inputs['start_xy']
        end_xy = inputs['end_xy']
        start_xy = (int(start_xy[0]), int(start_xy[1]))
        end_xy = (int(end_xy[0]), int(end_xy[1]))

        assert isinstance(mask, MasksResource)
        mask = mask.data[label].astype(bool)
        path = astar(mask, *start_xy, *end_xy)
        
        
        return {
            'path_finding_result': JsonResource({
                'has_path': len(path) > 0,
            })
        }

def astar(binary_mask, x1, y1, x2, y2, max_size=128):

    original_h, original_w = binary_mask.shape
    scale = 1.0
    if max(original_h, original_w) > max_size:
        scale = max_size / max(original_h, original_w)
        binary_mask = cv2.resize(binary_mask.astype(np.uint8), None, fx=scale, fy=scale).astype(bool)

    grid = Grid(matrix=binary_mask.astype(int))

    start = grid.node(int(x1 * scale), int(y1 * scale))
    end = grid.node(int(x2 * scale), int(y2 * scale))

    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    path, runs = finder.find_path(start, end, grid)

    # print('operations:', runs, 'path length:', len(path))
    # print(grid.grid_str(path=path, start=start, end=end))

    # len(path)==0 means infeasible
    return [(int(pt.x / scale), int(pt.y / scale)) for pt in path]


def execute_user_function(user_function_str, resources):
    # Define the expected function name and argument
    expected_function_name = "run"
    expected_function_args = "resources"

    # Try to compile the user function string
    try:
        compiled_function = compile(user_function_str, '<string>', 'exec')
    except SyntaxError:
        return "SyntaxError: Unable to compile the user function."

    # Execute the compiled code in a local namespace
    local_namespace = {}
    try:
        exec(compiled_function, globals(), local_namespace)
    except Exception as e:
        return f"Error while executing the user function: {str(e)}"

    # Check if the expected function name is defined in the local namespace
    if expected_function_name not in local_namespace:
        return f"User function '{expected_function_name}' not defined."

    # Check if the user function has the expected argument
    user_function = local_namespace[expected_function_name]
    if not hasattr(user_function, '__code__'):
        return "User function has no code object."

    function_args = user_function.__code__.co_varnames
    if len(function_args) != 1 or function_args[0] != expected_function_args:
        return f"User function should have a single argument named '{expected_function_args}'."

    # Execute the user function with the provided argument
    try:
        result = user_function(resources)
        return result
    except Exception as e:
        return f"Error while executing the user function: {str(e)}"

class PythonTool(Tool):

    def __init__(self) -> None:
        super().__init__()

    
    description = '''This tool will automatically write python code according to the specifications you provide to it. 
where the resources is the dict representing the database whose keys are resource id. You can read from it or modify it in the code. For this tool, you should directly provide the code itself instead of resource id.'''

    inputs = {
        'code': 'text'
    }
    outputs = {
        'return_value': 'any'
    }

    def use(self, inputs):
        code = inputs[code]
        # TODO: resources? 
        