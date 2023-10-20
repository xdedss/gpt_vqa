

from . import Tool, ToolError
from .resources import ImageResource, JsonResource

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

    
    description = '''This tool runs python code in a python environment. the code should be provided as a function:
def run(resources):
    ...
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
        