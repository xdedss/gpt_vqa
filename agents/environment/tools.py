

from . import Tool, ToolError


class ImageMetaTool(Tool):
    ''' this will retrieve certain info from image meta '''
    
    description = '(image meta tool)'

    inputs = {
        'image_input': 'image'
    }
    outputs = {
        'output': 'json'
    }

    def __init__(self, key, description) -> None:
        super().__init__()
        self.key = key
        self.description = description

    def use(self, inputs):
        image = inputs['image_input']
        if image.meta is None:
            raise ToolError("image meta does not exists")
        if (self.key not in image.meta):
            raise ToolError(f"image meta does not have {self.key}")
        return {
            'output': image.meta[self.key]
        }


