
from . import Resource
import json

class ImageResource(Resource):

    def __init__(self, cv_image, meta) -> None:
        super().__init__()
        self.data = cv_image
        self.type = 'image'
        self.meta = meta
    
    def detailed_desc(self) -> str:
        return 'An opencv image object'

class JsonResource(Resource):

    def __init__(self, obj) -> None:
        super().__init__()
        self.data = obj
        self.type = 'json'
    
    def detailed_desc(self) -> str:
        return 'A json object: ' + json.dumps(self.data)


class MasksResource(Resource):

    def __init__(self, masks_dict):
        super().__init__()
        self.data = masks_dict
        self.type = 'masks'
    
    def detailed_desc(self):
        # print('formatting', self.data)
        return f'A dict with these keys: {json.dumps([k for k in self.data])}, each with a mask of corresponding lable.'
