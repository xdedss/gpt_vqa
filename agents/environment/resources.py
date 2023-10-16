
from . import Resource


class ImageResource(Resource):

    def __init__(self, cv_image, meta) -> None:
        super().__init__()
        self.data = cv_image
        self.type = 'image'
        self.meta = meta

    def string_for_llm(self):
        return 'image'


