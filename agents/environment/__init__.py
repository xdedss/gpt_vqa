


class ToolError(Exception):
    def __init__(self, reason) -> None:
        super().__init__(reason)
        self.reason = reason

# specify inputs and outputs:
# example:
# inputs = {
#     'input_img': 'image',
#     'asdasdf': 'text',
#     'asdfffsdfasd': 'json',
# ]
# image for opencv image wrapped in python dict with some meta data
# text for short text that goes directly to llm
# json for long data python dict ore list

class Tool():

    description = "No description"
    inputs = {}
    outputs = {}

    def __init__(self) -> None:
        pass

    def use(self, inputs):
        raise NotImplementedError()


class Resource():

    # this goes to llm prompt
    type = None
    # this goes to the tool
    data = None
    # some other info
    meta = None

    def string_for_llm(self):
        raise NotImplementedError()

