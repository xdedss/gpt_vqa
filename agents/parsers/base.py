

class ParseFailed(Exception):
    pass

class LLMParser():

    def __init__(self):
        # do nothing by default
        pass

    def parse(self, text: str) -> str:
        raise NotImplementedError()


