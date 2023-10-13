

class ParseFailed(Exception):
    pass

class LLMParser():

    def __init__(self):
        pass

    def parse(self, text: str) -> str:
        raise NotImplementedError()


