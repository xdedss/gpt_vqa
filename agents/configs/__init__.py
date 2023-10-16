

from yacs.config import CfgNode as CN

# default config

_C = CN()

_C.temperature = 0.0




def get_default():
    return _C.clone()
