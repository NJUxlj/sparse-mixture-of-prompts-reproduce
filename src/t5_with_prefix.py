"""
Changing T5Attention's forward to support prefix tuning, along with subclassing other classes that
use T5Attention. Changes in T5Attention's forward from are marked with
"# <CHANGE>" and "# </CHANGE>". It's possible that the added logic can be separated as some code
that entirely preceeds the original forward, s.t. we can call super().forward() without code
duplciation. Even better, we might be able to use a pre-hook so that most of this won't be needed.
"""
import os
import sys

# 保证在脚本运行时处于项目的根目录
if "sparse-mixture-of-prompts-reproduce" not in os.getcwd():
    os.chdir("sparse-mixture-of-prompts-reproduce")
sys.path.append(os.getcwd())

import torch
from torch import nn
from transformers.models.t5.modeling_t5 import (
    T5Config,
    T5Attention,
    T5LayerSelfAttention,
    T5LayerCrossAttention,
    T5Block,
    T5Stack,
    T5ForConditionalGeneration,
)



class T5WithPrefixConfig(T5Config):
    def __init__(
        self, num_prefix=None, reparam=False, reparam_dim=512, no_decoder_self_attn=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_prefix = num_prefix 
        self.reparam = reparam
        self.reparam_dim = reparam_dim
        self.no_decoder_self_attn = no_decoder_self_attn
        
        
        
    @classmethod
    def get_config_dict(cls, *args, **kwargs):
        config_dict, kwargs = T5Config.get_config_dict(*args, **kwargs)
        for field in ("num_prefix", "reparam_dim"):
            assert field not in config_dict
            if field in kwargs:
                config_dict[field] = kwargs.pop(field)
        return config_dict, kwargs