import os
import sys

# 保证在脚本运行时处于项目的根目录
if "sparse-mixture-of-prompts-reproduce" not in os.getcwd():
    os.chdir("sparse-mixture-of-prompts-reproduce")
sys.path.append(os.getcwd())

import torch
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Config,
    Qwen2Attention,
    Qwen2MLP,
    Qwen2DecoderLayer,
    Qwen2FlashAttention2,
    Qwen2ForCausalLM
)



class Qwen2WithPrefixConfig(Qwen2Config):
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
        config_dict, kwargs = Qwen2Config.get_config_dict(*args, **kwargs)
        for field in ("num_prefix", "reparam_dim"):
            assert field not in config_dict
            if field in kwargs:
                config_dict[field] = kwargs.pop(field)
        return config_dict, kwargs