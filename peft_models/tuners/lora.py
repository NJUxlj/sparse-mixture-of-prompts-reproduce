import math
import warnings
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F



from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    PreTrainedModel,
)

import bitsandbytes as bnb


from typing import List, Dict, Tuple, Optional, Any

from ..utils import PeftConfig, PeftType, transpose


@dataclass
class LoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Lora`].

    Args:
        r (`int`): Lora attention dimension
        target_modules (`List[str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        merge_weights (`bool`):
            Whether to merge the weights of the Lora layers with the base transformer model in `eval` mode.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        enable_lora ( `List[bool]`): Used with `lora.MergedLinear`.
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """
    
    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[list] = field(default=None, metadata={"help": "List of modules to replace with Lora"})
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})

class LoraModel(nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig >>> from peft import LoraModel, LoraConfig >>>
        config = LoraConfig(
            peft_type="LORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.01, )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> lora_model = LoraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """
    
    def __init__(self, config, model):
        super().__init__()
        self.peft_config = config
        self.model: PreTrainedModel = model
        self._find_and_replace()
        mark_only_lora_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward
        
    def _find_and_replace(self):
        is_target_modules_in_base_model = False
        
        
        
        
    def _get_submodules(self, key):
        '''
        根据给定的模块路径key，获取该模块的父模块、目标模块和模块名称。
        '''
        
        '''
        key.split(".")[:-1]：将模块路径按"."分割，并去掉最后一部分
        ".".join(...)：将去掉最后一部分的路径重新用"."连接
        self.model.get_submodule(...)：获取父模块对象
        '''
        parent = self.model.get_submodule(".".join(key.split(".")[:-1])) 
        target_name = key.split(".")[-1] #
        target = self.model.get_submodule(key)
        
        return parent, target, target_name # 返回三个值：父模块对象、目标模块对象、目标模块名称
    
    
    def _replace_module(self, parent_module, child_name, new_module, old_module):
        pass
    
    
    
    
    def __getattr__(self, name:str):
        pass


























# had to adapt it for `lora_only` to work
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    pass