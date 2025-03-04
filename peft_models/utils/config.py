# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import enum
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional, Union

from transformers.utils import PushToHubMixin

from huggingface_hub import hf_hub_download

from .adapters_utils import CONFIG_NAME


class PeftType(str, enum.Enum):
    PROMPT_TUNING = "PROMPT_TUNING"
    P_TUNING = "P_TUNING"
    PREFIX_TUNING = "PREFIX_TUNING"
    LORA = "LORA"
    PROMPT_MIX = "PROMPT_MIX"
    PROMPT_ROUTING = "PROMPT_ROUTING"
    
    
    


class TaskType(str, enum.Enum):
    SEQ_CLS = "SEQ_CLS"
    SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
    CAUSAL_LM = "CAUSAL_LM"
    TOKEN_CLS = "TOKEN_CLS"
    
    
    
@dataclass
class PeftConfigMixin(PushToHubMixin):
    r"""
    This is the base configuration class for PEFT adapter models. It contains all the methods that are common to all
    PEFT adapter models. This class inherits from `transformers.utils.PushToHubMixin` which contains the methods to
    push your model to the Hub. The method `save_pretrained` will save the configuration of your adapter model in a
    directory. The method `from_pretrained` will load the configuration of your adapter model from a directory.

    Args:
        peft_type (Union[[`~peft.utils.config.PeftType`], `str`]): The type of Peft method to use.
    """
    peft_type: Optional[PeftType] = field(default=None, metadata={"help": "The type of PEFT model."})

    @property
    def __dict__(self):
        return asdict(self)
    
    def to_dict(self):
        return self.__dict__
    
    
    def save_pretrained(self, save_directory, **kwargs):
        r"""
        This method saves the configuration of your adapter model in a directory.

        Args:
            save_directory (`str`):
                The directory where the configuration will be saved.
            **kwargs:
                Additional keyword arguments passed along to the `transformers.utils.PushToHubMixin.push_to_hub`
                method.
        """
        
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)
        
        output_dict = self.__dict__
        output_path = os.path.join(save_directory, CONFIG_NAME)
        
        
        
        with open(output_path, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(output_dict, indent=2, sort_keys=True))
            
    
    
    @classmethod
    def from_pretrained(cls,):
        pass
    
    
    
    
    
    
    
    
    @classmethod
    def from_json_file(cls, path_json_file, **kwargs):
        
            





@dataclass
class PeftConfig(PeftConfigMixin):
    """
    This is the base configuration class to store the configuration of a :class:`~peft.PeftModel`.

    Args:
        peft_type (Union[[`~peft.utils.config.PeftType`], `str`]): The type of Peft method to use.
        task_type (Union[[`~peft.utils.config.TaskType`], `str`]): The type of task to perform.
        inference_mode (`bool`, defaults to `False`): Whether to use the Peft model in inference mode.
    """
    
    base_model_name_or_path: str = field(default=None, metadata={"help": "The name of the base model to use."})
    
    peft_type: Union[str, PeftType] = field(default=None, metadata = {"help": "peft type"})
    task_type:Union[str, TaskType] = field(default = None, metadata={"help": "task type"})
    inference_mode: bool = field(default=False, metadata={"help": "Whether to use the Peft model in inference mode."})



@dataclass
class PromptLearningConfig(PeftConfig):
    """
    This is the base configuration class to store the configuration of a Union[[`~peft.PrefixTuning`],
    [`~peft.PromptEncoder` (i.e., P-Tuning)], [`~peft.PromptTuning`]].

    Args:
        num_virtual_tokens (`int`): The number of virtual tokens to use.
        token_dim (`int`): The hidden embedding dimension of the base transformer model.
        num_transformer_submodules (`int`): The number of transformer submodules in the base transformer model.
        num_attention_heads (`int`): The number of attention heads in the base transformer model.
        num_layers (`int`): The number of layers in the base transformer model.
        
    ## 注意点：
    num_transformer_submodules:
        通常指一个Transformer块中包含的独立组件数量
        默认值为1，意味着每个Transformer层被视为一个整体单元
        在更复杂的架构中，可能包含多个子模块（如多头注意力机制和前馈网络）
    
    num_layers:
        表示Transformer模型的总层数
        指整个模型堆叠的Transformer层数量
    """
    
    num_virtual_tokens: int = field(default=None, metadata={"help": "Number of virtual tokens"})

    token_dim:int = field(default=None, metadata = {"help": "The hidden embedding dimension of the base transformer model"})
    
    num_transformer_submodules: Optional[int] = field(default=1, metadata={"help": "Number of transformer submodules"})

    num_attention_heads: Optional[int] = field(default=None, metadata={"help": "Number of attention heads"})
    
    num_layers:Optional[int] = field(default=None, metadata={"help": "Number of transformer layers"})

