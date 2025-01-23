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



import inspect
import os
import warnings
import torch

'''
dispatch_model函数:
    用于将模型分发到多个设备上，例如在多GPU环境中。
    它可以自动将模型的不同部分放置在不同的设备上，以实现并行计算。

infer_auto_device_map函数
    用于自动推断模型的设备映射。它可以根据模型的大小和可用的设备资源，
    自动决定如何将模型的各个层分配到不同的设备上。

hooks模块
    提供了几个有用的钩子函数，用于在训练过程中对模型和数据进行处理。
    例如，AlignDevicesHook用于确保模型的所有参数都在同一个设备上，add_hook_to_module和remove_hook_from_submodules用于向模型或其子模块添加或移除钩子。

utils模块
    提供了一些实用函数，例如get_balanced_memory用于获取每个设备上可用的最大内存，
    这对于在多GPU环境中分配内存非常有用。
'''

from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_module, remove_hook_from_submodules

from accelerate.utils import get_balanced_memory
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput

from transformers.utils import PushToHubMixin


from huggingface_hub import hf_hub_download


from .tuners import (
    LoraModel
)

from .utils import (
    PeftConfig,

    PeftType,
    PromptLearningConfig
)



class PeftModel(PushToHubMixin, torch.nn.Module):
    def __init__(self, model, peft_config:PeftConfig):
        super().__init__()
        self.peft_config = peft_config
        self.base_model = model
        self.config = self.base_model.config
        self.modules_to_save = None





    

    def save_pretrained(self, save_directory, **kwargs):
        pass


    @classmethod    # 可以用类名调用
    def from_pretrained(cls, model, model_id, **kwargs):
        pass


    
    def _setup_prompt_encoder(self):
        pass


    def get_prompt_embedding_to_save(self):
        pass



    def get_prompt(self, batch_size):
        pass


    def get_prompt_routing(self, batch_size, input_ids, inputs_embeds, attention_mask):
        pass    




    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """


    def __getattr__(self, name:str):
        """Forward missing attributes to the wrapped module."""



    def forward(self, *args, **kwargs):
        """
        Forward pass of the model.
        """






class PeftModelForSequenceClassification(PeftModel):
    def __init__(self, model, peft_config:PeftConfig):
        super().__init__(model, peft_config)
        self.modules_to_save = ["classifier", "score"]

        for name, _ in self.base_model.named_children():
            if any()
    


    def forward(self,):
        pass



    def _prefix_tuning_forward(self):
        pass









class PeftModelForCausalLM(PeftModel):
    """
    Peft model for Causal LM

    Args:
        model ([`PreTrainedModel`]): Base transformer model
        peft_config ([`PeftConfig`]): Peft config.


    Example::

        >>> from transformers import AutoModelForCausalLM >>> from peft import PeftModelForCausalLM, get_peft_config
        >>> config = {
                'peft_type': 'PREFIX_TUNING', 'task_type': 'CAUSAL_LM', 'inference_mode': False, 'num_virtual_tokens':
                20, 'token_dim': 1280, 'num_transformer_submodules': 1, 'num_attention_heads': 20, 'num_layers': 36,
                'encoder_hidden_size': 1280, 'prefix_projection': False, 'postprocess_past_key_value_function': None
            }
        >>> peft_config = get_peft_config(config) >>> model = AutoModelForCausalLM.from_pretrained("gpt2-large") >>>
        peft_model = PeftModelForCausalLM(model, peft_config) >>> peft_model.print_trainable_parameters() trainable
        params: 1843200 || all params: 775873280 || trainable%: 0.23756456724479544
    """
    def __init__(self):
        pass
