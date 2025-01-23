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
    PeftConfig
)



class PeftModel(PushToHubMixin, torch.nn.Module):
    pass