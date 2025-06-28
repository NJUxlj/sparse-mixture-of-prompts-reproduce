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
import math
import copy
from dataclasses import dataclass, field
from typing import Optional, Union
from collections import defaultdict

import torch
import torch.nn.functional as F

from transformers import BertForSequenceClassification, BertTokenizer, T5Tokenizer, BertConfig

from ..utils import PeftType, PromptLearningConfig

class PromptRoutingInit(str, enum.Enum):
    TEXT = "TEXT"
    RANDOM = "RANDOM"
    
    


@dataclass
class PromptRoutingConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.PromptRouting`].
    """
    
    
    
    perturb_router: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If true, a random perturbation is added to the routing values"
        }
    )
    
    
    gumbel: Optional[bool] = field(
        default=False, 
        metadata={
            "help": "Whether to use the auxiliary load balancing loss or not."
        }    
    )
    
    
    def __post_init__(self):
        self.peft_type = PeftType.PROMPT_ROUTING
