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


from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..utils import PeftType, PromptLearningConfig

from typing import Optional






@dataclass
class PrefixRoutingConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.PrefixEncoder`].

    Args:
        encoder_hidden_size (`int`): The hidden size of the prompt encoder.
        prefix_projection (`bool`): Whether to project the prefix embeddings.
    """
    
    encoder_hidden_size: int = field(
        default=None,
        metadata={"help": "The hidden size of the encoder"},
    )
    prefix_projection: bool = field(
        default=False,
        metadata={"help": "Whether to project the prefix tokens"},
    )
    
    num_virtual_tokens_full: Optional[int] = field(
        default=100, 
        metadata={
            "help": "The number of target tokens for top-k routing"
        }
    )
    
    def __post_init__(self):
        self.peft_type = PeftType.PREFIX_TUNING






# Based on https://github.com/THUDM/P-tuning-v2/blob/main/model/prefix_encoder.py
# with some refactor
class PrefixRoutingEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        token_dim = config.token_dim
        num_layers = config.num_layers
        encoder_hidden_size = config.encoder_hidden_size
        num_virtual_tokens = config.num_virtual_tokens
        
        if self.prefix_projection and not config.inference_mode:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(num_virtual_tokens, token_dim)
            self.transform = nn.Sequential(
                nn.Linear(token_dim, encoder_hidden_size),
                nn.Tanh(),
                nn.Linear(encoder_hidden_size, num_layers * 2 * token_dim),
            )
        
        
        else:
            self.embedding = torch.nn.Embedding(num_virtual_tokens, num_layers * 2 * token_dim)
        
        
        
    
    
    def forward(self, prefix:torch.Tensor):
        '''
        prefix.shape = [B, prefix_len]
        '''
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix) # shape = [B, L, D]
            past_key_values = self.transform(prefix_tokens) 
        else:
            past_key_values = self.embedding(prefix)
            
        
        return past_key_values