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
    LoraModel,
)

from .utils import (
    PeftConfig,

    PeftType,
    PromptLearningConfig,
    _set_trainable,
)



class PeftModel(PushToHubMixin, torch.nn.Module):
    """
    Parameter-Efficient Fine-Tuning Model. Base model encompassing various Peft methods.

    Args:
        model ([`PreTrainedModel`]): The base transformer model used for Peft.
        peft_config ([`PeftConfig`]): The configuration of the Peft model.


    **Attributes**:
        - **base_model** ([`PreTrainedModel`]) -- The base transformer model used for Peft.
        - **peft_config** ([`PeftConfig`]) -- The configuration of the Peft model.
        - **modules_to_save** (`list` of `str`) -- The list of sub-module names to save when
        saving the model.
        - **prompt_encoder** ([`PromptEncoder`]) -- The prompt encoder used for Peft if
        `isinstance(self.peft_config, PromptLearningConfig)`.

        - **prompt_tokens** (`torch.Tensor`) -- The virtual prompt tokens used for Peft if
        `isinstance(self.peft_config, PromptLearningConfig)`.

        - **transformer_backbone_name** (`str`) -- The name of the transformer
        backbone in the base model if `isinstance(self.peft_config, PromptLearningConfig)`.
            - 骨干网络，通常是指一个预训练的大型模型

        - **word_embeddings** (`torch.nn.Embedding`) -- The word embeddings of the transformer backbone
        in the base model if `isinstance(self.peft_config, PromptLearningConfig)`.
    """
    def __init__(self, model, peft_config:PeftConfig):
        super().__init__()
        self.peft_config = peft_config
        self.base_model = model
        self.config = self.base_model.config
        self.modules_to_save = None

        if isinstance(self.peft_config, PromptLearningConfig):
            self._setup_prompt_encoder()
        else:
            self.base_model = LoraModel(peft_config, model)

        if getattr(self.peft_config, "modules_to_save", None) is not None:
            self.modules_to_save = self.peft_config.modules_to_save
            _set_trainable(self)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    def save_pretrained(self, save_directory, **kwargs):
        r"""
        Args:
        This function saves the adapter model and the adapter configuration files to a directory, so that it can be
        re-loaded using the `LoraModel.from_pretrained` class method, and also used by the `LoraModel.push_to_hub`
        method.
            save_directory (`str`):
                Directory where the adapter model and configuration files will be saved (will be created if it does not
                exist).
            **kwargs:
                Additional keyword arguments passed along to the `push_to_hub` method.
        """

        if os.path.isfile(save_directory):
            raise ValueError(f"{save_directory} should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)


        # save only the trainable weights


        

    @classmethod    # 可以用类名调用
    def from_pretrained(cls, model, model_id, **kwargs):
        pass


    
    def _setup_prompt_encoder(self):
        num_transformer_submodules = 0
        transformer_backbone = None


    def get_prompt_embedding_to_save(self):
        """
        Returns the prompt embedding to save when saving the model. Only applicable when `peft_config.peft_type !=
        PeftType.LORA`.
        """
        
        prompt_tokens = ""
        
        if self.peft_config.peft_type == PeftType.PrefixTuning:
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

    def __init__(self, model, peft_config: PeftConfig):
        super().__init__(model, peft_config)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if not isinstance(self.peft_config, PromptLearningConfig):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
            
            
            
            




class PeftModelForSeq2SeqLM(PeftModel):
    """
    Peft model for Seq2Seq LM

    Args:
        model ([`PreTrainedModel`]): Base transformer model
        peft_config ([`PeftConfig`]): Peft config.


    Example::

        >>> from transformers import AutoModelForSeq2SeqLM >>> from peft import PeftModelForSeq2SeqLM, get_peft_config
        >>> config = {
                'peft_type': 'LORA', 'task_type': 'SEQ_2_SEQ_LM', 'inference_mode': False, 'r': 8, 'target_modules':
                ['q', 'v'], 'lora_alpha': 32, 'lora_dropout': 0.1, 'merge_weights': False, 'fan_in_fan_out': False,
                'enable_lora': None, 'bias': 'none'
            }
        >>> peft_config = get_peft_config(config) >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>>
        peft_model = PeftModelForSeq2SeqLM(model, peft_config) >>> peft_model.print_trainable_parameters() trainable
        params: 884736 || all params: 223843584 || trainable%: 0.3952474242013566
    """

    def __init__(self, model, peft_config: PeftConfig):
        super().__init__(model, peft_config)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        self.base_model_prepare_encoder_decoder_kwargs_for_generation = (
            self.base_model._prepare_encoder_decoder_kwargs_for_generation
        )
        self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
            self._prepare_encoder_decoder_kwargs_for_generation
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if not isinstance(self.peft_config, PromptLearningConfig):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
            
            
            




class PeftModelForTokenClassification(PeftModel):
    """
    Peft model for sequence classification tasks.

    Args:
        model ([`PreTrainedModel`]): Base transformer model
        peft_config ([`PeftConfig`]): Peft config.

    **Attributes**:
        - **config** ([`PretrainedConfig`]) -- The configuration object of the base model.
        - **cls_layer_name** (`str`) -- The name of the classification layer.

    Example::

        >>> from transformers import AutoModelForSequenceClassification >>> from peft import
        PeftModelForTokenClassification, get_peft_config >>> config = {
                'peft_type': 'PREFIX_TUNING', 'task_type': 'TOKEN_CLS', 'inference_mode': False, 'num_virtual_tokens':
                20, 'token_dim': 768, 'num_transformer_submodules': 1, 'num_attention_heads': 12, 'num_layers': 12,
                'encoder_hidden_size': 768, 'prefix_projection': False, 'postprocess_past_key_value_function': None
            }
        >>> peft_config = get_peft_config(config) >>> model =
        AutoModelForTokenClassification.from_pretrained("bert-base-cased") >>> peft_model =
        PeftModelForTokenClassification(model, peft_config) >>> peft_model.print_trainable_parameters() trainable
        params: 370178 || all params: 108680450 || trainable%: 0.3406113979101117
    """

    def __init__(self, model, peft_config: PeftConfig):
        super().__init__(model, peft_config)
        self.modules_to_save = ["classifier", "score"]

        for name, _ in self.base_model.named_children():
            if any(module_name in name for module_name in self.modules_to_save):
                self.cls_layer_name = name
                break

        # to make sure classifier layer is trainable
        _set_trainable(self)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not isinstance(self.peft_config, PromptLearningConfig):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )