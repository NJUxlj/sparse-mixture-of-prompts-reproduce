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
    TaskType,
    PeftType,
    PromptLearningConfig,
    _set_trainable,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    shift_tokens_right,
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


        # 仅保存可训练的权重。调用 get_peft_model_state_dict 函数获取可训练权重的状态字典，
        # 若 kwargs 中提供了 state_dict 则使用该字典，否则使用默认逻辑获取。
        # 然后将这些可训练权重保存到指定目录下的权重文件中。
        output_state_dict = get_peft_model_state_dict(self, kwargs.get("state_dict", None))
        torch.save(output_state_dict, os.path.join(save_directory, WEIGHTS_NAME))
        
        # 保存配置文件，并在保存前将推理模式设置为 True，保存完成后恢复原推理模式。
        # 首先检查 peft_config 中的 base_model_name_or_path 是否为空，
        # 若为空，则根据 peft_config 的类型从对应的模型中获取 name_or_path 属性赋值。
        if self.peft_config.base_model_name_or_path is None:
            self.peft_config.base_model_name_or_path = (
                self.base_model.__dict__.get("name_or_path", None)
                if isinstance(self.peft_config, PromptLearningConfig)
                else self.base_model.model.__dict__.get("name_or_path", None)
            )
        
        # 保存当前的推理模式，以便后续恢复
        inference_mode = self.peft_config.inference_mode
        # 将推理模式设置为 True
        self.peft_config.inference_mode = True
        
        # 将 peft 配置保存到指定目录
        self.peft_config.save_pretrained(save_directory)
        # 恢复之前保存的推理模式
        self.peft_config.inference_mode = inference_mode


        

    @classmethod    # 可以用类名调用
    def from_pretrained(cls, model, model_id, **kwargs):
        r"""
        Args:
        Instantiate a `LoraModel` from a pretrained Lora configuration and weights.
            model (`transformers.PreTrainedModel`):
                The model to be adapted. The model should be initialized with the `from_pretrained` method. from
                `transformers` library.
            model_id (`str`):
                The name of the Lora configuration to use. Can be either:
                    - A string, the `model id` of a Lora configuration hosted inside a model repo on
                        huggingface Hub
                    - A path to a directory containing a Lora configuration file saved using the
                        `save_pretrained` method, e.g., ``./my_lora_config_directory/``.
        """ 
        from .mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING, PEFT_TYPE_TO_CONFIG_MAPPING

        # load the config
        config = PEFT_TYPE_TO_CONFIG_MAPPING[PeftConfig.from_pretrained(model_id).peft_type].from_pretrained(model_id)

        if getattr(model, "hf_device_map", None) is not None:
            remove_hook_from_submodules(model)

        if config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys():
            model = cls(model, config)
        else:
            model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type](model, config)
            
        # load weights if any
        if os.path.exists(os.path.join(model_id, WEIGHTS_NAME)):
            filename = os.path.join(model_id, WEIGHTS_NAME)
        else:
            try:
                filename = hf_hub_download(model_id, WEIGHTS_NAME)
            except:  # noqa
                raise ValueError(
                    f"Can't find weights for {model_id} in {model_id} or in the Hugging Face Hub. "
                    f"Please check that the file {WEIGHTS_NAME} is present at {model_id}."
                )

        adapters_weights = torch.load(filename)
        # load the weights into the model
        model = set_peft_model_state_dict(model, adapters_weights)
        if getattr(model, "hf_device_map", None) is not None:
            device_map = kwargs.get("device_map", "auto")
            max_memory = kwargs.get("max_memory", None)
            no_split_module_classes = model._no_split_modules
            if device_map != "sequential":
                max_memory = get_balanced_memory(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                    low_zero=(device_map == "balanced_low_0"),
                )
                
            if isinstance(device_map, str):
                device_map = infer_auto_device_map(
                    model, max_memory=max_memory, no_split_module_classes=no_split_module_classes
                )
            model = dispatch_model(model, device_map=device_map)
            hook = AlignDevicesHook(io_same_device=True)
            if model.peft_config.peft_type == PeftType.LORA:
                add_hook_to_module(model.base_model.model, hook)
            else:
                remove_hook_from_submodules(model.prompt_encoder)
                add_hook_to_module(model.base_model, hook)
        return model


    
    def _setup_prompt_encoder(self):
        num_transformer_submodules = 0
        transformer_backbone = None
        
        for name, module in self.base_model.named_children():
            for param in module.parameters():
                param.requires_grad = False
                
            if isinstance(module, PreTrainedModel):
                # Make sure to freeze Tranformers model
                if transformer_backbone is None:
                    transformer_backbone = module
                    self.transformerd_backbone_name = name
                    
                num_transformer_submodules += 1     # 含义： 表示 Transformer 模型的子模块数量，encoder, decoder 都是子模块， 所以 T5 有两个子模块， qwen只有一个

        self.peft_config.num_transformer_submodules = 2 if self.peft_config.task_type == TaskType.SEQ_2_SEQ_LM else 1


        for named_param, value in list(transformer_backbone.named_parameters()):
            if value.shape[0]==self.base_model.config.vocab_size:
                self.word_embeddings = transformer_backbone.get_submodule(named_param.replace(".weight",""))
                break
            
            
        if self.peft_config.peft_type == PeftType.PROMPT_TUNING:
            prompt_encoder = PromptEmbedding(self.peft_config, self.word_embeddings)
        elif self.peft_config.peft_type == PeftType.P_TUNING:
            prompt_encoder = PromptEncoder(self.peft_config)
        elif self.peft_config.peft_type == PeftType.PREFIX_TUNING:
            prompt_encoder = PrefixEncoder(self.peft_config)
        elif self.peft_config.peft_type == PeftType.PROMPT_MIX:
            prompt_encoder = PromptMixEmbedding(self.peft_config, self.word_embeddings)
        elif self.peft_config.peft_type == PeftType.PROMPT_ROUTING:
            prompt_encoder = PromptRoutingEmbedding(self.peft_config, self.word_embeddings)
        else:
            raise ValueError("Not supported")
        self.prompt_encoder = prompt_encoder
        
        
        try:
        # if self.peft_config.num_virtual_tokens_full is not None:
            self.prompt_tokens = torch.arange(
                self.peft_config.num_virtual_tokens_full # * self.peft_config.num_transformer_submodules
            ).long()
        except AttributeError:
            self.prompt_tokens = torch.arange(
                self.peft_config.num_virtual_tokens * self.peft_config.num_transformer_submodules
            ).long()


    def get_prompt_embedding_to_save(self):
        """
        Returns the prompt embedding to save when saving the model. Only applicable when `peft_config.peft_type !=
        PeftType.LORA`.
        """
        prompt_tokens = self.prompt_tokens.unsqueeze(0).expand(1, -1).to(self.device)
        if self.peft_config.peft_type == PeftType.PREFIX_TUNING:
            prompt_tokens = prompt_tokens[:, : self.peft_config.num_virtual_tokens]
        prompt_embeddings = self.prompt_encoder(prompt_tokens)
        return prompt_embeddings[0].detach().cpu()
        # return self.prompt_encoder



    def get_prompt(self, batch_size):
        pass


    def get_prompt_routing(self, batch_size, input_ids, inputs_embeds, attention_mask):
        '''
        作用：
        1. 输入：batch_size, input_ids, inputs_embeds, attention_mask
        2. 输出：prompts
        3. 作用：
            在输入的 input_ids 的前面加上 prompt tokens 作为前缀， 然后一起输入 prompt encoder， 
                得到一个 (batch_size, num_virtual_tokens, hidden_size) 的张量
        '''
        prompt_tokens = self.prompt_tokens.unsqueeze(0).expand(batch_size, -1).to(self.device)
        prompts = self.prompt_encoder(prompt_tokens, input_ids, inputs_embeds, attention_mask)  # shape = (batch_size, num_virtual_tokens, hidden_size)
        return prompts   




    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        
        trainable_params = 0
        all_param = 0
        
        for name, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                print(f"parameter name:{name}, param num:{param.numel()}")
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )


    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.base_model, name)



    def forward(self, *args, **kwargs):
        """
        Forward pass of the model.
        """

        if isinstance(self.peft_config, PromptLearningConfig):
            return self.base_model(*args, **kwargs)
        else:
            return self.base_model.model(*args, **kwargs)




class PeftModelForSequenceClassification(PeftModel):
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
        PeftModelForSequenceClassification, get_peft_config >>> config = {
                'peft_type': 'PREFIX_TUNING', 'task_type': 'SEQ_CLS', 'inference_mode': False, 'num_virtual_tokens':
                20, 'token_dim': 768, 'num_transformer_submodules': 1, 'num_attention_heads': 12, 'num_layers': 12,
                'encoder_hidden_size': 768, 'prefix_projection': False, 'postprocess_past_key_value_function': None
            }
        >>> peft_config = get_peft_config(config) >>> model =
        AutoModelForSequenceClassification.from_pretrained("bert-base-cased") >>> peft_model =
        PeftModelForSequenceClassification(model, peft_config) >>> peft_model.print_trainable_parameters() trainable
        params: 370178 || all params: 108680450 || trainable%: 0.3406113979101117
    """

    def __init__(self, model, peft_config:PeftConfig):
        super().__init__(model, peft_config)
        self.modules_to_save = ["classifier", "score"]

        for name, _ in self.base_model.named_children():  # 在所有的子模块中搜索到分类头
            if any(module_name in name  for module_name in self.modules_to_save):
                self.cls_layer_name = name
                break
            
            
        # to make sure classifier layer is trainable
        _set_trainable(self)
                
    


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