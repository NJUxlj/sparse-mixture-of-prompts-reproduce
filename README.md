# Sparse Mixture-of-Prompts-reproduce (SMoP)
---
- Reproduce a prompt-learning technique called: Mixture-of-Prompts, from the paper 《SMoP: Towards Efficient and Effective Prompt Tuning with Sparse Mixture-of-Prompts》

![image](https://github.com/user-attachments/assets/5757d121-3c3c-4237-9234-df0035b7a98f)



##  SMoP (Sparse Mixture-of-Prompts) 
---
### Core Idea
- SMoP 是一种新的提示调优(Prompt Tuning)方法,通过使用多个短软提示来实现高效的训练和推理
- 不同于传统方法使用单个长软提示(100个token), SMoP 使用多个短软提示(5个token),每个提示专门处理数据的不同子集
- 通过**门控机制(gating mechanism)**将输入实例路由到最合适的软提示

### Details
#### 门控机制:
- 引入一个小型**线性路由器模型(linear router model)**
- 基于输入的嵌入表示决定将输入路由到哪个软提示
- 使用路由器扰动(router perturbation)技术确保**软提示之间的负载平衡**
- 
#### 参数设置:
- 通常使用4个软提示,每个长度为5个token
- 总提示长度为k·l(k为提示数量,l为每个提示长度)
- 但每次只使用长度为l的单个提示

### Advantages
效率提升:
训练时间减少14.6%
训练内存减少22.9%
推理FLOPs减少27.2%

性能提升:
T5-base上平均提升2.5%
T5-large上平均提升3.4%




## Install Requirements
---

```bash
pip install -r requirements.txt
```




## Download the SuperGlue dataset
---
- Download the SuperGLUE datasets by
```bash
python data/superglue/get_huggingface_superglue.py
```
- or use your custom dataset. In that case, you need to create your custom `Dataset` class for your dataset in `src/dataset.py` and apply mandatory changes such as importing your dataset or modifying the training script.



## Training
---
- Then, you can execute `scripts/train.py` with training arguments as follows
```bash
python scripts/train.py --lr 0.5  --batch_size 32  --epoch 50 --max_length 512  --model_name_or_path t5-base --tokenizer_name_or_path t5-base --warmup_ratio 0.06 --method prompt-routing --dataset_name rte_superglue --num_virtual_tokens 5 --num_virtual_tokens_full 20 --perturb_router True --topk 1
```



## Arguments
- `method`: The training method
  - `full`: Full model fine-tuning
  - `prompt-tuning`: Directly fine-tuning the soft prompts (from Lester et al., 2021)
  - `p-tuning`: Utilizing a reparameterization model on the soft prompts (from Liu et al, 2021)
  - `prompt-routing`: Use SMoP for training


num_virtual_tokens: The number of the soft prompt tokens attached to the input instance. No impact when the training method is full

num_virtual_tokens_full: The total number of soft prompt tokens used during training. For prompt-routing, this is different from 'num_virtual_tokens', while it is the same on other methods.

For example, if you want to use SMoP with 4 soft prompts of length 5, you need to set num_virtual_tokens as 5 and num_virtual_tokens_full as 20.
perturb_router: If True, scaled Gaussian noise (Section 2.3 of our paper) is applied during training.

topk: Number of soft prompt tokens to route each input instance. If larger than 2, the weighted sum of multiple soft prompts is applied.

## Citation
```bibtxt
@inproceedings{choi2023smop,
  title={SMoP: Towards Efficient and Effective Prompt Tuning with Sparse Mixture-of-Prompts},
  author={Choi, Joon-Young and Kim, Junho and Park, Jun-Hyung and Mok, Wing-Lam and Lee, SangKeun},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages={14306--14316},
  year={2023}
}
```


---



## Acknowledgement
---
- This implementation is largely based on the [HuggingFace PEFT](https://github.com/huggingface/peft) library.
