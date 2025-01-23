# mixture-of-prompts-reproduce
Reproduce a prompt-learning technique called: Mixture-of-Prompts, from the paper 《SMoP: Towards Efficient and Effective Prompt Tuning with Sparse Mixture-of-Prompts》
---

## Contributions of the Paper

---
##  SMoP (Sparse Mixture-of-Prompts) 

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
