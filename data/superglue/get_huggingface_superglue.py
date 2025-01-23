import os
import sys
from datasets import load_dataset, list_datasets, DatasetDict

import pickle

'''

这段代码的作用是检查当前工作目录是否包含字符串 "SMoP"。如果当前工作目录不包含 "SMoP"，则将工作目录更改为 "SMoP"。

具体来说，代码的执行流程如下：

os.getcwd() 函数获取当前工作目录的路径。
if 'SMoP' not in os.getcwd(): 检查当前工作目录的路径中是否不包含字符串 "SMoP"。
如果当前工作目录不包含 "SMoP"，则执行 os.chdir("SMoP")，将工作目录更改为 "SMoP"。
这段代码通常用于确保脚本在特定的目录下运行，或者在需要时切换到特定的目录。

'''

if 'sparse-mixture-of-prompts-reproduce' not in os.getcwd():
    os.chdir("sparse-mixture-of-prompts-reproduce")

dataset_names = ['boolq', 'cb', 'copa', 'multirc', 'rte', 'wic']


for dataset in dataset_names:
    os.makedirs(f"data/superglue/{dataset}", exist_ok=True)
    data:DatasetDict = load_dataset("super_glue", dataset)
    for key in data.keys(): # train validation test
        # {key}是当前数据集划分的名称。
        path = f"data/superglue/{dataset}/{key}.pkl"

        pickle.dump(data[key], file = open(path, "wb"))


