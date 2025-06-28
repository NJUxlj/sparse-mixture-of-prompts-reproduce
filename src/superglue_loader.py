import os
import sys


'''
os.chdir(path):

这是 os 模块中的另一个函数，表示 "change directory"，即更改当前工作目录。
它会将当前工作目录切换到指定的路径 path。
'''

# 保证在脚本运行时处于项目的根目录
if "sparse-mixture-of-prompts-reproduce" not in os.getcwd():
    os.chdir("sparse-mixture-of-prompts-reproduce")
sys.path.append(os.getcwd())



import random
from tqdm import tqdm
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers.data.datasets import *
from datasets import load_dataset
from collections import defaultdict



def construct_dev_from_train_t2t(train_list, num_labels):
    pass



def boolq(text_to_text):

    path = "data/superglue/boolq"
    train = pickle.load(open(os.path.join(path, 'train.pkl'), 'rb'))
    val = pickle.load(open(os.path.join(path, 'validation.pkl'), 'rb'))

    if text_to_text: # 特点： passage 和 question 组成一个大的字符串
        # https://huggingface.co/datasets/stjokerli/TextToText_boolq/viewer/stjokerli--TextToText_boolq/train
        text_format = f"boolq passage: **passage** question: **question**"
        labels = ("False", "True")
        train_list = [(text_format.replace("**passage**", d['passage']).replace("**question**", d['question']), labels[d['label']]) for d in train]
        val_list = [(text_format.replace("**passage**", d['passage']).replace("**question**", d['question']), labels[d['label']]) for d in val]
        num_labels = len(labels)
    else: # 特点： passage 和 question 做为各自独立的字符串组成元组
        train_list = [((d['question'], d['passage']), d['label']) for d in train]
        val_list = [((d['question'], d['passage']), d['label']) for d in val]
        num_labels = max(train['label']) + 1

    return train_list, val_list, num_labels






def cb(text_to_text):
    path = "data/superglue/cb"
    train = pickle.load(open(os.path.join(path, 'train.pkl'), 'rb'))
    val = pickle.load(open(os.path.join(path, 'validation.pkl'), 'rb'))

    if text_to_text:
        pass

    else:
        pass





def copa(text_to_text):
    path = "data/superglue/copa"
    train = pickle.load(open(os.path.join(path, 'train.pkl'), 'rb'))
    val = pickle.load(open(os.path.join(path, 'validation.pkl'), 'rb'))

    if text_to_text:
        pass
    else:
        pass



def multirc(text_to_text):
    path = "data/superglue/multirc"
    train = pickle.load(open(os.path.join(path, 'train.pkl'), 'rb'))
    val = pickle.load(open(os.path.join(path, 'validation.pkl'), 'rb'))

    if text_to_text:
        text_format = "multirc question: **question** answer: **answer**. paragraph: **paragraph**"
        labels = ["False", "True"]

        train_list = [(f"question: {d['question']} answer: {d['answer']}. paragraph: {d['paragraph']}", labels[d['label']]) for d in train]
        val_list = [(f"question: {d['question']} answer: {d['answer']}. paragraph: {d['paragraph']}", labels[d['label']]) for d in val]
        num_labels = len(labels)

    else:
        train_list = [((d['paragraph'], f"{d['question']} {d['answer']}"), d['label']) for d in train]
        val_list = [((d['paragraph'], f"{d['question']} {d['answer']}"), d['label']) for d in val]
        num_labels = max(train['label']) + 1

    return train_list, val_list, num_labels







def rte(text_to_text):
    path = "data/superglue/rte"
    train = pickle.load(open(os.path.join(path, 'train.pkl'), 'rb'))
    val = pickle.load(open(os.path.join(path, 'validation.pkl'), 'rb'))



def wic(text_to_text):
    path = "data/superglue/wic"
    train = pickle.load(open(os.path.join(path, 'train.pkl'), 'rb'))
    val = pickle.load(open(os.path.join(path, 'validation.pkl'), 'rb'))


def get_superglue(data_name, split, text_to_text=False):
    # ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc']

    # Define a dictionary that maps data names to corresponding functions
    data_funcs = {'boolq': boolq, 'cb': cb, 'copa': copa, 'multirc': multirc, 'rte_superglue': rte, 'wic': wic}

    if data_name not in data_funcs:
        raise ValueError(f"Invalid data_name '{data_name}'.")

    if data_name == 'semeval':
        train_list, val_list, test_list, num_labels = data_funcs[data_name](text_to_text)

    else:
        train_list, val_list, num_labels = data_funcs[data_name](text_to_text)

        if text_to_text:

            test_list = val_list
            train_list, val_list = construct_dev_from_train_t2t(train_list, num_labels)

        else:
            val_list, test_list = construct_test_from_dev(val_list, num_labels)

    if split == "train":
        return train_list, num_labels

    elif split == "dev":
        return val_list, num_labels

    elif split == "test":
        return test_list, num_labels
    else:
        raise ValueError
