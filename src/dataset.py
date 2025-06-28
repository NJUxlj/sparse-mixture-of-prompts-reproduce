import os
import sys
if "sparse-mixture-of-prompts-reproduce" not in os.getcwd():
    os.chdir("sparse-mixture-of-prompts-reproduce")
sys.path.append(os.getcwd())
import random
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers.data.datasets import *
from datasets import load_dataset
from src.superglue_loader import get_superglue



class SuperGlueData(Dataset):
    def __init__(self, dataset_name, split, tokenizer, max_length=512, text_to_text=False):
        super().__init__()
        # ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc']
        self.dataset = []
        self.split = split
        self.data_list, self.num_labels = get_superglue(dataset_name, split, text_to_text)     
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_to_text = text_to_text


    def create_dataset(self, debug=False):
        zeros, ones = 0, 0
        if debug:
            self.data_list = self.data_list[:64]

        for i, data in tqdm(enumerate(self.data_list), total=len(self.data_list), desc="Formatting dataset", ncols=100):
            if len(data) == 2:
                input_seq, label = data   # text-to-text

            elif len(data) == 3:
                input_seq, label, _ = data
            
            input_ids, segment_ids, label= self.formatting(input_seq, label)

            if input_ids is not None:
                self.dataset.append({"ids": i,
                                    "input_ids": input_ids,
                                    "segment_ids": segment_ids,
                                    "label": label})
                
        random.shuffle(self.dataset)


    def formatting(self, input_seq, label):
        if type(input_seq) == tuple:
            s0, s1 = input_seq

            if type(s0) == tuple:
                s00, s01 = s0