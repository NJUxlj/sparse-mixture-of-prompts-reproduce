import os
import sys
if "sparse-mixture-of-prompts-reproduce" not in os.getcwd():
    os.chdir("sparse-mixture-of-prompts-reproduce")
sys.path.append(os.getcwd())
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'



import random
from tqdm import tqdm
import argparse
from distutils.util import strtobool as _bool
import time
import re
import string



import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from thop import profile


# from torchviz import make_dot, make_dot_from_trace


from tensorboardX import SummaryWriter
from scipy.stats import spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, recall_score


from transformers import Trainer, TrainingArguments
from transformers import logging
from transformers import get_linear_schedule_with_warmup
from transformers import Adafactor
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer, AutoModelWithLMHead, AutoConfig
from transformers import RobertaForMultipleChoice, T5ForConditionalGeneration

from peft_models import get_peft_config, get_peft_model, PeftConfig, PeftModel, LoraConfig, PrefixTuningConfig, PromptEncoderConfig, PromptTuningConfig, PromptRoutingConfig, TaskType

from src.t5_with_prefix import T5ForConditionalGenerationWithPrefix
from src.dataset import SuperGlueData
from config.model_config import Config


# 设置 transformers 库的日志级别为 ERROR，仅显示错误信息，过滤掉警告和其他级别的日志
logging.set_verbosity_error()
# 注释中的代码用于开启 PyTorch 的自动梯度异常检测功能，开启后可以帮助调试反向传播过程中的梯度问题，当前处于注释状态
# torch.autograd.set_detect_anomaly(True)

# SEQ_2_SEQ_LM -> AutoModelForSeq2SeqLM
# SEQ_CLS -> AutoModelForSequenceClassification



def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        # 此正则表达式模式 `\b(a|an|the)\b` 用于匹配英文中的不定冠词 "a"、"an" 和定冠词 "the"。
        # `\b` 是单词边界，确保只匹配完整的单词，而不是单词的一部分。
        # `(a|an|the)` 是一个分组，使用 `|` 表示或关系，即匹配 "a" 或 "an" 或 "the"。
        # 匹配到的冠词会被替换为空格。
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    
    


def train(args):
    pass


def train_epoch(model, optimizer, scheduler):
    pass




def evaluate_epoch(model, scaler, val_dataloader, device, tokenizer, text_to_text, test):
    pass




def symmetric_KL_loss(input, target, reduction='batchmean'):
    pass



def score(metric, preds, answers,):
    pass






def prepare_dataset(dataset_name, tokenizer, batch_size, max_length, text_to_text, seed):
    superglue_dataset_names = ['boolq', 'cb', 'copa', 'multirc', 'rte_superglue', 'wic', 'semeval']
    # assert dataset_name in dataset_names
    dataset_metrics = {'multirc': 'f1a'}
    
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


