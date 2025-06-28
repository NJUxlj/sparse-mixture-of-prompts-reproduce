import os
import sys
if "SMoP" not in os.getcwd():
    os.chdir("SMoP")
sys.path.append(os.getcwd())
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'



