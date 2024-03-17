import os
import numpy as np
from Arguments import DataTrainingArguments, ModelArguments, VITSTrainingArguments

from VitsModelSplit import VitsModelSplit
from transformers import AutoTokenizer,HfArgumentParser

from Trainer import vits_trainin   

#...............................................................................    

if __name__=="__main__":
    
    

    model = VitsModelSplit.from_pretrained("facebook/mms-tts-ara",cache_dir="./")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ara",cache_dir="./")
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, VITSTrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath('finetune_config_ara.json'))
    
    
    vits_trainin(
        model = model,
        tokenizer = tokenizer,
        model_args = model_args,
        data_args = data_args,
        training_args = training_args,
        train_dataset = [],
        eval_dataset = []
        )
    
    

    
  