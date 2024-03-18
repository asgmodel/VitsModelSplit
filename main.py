import os
import numpy as np
from datasets import DatasetDict,Dataset
from Arguments import DataTrainingArguments, ModelArguments, VITSTrainingArguments
from VitsModelSplit import VitsModelSplit
from transformers import AutoTokenizer,HfArgumentParser
from Trainer import vits_trainin   

#...............................................................................

def getDataset(data_args):
    
    dataset = DatasetDict.load_from_disk(data_args.dataset_name)
    train_dataset = dataset[data_args.train_split_name].select(range(5))
    eval_dataset  = dataset[data_args.eval_split_name].select(range(5))
    
    
    return train_dataset,eval_dataset
       
#.....................

# huggingface key => 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
# wandb key => 6b9663a47062735193b9298ae35a15f982f92a9a

if __name__=="__main__":
    
    

    model = VitsModelSplit.from_pretrained("facebook/mms-tts-ara",cache_dir="./")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ara",cache_dir="./")
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, VITSTrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath('finetune_config_ara.json'))
    
    train_dataset,eval_dataset = getDataset(data_args)

    
    vits_trainin(
        model = model,
        tokenizer = tokenizer,
        model_args = model_args,
        data_args = data_args,
        training_args = training_args,
        train_dataset = train_dataset,
        eval_dataset =  eval_dataset
        )
    
    

    
  