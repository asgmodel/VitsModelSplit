import math
import torch
from accelerate.utils import ProjectConfiguration, is_wandb_available, set_seed
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers.utils import send_example_telemetry
import logging
import sys
import datasets
from datasets import DatasetDict
import transformers
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.trainer_pt_utils import LengthGroupedSampler
from feature_extraction import VitsFeatureExtractor
from transformers.optimization import get_scheduler
from plot import plot_alignment_to_numpy, plot_spectrogram_to_numpy

#.............................................






#.............................................

def vits_trainin(
                model,
                tokenizer,
                model_args,
                data_args,
                training_args,
                train_dataset,
                eval_dataset,
                
                ):
    
    set_seed(training_args.seed)
    config = model.config
    feature_extractor = VitsFeatureExtractor()
    forward_attention_mask = True
    
    
    