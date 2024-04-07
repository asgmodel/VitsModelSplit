import os
import shutil
import tempfile

import numpy as np
import tqdm
import wandb
from data_collator import DataCollatorTTSWithPadding
from discriminator import VitsDiscriminator

from transformers import VitsModel
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

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

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
    
    send_example_telemetry("run_vits_finetuning", model_args, data_args)
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_seed(training_args.seed)

    config = model.config
    feature_extractor = VitsFeatureExtractor()
    forward_attention_mask = True
    
    
    # save feature extractor, tokenizer and config
    with training_args.main_process_first(desc="save_feature_extractor_tokenizer_config"):
         # only the main process saves them
         if is_main_process(training_args.local_rank):
            feature_extractor.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            config.save_pretrained(training_args.output_dir)
    
    # apply weight norms decoder,flows
    with training_args.main_process_first(desc="apply_weight_norm"):
        model.decoder.apply_weight_norm()
        for flow in model.flow.flows:
            torch.nn.utils.weight_norm(flow.conv_pre)
            torch.nn.utils.weight_norm(flow.conv_post)
    
    # tokenizer full generation sample text
    with training_args.main_process_first(desc="full_generation_sample_text"):
        input_str = data_args.full_generation_sample_text
        full_generation_sample = tokenizer(input_str, return_tensors="pt")
    
    # Define data collator
    data_collator = DataCollatorTTSWithPadding(
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        forward_attention_mask=forward_attention_mask,
    )
    
    # Define accelerator 
    logging_dir = os.path.join(training_args.output_dir, training_args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=training_args.output_dir,
                                                      logging_dir=logging_dir
                                                      )
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        log_with=training_args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )
    
    if training_args.per_device_train_batch_size:
        per_device_train_batch_size = training_args.per_device_train_batch_size
    else:
        per_device_train_batch_size = training_args.per_device_train_batch_size
        
    total_batch_size = (per_device_train_batch_size * 
                        accelerator.num_processes *
                        training_args.gradient_accumulation_steps
                        )
    
    # Define train_dataloader and eval_dataloader
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=training_args.per_device_train_batch_size,
            num_workers=training_args.dataloader_num_workers,
            sampler=None,
        )
    
    eval_dataloader = None
    if training_args.do_eval:
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            shuffle=False,
            collate_fn=data_collator,
            batch_size=training_args.per_device_eval_batch_size,
            num_workers=training_args.dataloader_num_workers,
            sampler=None,
        )
    
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if training_args.max_steps == -1:
        training_args.max_steps = training_args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        training_args.max_steps = int(training_args.num_train_epochs * num_update_steps_per_epoch)
    # Afterwards we recalculate our number of training epochs
    training_args.num_train_epochs = math.ceil(training_args.max_steps / num_update_steps_per_epoch)
    
    
    # hack to be able to train on multiple device
    with tempfile.TemporaryDirectory() as tmpdirname:
        model.discriminator.save_pretrained(tmpdirname)
        discriminator = VitsDiscriminator.from_pretrained(tmpdirname)
        for disc in discriminator.discriminators:
            disc.apply_weight_norm()
    del model.discriminator
    
    
    # init gen_optimizer, gen_lr_scheduler, disc_optimizer, dics_lr_scheduler
    gen_optimizer = torch.optim.AdamW(
        model.parameters(),
        training_args.learning_rate,
        betas=[training_args.adam_beta1, training_args.adam_beta2],
        eps=training_args.adam_epsilon,
    )

    disc_optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        training_args.learning_rate,
        betas=[training_args.adam_beta1, training_args.adam_beta2],
        eps=training_args.adam_epsilon,
    )
    
    num_warmups_steps = training_args.get_warmup_steps(training_args.num_train_epochs * accelerator.num_processes)
    num_training_steps = training_args.num_train_epochs * accelerator.num_processes

    gen_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        gen_optimizer, gamma=training_args.lr_decay, last_epoch=-1
    )
    disc_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        disc_optimizer, gamma=training_args.lr_decay, last_epoch=-1
    )
        
    
    # Prepare everything with our `accelerator`.
    (
        model,
        discriminator,
        gen_optimizer,
        gen_lr_scheduler,
        disc_optimizer,
        disc_lr_scheduler,
        train_dataloader,
        eval_dataloader,
    ) = accelerator.prepare(
        model,
        discriminator,
        gen_optimizer,
        gen_lr_scheduler,
        disc_optimizer,
        disc_lr_scheduler,
        train_dataloader,
        eval_dataloader,
    )
    
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = training_args.to_sanitized_dict()
        accelerator.init_trackers(data_args.project_name, tracker_config)
    
    
    # ......................Train!..................................
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {training_args.max_steps}")
    
    progress_bar = tqdm(
        range(0, training_args.max_steps),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    #.......................loop training............................
    for epoch in range(0, training_args.num_train_epochs):
        
        # keep track of train losses
        train_losses = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        disc_lr_scheduler.step()
        gen_lr_scheduler.step()
        
        for step, batch in enumerate(train_dataloader):
            
            print(f"TRAINIG - batch {step}, process{accelerator.process_index}, 
                  waveform {(batch['waveform'].shape)}, 
                  tokens {(batch['input_ids'].shape)}... ")
            
            with accelerator.accumulate(model, discriminator):
                
                # forward through model
                model_outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    labels_attention_mask=batch["labels_attention_mask"],
                    speaker_id=batch["speaker_id"],
                    encoder_output = batch['text_encoder_output'],
                    return_dict=True,
                    monotonic_alignment_function=None,
                )
                
                mel_scaled_labels = batch["mel_scaled_input_features"]
                mel_scaled_target = model.slice_segments(mel_scaled_labels,
                                                         model_outputs.ids_slice, 
                                                         model.segment_size
                                                         )
                mel_scaled_generation = feature_extractor._torch_extract_fbank_features(
                    model_outputs.waveform.squeeze(1)
                )[1]

                target_waveform = batch["waveform"].transpose(1, 2)
                target_waveform = model.slice_segments(
                    target_waveform,
                    model_outputs.ids_slice * feature_extractor.hop_length,
                    model.config.segment_size
                )
                
            
            
            

        

    
    

    
    
    
