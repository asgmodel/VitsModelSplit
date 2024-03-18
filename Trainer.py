import os
import shutil
import tempfile

import numpy as np
import tqdm
import wandb
from DataCollatorTTSWithPadding import DataCollatorTTSWithPadding
from VitsDiscriminator import VitsDiscriminator

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
from feature_extraction_vits import VitsFeatureExtractor
from transformers.optimization import get_scheduler

from plot import plot_alignment_to_numpy, plot_spectrogram_to_numpy

#.............................................

if is_wandb_available():
    import wandb

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
logger = logging.getLogger(__name__)
#.............................................

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    real_losses = 0
    generated_losses = 0
    for disc_real, disc_generated in zip(disc_real_outputs, disc_generated_outputs):
        real_loss = torch.mean((1 - disc_real) ** 2)
        generated_loss = torch.mean(disc_generated**2)
        loss += real_loss + generated_loss
        real_losses += real_loss
        generated_losses += generated_loss

    return loss, real_losses, generated_losses


def feature_loss(feature_maps_real, feature_maps_generated):
    loss = 0
    for feature_map_real, feature_map_generated in zip(feature_maps_real, feature_maps_generated):
        for real, generated in zip(feature_map_real, feature_map_generated):
            real = real.detach()
            loss += torch.mean(torch.abs(real - generated))

    return loss * 2


def generator_loss(disc_outputs):
    total_loss = 0
    gen_losses = []
    for disc_output in disc_outputs:
        disc_output = disc_output
        loss = torch.mean((1 - disc_output) ** 2)
        gen_losses.append(loss)
        total_loss += loss

    return total_loss, gen_losses


def kl_loss(prior_latents, posterior_log_variance, prior_means, prior_log_variance, labels_mask):
    """
    z_p, logs_q: [b, h, t_t]
    prior_means, prior_log_variance: [b, h, t_t]
    """

    kl = prior_log_variance - posterior_log_variance - 0.5
    kl += 0.5 * ((prior_latents - prior_means) ** 2) * torch.exp(-2.0 * prior_log_variance)
    kl = torch.sum(kl * labels_mask)
    loss = kl / torch.sum(labels_mask)
    return loss


def log_on_trackers(
    trackers,
    generated_audio,
    generated_attn,
    generated_spec,
    target_spec,
    full_generation_waveform,
    epoch,
    sampling_rate,
):
    max_num_samples = min(len(generated_audio), 50)
    generated_audio = generated_audio[:max_num_samples]
    generated_attn = generated_attn[:max_num_samples]
    generated_spec = generated_spec[:max_num_samples]
    target_spec = target_spec[:max_num_samples]

    for tracker in trackers:
        if tracker.name == "tensorboard":
            for cpt, audio in enumerate(generated_audio):
                tracker.writer.add_audio(f"train_step_audio_{cpt}", audio[None, :], epoch, sample_rate=sampling_rate)

            for cpt, audio in enumerate(full_generation_waveform):
                tracker.writer.add_audio(
                    f"full_generation_sample{cpt}", audio[None, :], epoch, sample_rate=sampling_rate
                )

            tracker.writer.add_images("alignements", np.stack(generated_attn), dataformats="NHWC")
            tracker.writer.add_images("spectrogram", np.stack(generated_spec), dataformats="NHWC")
            tracker.writer.add_images("target spectrogram", np.stack(target_spec), dataformats="NHWC")
        elif tracker.name == "wandb":
            # wandb can only loads 100 audios per step
            tracker.log(
                {
                    "alignments": [wandb.Image(attn, caption=f"Audio epoch {epoch}") for attn in generated_attn],
                    "spectrogram": [wandb.Image(spec, caption=f"Audio epoch {epoch}") for spec in generated_spec],
                    "target spectrogram": [wandb.Image(spec, caption=f"Audio epoch {epoch}") for spec in target_spec],
                    "train generated audio": [
                        wandb.Audio(
                            audio[0],
                            caption=f"Audio during train step epoch {epoch}",
                            sample_rate=sampling_rate,
                        )
                        for audio in generated_audio
                    ],
                    "full generations samples": [
                        wandb.Audio(w, caption=f"Full generation sample {epoch}", sample_rate=sampling_rate)
                        for w in full_generation_waveform
                    ],
                }
            )
        else:
            logger.warn(f"audio logging not implemented for {tracker.name}")


def compute_val_metrics_and_losses(
    val_losses,
    accelerator,
    model_outputs,
    mel_scaled_generation,
    mel_scaled_target,
    batch_size,
    compute_clap_similarity=False,
):
    loss_mel = torch.nn.functional.l1_loss(mel_scaled_target, mel_scaled_generation)
    loss_kl = kl_loss(
        model_outputs.prior_latents,
        model_outputs.posterior_log_variances,
        model_outputs.prior_means,
        model_outputs.prior_log_variances,
        model_outputs.labels_padding_mask,
    )

    losses_mel_kl = loss_mel + loss_kl

    losses = torch.stack([loss_mel, loss_kl, losses_mel_kl])
    losses = accelerator.gather(losses.repeat(batch_size, 1)).mean(0)

    for key, loss in zip(["val_loss_mel", "val_loss_kl", "val_loss_mel_kl"], losses):
        val_losses[key] = val_losses.get(key, 0) + loss.item()

    return val_losses


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
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.enable_default_handler()
    # transformers.utils.logging.enable_explicit_format()
    # # logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    # if is_main_process(training_args.local_rank):
    #     transformers.utils.logging.set_verbosity_info()    

    
    
     
    set_seed(training_args.seed)
    
 
    
    config = model.config
    feature_extractor = VitsFeatureExtractor()
    
    forward_attention_mask = True
    
    
    with training_args.main_process_first(desc="apply_weight_norm"):
        # apply weight norms
        model.decoder.apply_weight_norm()
        for flow in model.flow.flows:
            torch.nn.utils.weight_norm(flow.conv_pre)
            torch.nn.utils.weight_norm(flow.conv_post)
    
    
    with training_args.main_process_first():
        # only the main process saves them
        if is_main_process(training_args.local_rank):
            # save feature extractor, tokenizer and config
            feature_extractor.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            config.save_pretrained(training_args.output_dir)
    
    
    data_collator = DataCollatorTTSWithPadding(
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        forward_attention_mask=forward_attention_mask,
    )

    with training_args.main_process_first():
        input_str = data_args.full_generation_sample_text
        full_generation_sample = tokenizer(input_str, return_tensors="pt")
    
  
    project_name = data_args.project_name
    logging_dir = os.path.join(training_args.output_dir, training_args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=training_args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        log_with=training_args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    per_device_train_batch_size = (
        training_args.per_device_train_batch_size if training_args.per_device_train_batch_size else 1
    )
    total_batch_size = (
        per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps
    )

    num_speakers = model.config.num_speakers
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
        
    
    train_dataloader = None
    if training_args.do_train:
        sampler = (
            LengthGroupedSampler(
                batch_size=per_device_train_batch_size,
                dataset=train_dataset,
                lengths=train_dataset["tokens_input_length"],
            )
            if training_args.group_by_length
            else None
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=False,#not training_args.group_by_length,
            collate_fn=data_collator,
            batch_size=training_args.per_device_train_batch_size,
            num_workers=training_args.dataloader_num_workers,
            sampler=sampler,
        )

    eval_dataloader = None
    if training_args.do_eval:
        eval_sampler = (
            LengthGroupedSampler(
                batch_size=training_args.per_device_eval_batch_size,
                dataset=eval_dataset,
                lengths=eval_dataset["tokens_input_length"],
            )
            if training_args.group_by_length
            else None
        )

        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            shuffle=False,
            collate_fn=data_collator,
            batch_size=training_args.per_device_eval_batch_size,
            num_workers=training_args.dataloader_num_workers,
            sampler=eval_sampler,
        )

    model_segment_size = model.segment_size
    config_segment_size = model.config.segment_size
    sampling_rate = model.config.sampling_rate
    
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
        accelerator.init_trackers(project_name, tracker_config)
    
    
    
    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {training_args.max_steps}")
    global_step = 0
    first_epoch = 0
    
    
    
    # Potentially load in the weights and states from a previous save
    if training_args.resume_from_checkpoint:
        if training_args.resume_from_checkpoint != "latest":
            path = os.path.basename(training_args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(training_args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{training_args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            training_args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(training_args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0


    
    #.......................loop training............................

    for epoch in range(first_epoch, training_args.num_train_epochs):
        # keep track of train losses
        train_losses = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        disc_lr_scheduler.step()
        gen_lr_scheduler.step()
        
        for step, batch in enumerate(train_dataloader):
            print(f"TRAINIG - batch {step}, process{accelerator.process_index}, waveform {(batch['waveform'].shape)}, tokens {(batch['input_ids'].shape)}... ")
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
                mel_scaled_target = model.slice_segments(mel_scaled_labels, model_outputs.ids_slice, model_segment_size)
                mel_scaled_generation = feature_extractor._torch_extract_fbank_features(
                    model_outputs.waveform.squeeze(1)
                )[1]

                target_waveform = batch["waveform"].transpose(1, 2)
                target_waveform = model.slice_segments(
                    target_waveform, model_outputs.ids_slice * feature_extractor.hop_length, config_segment_size
                )

                # -----------------------
                #  Train Discriminator
                # -----------------------
            
                discriminator_target, _ = discriminator(target_waveform)
                discriminator_candidate, _ = discriminator(model_outputs.waveform.detach())

                loss_disc, loss_real_disc, loss_fake_disc = discriminator_loss(
                    discriminator_target, discriminator_candidate
                )

                # backpropagate
                accelerator.backward(loss_disc * training_args.weight_disc)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(discriminator.parameters(), training_args.max_grad_norm)
                disc_optimizer.step()
                if not training_args.do_step_schedule_per_epoch:
                    disc_lr_scheduler.step()
                disc_optimizer.zero_grad()

                # -----------------------
                #  Train Generator
                # -----------------------
                
                _, fmaps_target = discriminator(target_waveform)
                discriminator_candidate, fmaps_candidate = discriminator(model_outputs.waveform)

                loss_duration = torch.sum(model_outputs.log_duration)
                loss_mel = torch.nn.functional.l1_loss(mel_scaled_target, mel_scaled_generation)
                loss_kl = kl_loss(
                    model_outputs.prior_latents,
                    model_outputs.posterior_log_variances,
                    model_outputs.prior_means,
                    model_outputs.prior_log_variances,
                    model_outputs.labels_padding_mask,
                )
                loss_fmaps = feature_loss(fmaps_target, fmaps_candidate)
                loss_gen, losses_gen = generator_loss(discriminator_candidate)

                total_generator_loss = (
                    loss_duration * training_args.weight_duration
                    + loss_mel * training_args.weight_mel
                    + loss_kl * training_args.weight_kl
                    + loss_fmaps * training_args.weight_fmaps
                    + loss_gen * training_args.weight_gen
                )

                # backpropagate
                accelerator.backward(total_generator_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
                gen_optimizer.step()
                if not training_args.do_step_schedule_per_epoch:
                    gen_lr_scheduler.step()
                gen_optimizer.zero_grad()

                # update and gather losses
                losses = torch.stack(
                    [
                        # for fair comparison, don't use weighted loss
                        loss_duration + loss_mel + loss_kl + loss_fmaps + loss_gen,
                        loss_duration,
                        loss_mel,
                        loss_kl,
                        loss_fmaps,
                        loss_gen,
                        loss_disc,
                        loss_real_disc,
                        loss_fake_disc,
                    ]
                )
                losses = accelerator.gather(losses.repeat(per_device_train_batch_size, 1)).mean(0)

                train_losses = [
                    l + losses[i].item() / training_args.gradient_accumulation_steps
                    for (i, l) in enumerate(train_losses)
                ]

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                (
                    train_summed_losses,
                    train_loss_duration,
                    train_loss_mel,
                    train_loss_kl,
                    train_loss_fmaps,
                    train_loss_gen,
                    train_loss_disc,
                    train_loss_real_disc,
                    train_loss_fake_disc,
                ) = train_losses
                
                global_step += 1
                accelerator.log(
                    {
                        "train_summed_losses": train_summed_losses,
                        "train_loss_disc": train_loss_disc,
                        "train_loss_real_disc": train_loss_real_disc,
                        "train_loss_fake_disc": train_loss_fake_disc,
                        "train_loss_duration": train_loss_duration,
                        "train_loss_mel": train_loss_mel,
                        "train_loss_kl": train_loss_kl,
                        "train_loss_fmaps": train_loss_fmaps,
                        "train_loss_gen": train_loss_gen,
                        "lr": disc_lr_scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )
                train_losses = [0.0 for _ in train_losses]
                
                if global_step % training_args.save_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `save_total_limit`
                        if training_args.save_total_limit is not None:
                            checkpoints = os.listdir(training_args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `save_total_limit - 1` checkpoints
                            if len(checkpoints) >= training_args.save_total_limit:
                                num_to_remove = len(checkpoints) - training_args.save_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(training_args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(training_args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "step_loss": total_generator_loss.detach().item(),
                "lr": disc_lr_scheduler.get_last_lr()[0],
                "step_loss_duration": loss_duration.detach().item(),
                "step_loss_mel": loss_mel.detach().item(),
                "step_loss_kl": loss_kl.detach().item(),
                "step_loss_fmaps": loss_fmaps.detach().item(),
                "step_loss_gen": loss_gen.detach().item(),
                "step_loss_disc": loss_disc.detach().item(),
                "step_loss_real_disc": loss_real_disc.detach().item(),
                "step_loss_fake_disc": loss_fake_disc.detach().item(),
            }
           

            if global_step >= training_args.max_steps:
                break

            eval_steps = training_args.eval_steps if training_args.eval_steps else 1
            do_eval = training_args.do_eval and (global_step % eval_steps == 0) and accelerator.sync_gradients

            if do_eval:
                logger.info("Running validation... ")
                generated_audio = []
                generated_attn = []
                generated_spec = []
                target_spec = []
                val_losses = {}
                for step, batch in enumerate(eval_dataloader):
                    print(
                        f"VALIDATION - batch {step}, process{accelerator.process_index}, waveform {(batch['waveform'].shape)}, tokens {(batch['input_ids'].shape)}... "
                    )
                    with torch.no_grad():
                        model_outputs_train = model(
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
                        mel_scaled_target = model.slice_segments(
                            mel_scaled_labels, model_outputs_train.ids_slice, model_segment_size
                        )
                        mel_scaled_generation = feature_extractor._torch_extract_fbank_features(
                            model_outputs_train.waveform.squeeze(1)
                        )[1]

                        val_losses = compute_val_metrics_and_losses(
                            val_losses,
                            accelerator,
                            model_outputs_train,
                            mel_scaled_generation,
                            mel_scaled_target,
                            per_device_train_batch_size,
                            compute_clap_similarity=False,
                        )

                    print(f"VALIDATION - batch {step}, process{accelerator.process_index}, PADDING AND GATHER... ")
                    specs = feature_extractor._torch_extract_fbank_features(model_outputs_train.waveform.squeeze(1))[0]
                    padded_attn, specs, target_specs = accelerator.pad_across_processes(
                        [model_outputs_train.attn.squeeze(1), specs, batch["labels"]], dim=1
                    )
                    padded_attn, specs, target_specs = accelerator.pad_across_processes(
                        [padded_attn, specs, target_specs], dim=2
                    )

                    generated_train_waveform, padded_attn, specs, target_specs = accelerator.gather_for_metrics(
                        [model_outputs_train.waveform, padded_attn, specs, target_specs]
                    )

                    if accelerator.is_main_process:
                        with torch.no_grad():
                            speaker_id = None if num_speakers < 2 else list(range(min(5, num_speakers)))
                            full_generation = model(**full_generation_sample.to(model.device), speaker_id=speaker_id)

                        generated_audio.append(generated_train_waveform.cpu())
                        generated_attn.append(padded_attn.cpu())
                        generated_spec.append(specs.cpu())
                        target_spec.append(target_specs.cpu())

                logger.info("Validation inference done, now evaluating... ")
                if accelerator.is_main_process:
                    generated_audio = [audio.numpy() for audio_batch in generated_audio for audio in audio_batch]
                    generated_attn = [
                        plot_alignment_to_numpy(attn.numpy()) for attn_batch in generated_attn for attn in attn_batch
                    ]
                    generated_spec = [
                        plot_spectrogram_to_numpy(attn.numpy()) for attn_batch in generated_spec for attn in attn_batch
                    ]
                    target_spec = [
                        plot_spectrogram_to_numpy(attn.numpy()) for attn_batch in target_spec for attn in attn_batch
                    ]
                    full_generation_waveform = full_generation.waveform.cpu().numpy()

                    accelerator.log(val_losses, step=global_step)

                    log_on_trackers(
                        accelerator.trackers,
                        generated_audio,
                        generated_attn,
                        generated_spec,
                        target_spec,
                        full_generation_waveform,
                        epoch,
                        sampling_rate,
                    )

                    logger.info("Validation finished... ")

                accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        epoch = training_args.num_train_epochs if training_args.num_train_epochs else 1
        eval_steps = training_args.eval_steps if training_args.eval_steps else 1

        # Run a final round of inference.
        do_eval = training_args.do_eval

        if do_eval:
            logger.info("Running final validation... ")
            generated_audio = []
            generated_attn = []
            generated_spec = []
            target_spec = []
            val_losses = {}
            for step, batch in enumerate(eval_dataloader):
                print(
                    f"VALIDATION - batch {step}, process{accelerator.process_index}, waveform {(batch['waveform'].shape)}, tokens {(batch['input_ids'].shape)}... "
                )
                with torch.no_grad():
                    model_outputs_train = model(
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
                    mel_scaled_target = model.slice_segments(
                        mel_scaled_labels, model_outputs_train.ids_slice, model_segment_size
                    )
                    mel_scaled_generation = feature_extractor._torch_extract_fbank_features(
                        model_outputs_train.waveform.squeeze(1)
                    )[1]

                    val_losses = compute_val_metrics_and_losses(
                        val_losses,
                        accelerator,
                        model_outputs_train,
                        mel_scaled_generation,
                        mel_scaled_target,
                        per_device_train_batch_size,
                        compute_clap_similarity=False,
                    )
                specs = feature_extractor._torch_extract_fbank_features(model_outputs_train.waveform.squeeze(1))[0]
                padded_attn, specs, target_specs = accelerator.pad_across_processes(
                    [model_outputs_train.attn.squeeze(1), specs, batch["labels"]], dim=1
                )
                padded_attn, specs, target_specs = accelerator.pad_across_processes(
                    [padded_attn, specs, target_specs], dim=2
                )

                generated_train_waveform, padded_attn, specs, target_specs = accelerator.gather_for_metrics(
                    [model_outputs_train.waveform, padded_attn, specs, target_specs]
                )

                if accelerator.is_main_process:
                    with torch.no_grad():
                        speaker_id = None if num_speakers < 2 else list(range(min(5, num_speakers)))
                        full_generation = model(**full_generation_sample.to(model.device), speaker_id=speaker_id)

                    generated_audio.append(generated_train_waveform.cpu())
                    generated_attn.append(padded_attn.cpu())
                    generated_spec.append(specs.cpu())
                    target_spec.append(target_specs.cpu())

            logger.info("Validation inference done, now evaluating... ")
            if accelerator.is_main_process:
                generated_audio = [audio.numpy() for audio_batch in generated_audio for audio in audio_batch]
                generated_attn = [
                    plot_alignment_to_numpy(attn.numpy()) for attn_batch in generated_attn for attn in attn_batch
                ]
                generated_spec = [
                    plot_spectrogram_to_numpy(attn.numpy()) for attn_batch in generated_spec for attn in attn_batch
                ]
                target_spec = [
                    plot_spectrogram_to_numpy(attn.numpy()) for attn_batch in target_spec for attn in attn_batch
                ]
                full_generation_waveform = full_generation.waveform.cpu().numpy()

                log_on_trackers(
                    accelerator.trackers,
                    generated_audio,
                    generated_attn,
                    generated_spec,
                    target_spec,
                    full_generation_waveform,
                    epoch,
                    sampling_rate,
                )

                accelerator.log(val_losses, step=global_step)
                logger.info("Validation finished... ")

            accelerator.wait_for_everyone()

        # unwrap, save and push final model
        model = accelerator.unwrap_model(model)
        discriminator = accelerator.unwrap_model(discriminator)

        model.discriminator = discriminator

        # add weight norms
        for disc in model.discriminator.discriminators:
            disc.remove_weight_norm()
        model.decoder.remove_weight_norm()
        for flow in model.flow.flows:
            torch.nn.utils.remove_weight_norm(flow.conv_pre)
            torch.nn.utils.remove_weight_norm(flow.conv_post)

        model.save_pretrained(training_args.output_dir)

        if training_args.push_to_hub:
            VitsModel.from_pretrained(training_args.output_dir).push_to_hub(training_args.hub_model_id)

    accelerator.end_training()



    logger.info("***** Training / Inference Done *****")
        
    
    
    
#............................................................................... 