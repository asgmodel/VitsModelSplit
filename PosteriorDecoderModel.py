import os
import sys
from typing import Optional
import numpy as np
import torch
from torch import nn
from transformers import set_seed
import wandb
import logging

from .vits_config import VitsConfig, VitsPreTrainedModel
from .feature_extraction import VitsFeatureExtractor
from .vits_output import PosteriorDecoderModelOutput
from  .dataset_features_collector import FeaturesCollectionDataset
from .posterior_encoder import VitsPosteriorEncoder
from .decoder import VitsHifiGan

class PosteriorDecoderModel(VitsPreTrainedModel):
    
    def __init__(self, config: VitsConfig):
        super().__init__(config)
        
        self.config = config
        self.posterior_encoder = VitsPosteriorEncoder(config)
        self.decoder = VitsHifiGan(config)
        
        if config.num_speakers > 1:
            self.embed_speaker = nn.Embedding(config.num_speakers, 
                                              config.speaker_embedding_size
                                              )
        self.sampling_rate = config.sampling_rate
        self.speaking_rate = config.speaking_rate
        self.noise_scale = config.noise_scale
        self.noise_scale_duration = config.noise_scale_duration
        self.segment_size = self.config.segment_size // self.config.hop_length
        
        self.post_init()
    
    
    #....................................
    
    def slice_segments(self,hidden_states, ids_str, segment_size=4):
        
        batch_size, channels, _ = hidden_states.shape
        # 1d tensor containing the indices to keep
        indices = torch.arange(segment_size).to(ids_str.device)
        # extend the indices to match the shape of hidden_states
        indices = indices.view(1, 1, -1).expand(batch_size, channels, -1)
        # offset indices with ids_str
        indices = indices + ids_str.view(-1, 1, 1)
        # gather indices
        output = torch.gather(hidden_states, dim=2, index=indices)

        return output
    
    #....................................
    
    def rand_slice_segments(self,hidden_states, sample_lengths=None, segment_size=4):
        batch_size, _, seq_len = hidden_states.size()
        if sample_lengths is None:
            sample_lengths = seq_len
        ids_str_max = sample_lengths - segment_size + 1
        ids_str = (torch.rand([batch_size]).to(device=hidden_states.device) * ids_str_max).to(dtype=torch.long)
        ret = self.slice_segments(hidden_states, ids_str, segment_size)
        
        return ret, ids_str

    #....................................
    
    def forward(
        self,
        labels: Optional[torch.FloatTensor] = None,
        labels_attention_mask: Optional[torch.Tensor] = None,
        speaker_id: Optional[int] = None,
        return_dict: Optional[bool] = None,
        ) :
        
        if self.config.num_speakers > 1 and speaker_id is not None:
            if isinstance(speaker_id, int):
                speaker_id = torch.full(size=(1,), fill_value=speaker_id, device=self.device)
            elif isinstance(speaker_id, (list, tuple, np.ndarray)):
                speaker_id = torch.tensor(speaker_id, device=self.device)

            if not ((0 <= speaker_id).all() and (speaker_id < self.config.num_speakers).all()).item():
                raise ValueError(f"Set `speaker_id` in the range 0-{self.config.num_speakers - 1}.")
            
            if not (len(speaker_id) == 1 or len(speaker_id == len(labels))):
                raise ValueError(
                    f"You passed {len(speaker_id)} `speaker_id` but you should either pass one speaker id or `batch_size` `speaker_id`."
                )

            speaker_embeddings = self.embed_speaker(speaker_id).unsqueeze(-1)
        else:
            speaker_embeddings = None
        
        
        if labels_attention_mask is not None:
            labels_padding_mask = labels_attention_mask.unsqueeze(1).float()
        else:
            labels_attention_mask = torch.ones((labels.shape[0], labels.shape[2])).float().to(self.device)
            labels_padding_mask = labels_attention_mask.unsqueeze(1)
        
        
        posterior_latents, posterior_means, posterior_log_variances = self.posterior_encoder(
            labels, labels_padding_mask, speaker_embeddings
        )
        
        label_lengths = labels_attention_mask.sum(dim=1)
        latents_slice, ids_slice = self.rand_slice_segments(posterior_latents, 
                                                            label_lengths, 
                                                            segment_size=self.segment_size
                                                            )

        waveform = self.decoder(latents_slice, speaker_embeddings)
        
        if not return_dict:
            outputs = (
                labels_padding_mask,
                posterior_latents,
                posterior_means,
                posterior_log_variances,
                ids_slice,
                waveform,
            )
            return outputs
        
        return PosteriorDecoderModelOutput(
                labels_padding_mask = labels_padding_mask,
                posterior_latents = posterior_latents,
                posterior_means = posterior_means,
                posterior_log_variances = posterior_log_variances,
                ids_slice = ids_slice,
                waveform = waveform,
        )
    
    
    
    #....................................
    
    def train(self,
              train_dataset_dir = None,
              eval_dataset_dir = None,
              feature_extractor = VitsFeatureExtractor(),
              training_args = None,
              full_generation_sample_index= 0,
              project_name = "Posterior_Decoder_Finetuning",
              wandbKey = "782b6a6e82bbb5a5348de0d3c7d40d1e76351e79",
              
              
              ):
        
        os.makedirs(training_args.output_dir,exist_ok=True)
        logger = logging.getLogger(f"{__name__} Training")
        log_level = training_args.get_process_log_level()
        logger.setLevel(log_level)
        
        wandb.login(key= wandbKey)
        wandb.init(project= project_name,config = training_args.to_dict())
        
        
        set_seed(training_args.seed)
        # Apply Weight Norm Decoder
        self.decoder.apply_weight_norm()
        # Save Config
        self.config.save_pretrained(training_args.output_dir)
        
        train_dataset = FeaturesCollectionDataset(root_dir = train_dataset_dir)
        
        # train_dataloader = torch.utils.data.DataLoader(
        #     train_dataset,
        #     shuffle=False,
        #     batch_size=training_args.per_device_train_batch_size,
        # )
        
        eval_dataset = None
        if training_args.do_eval:
            eval_dataset = FeaturesCollectionDataset(root_dir = eval_dataset_dir)
            # eval_dataloader = torch.utils.data.DataLoader(
            #     eval_dataset,
            #     shuffle=False,
            #     batch_size=training_args.per_device_eval_batch_size,
            # )
        
        
        full_generation_sample = train_dataset[full_generation_sample_index]
        
        # init optimizer, lr_scheduler 
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            training_args.learning_rate,
            betas=[training_args.adam_beta1, training_args.adam_beta2],
            eps=training_args.adam_epsilon,
        )
        
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=training_args.lr_decay, last_epoch=-1
            )
        
        
        
        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {training_args.num_train_epochs}")

        
        #.......................loop training............................
       
        global_step = 0
     
        for epoch in range(training_args.num_train_epochs):
            train_losses_sum = 0
            lr_scheduler.step()
            
            for step, batch in enumerate(train_dataset):
                
                # forward through model
                outputs = self.forward(
                    labels=batch["labels"],
                    labels_attention_mask=batch["labels_attention_mask"],
                    speaker_id=batch["speaker_id"]
                    )
                
                mel_scaled_labels = batch["mel_scaled_input_features"]
                mel_scaled_target = self.slice_segments(mel_scaled_labels, outputs.ids_slice,self.segment_size)
                mel_scaled_generation = feature_extractor._torch_extract_fbank_features(outputs.waveform.squeeze(1))[1]

                target_waveform = batch["waveform"].transpose(1, 2)
                target_waveform = self.slice_segments(
                                    target_waveform, 
                                    outputs.ids_slice * feature_extractor.hop_length, 
                                    self.config.segment_size
                                )
                
                
                # backpropagate
                
                loss_mel = torch.nn.functional.l1_loss(mel_scaled_target, mel_scaled_generation)
                loss = loss_mel.detach().item()
                train_losses_sum = train_losses_sum + loss
                loss_mel.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                print(f"TRAINIG - batch {step}, waveform {(batch['waveform'].shape)}, step_loss_mel {loss}, lr {lr_scheduler.get_last_lr()[0]}... ")
                global_step +=1
                
                # validation  
                
                do_eval = training_args.do_eval and (global_step % training_args.eval_steps == 0)
                if do_eval:
                    logger.info("Running validation... ")
                    eval_losses_sum = 0
                    for step, batch in enumerate(eval_dataset):
                        
                        with torch.no_grad():
                            outputs = self.forward(
                                labels=batch["labels"],
                                labels_attention_mask=batch["labels_attention_mask"],
                                speaker_id=batch["speaker_id"]
                                )
                        
                        mel_scaled_labels = batch["mel_scaled_input_features"]
                        mel_scaled_target = self.slice_segments(mel_scaled_labels, outputs.ids_slice,self.segment_size)
                        mel_scaled_generation = feature_extractor._torch_extract_fbank_features(outputs.waveform.squeeze(1))[1]
                        loss = loss_mel.detach().item()
                        eval_losses_sum +=loss
                        loss_mel = torch.nn.functional.l1_loss(mel_scaled_target, mel_scaled_generation)
                        print(f"VALIDATION - batch {step}, waveform {(batch['waveform'].shape)}, step_loss_mel {loss} ... ")
                        
                    
                    
                    with torch.no_grad():
                        full_generation_sample = full_generation_sample.to(self.device)
                        full_generation =self.forward(
                                labels=full_generation_sample["labels"],
                                labels_attention_mask=full_generation_sample["labels_attention_mask"],
                                speaker_id=full_generation_sample["speaker_id"]
                                )
                    
                    full_generation_waveform = full_generation.waveform.cpu().numpy()
                    
                    wandb.log({
                    "eval_losses": eval_losses_sum,
                    "full generations samples": [
                        wandb.Audio(w, caption=f"Full generation sample {epoch}", sample_rate=self.sampling_rate)
                        for w in full_generation_waveform],})
                
            wandb.log({"train_losses":train_losses_sum})
            
        # add weight norms
        self.decoder.remove_weight_norm()
        self.save_pretrained(training_args.output_dir)
        
        logger.info("Running final full generations samples... ")
        
        
        with torch.no_grad():
            full_generation_sample = full_generation_sample.to(self.device)
            full_generation = self.forward(
                    labels=full_generation_sample["labels"],
                    labels_attention_mask=full_generation_sample["labels_attention_mask"],
                    speaker_id=full_generation_sample["speaker_id"]
                    )
       
        full_generation_waveform = full_generation.waveform.cpu().numpy()
        wandb.log({"eval_losses": eval_losses_sum,
                   "full generations samples": [
                       wandb.Audio(w, caption=f"Full generation sample {epoch}",
                                   sample_rate=self.sampling_rate) for w in full_generation_waveform],
                   })
        
        
    
        logger.info("***** Training / Inference Done *****")
        
    #....................................
    
    
    
    
    #....................................
