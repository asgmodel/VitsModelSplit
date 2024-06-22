
import numpy as np
import torch
from torch import nn
import math
from typing import Any, Callable, Optional, Tuple, Union
from torch.cuda.amp import autocast, GradScaler

from .vits_config import VitsConfig,VitsPreTrainedModel
from .flow import VitsResidualCouplingBlock
from .duration_predictor import VitsDurationPredictor, VitsStochasticDurationPredictor
from .encoder import VitsTextEncoder
from .decoder import VitsHifiGan
from .posterior_encoder import VitsPosteriorEncoder
from .discriminator import VitsDiscriminator
from .vits_output import VitsModelOutput, VitsTrainingOutput
from  .dataset_features_collector import FeaturesCollectionDataset
from .feature_extraction import VitsFeatureExtractor

import os
import sys
from typing import Optional
import tempfile
from torch.cuda.amp import autocast, GradScaler

from IPython.display import clear_output
from transformers import set_seed
import wandb
import logging
import copy
Lst=['input_ids',
 'attention_mask',
 'waveform',
 'labels',
 'labels_attention_mask',
 'mel_scaled_input_features']

def covert_cuda_batch(d):
  #return d
  for key in Lst:
      d[key]=d[key].cuda(non_blocking=True)
  # for key in d['text_encoder_output']:
  #   d['text_encoder_output'][key]=d['text_encoder_output'][key].cuda(non_blocking=True)
  for key in d['posterior_encode_output']:
    d['posterior_encode_output'][key]=d['posterior_encode_output'][key].cuda(non_blocking=True)

  return d
def generator_loss(disc_outputs):
    total_loss = 0
    gen_losses = []
    for disc_output in disc_outputs:
        disc_output = disc_output
        loss = torch.mean((1 - disc_output) ** 2)
        gen_losses.append(loss)
        total_loss += loss

    return total_loss, gen_losses

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
def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
  """
  z_p, logs_q: [b, h, t_t]
  m_p, logs_p: [b, h, t_t]
  """
  z_p = z_p.float()
  logs_q = logs_q.float()
  m_p = m_p.float()
  logs_p = logs_p.float()
  z_mask = z_mask.float()

  kl = logs_p - logs_q - 0.5
  kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
  kl = torch.sum(kl * z_mask)
  l = kl / torch.sum(z_mask)
  return l
#.............................................
# def kl_loss(prior_latents, posterior_log_variance, prior_means, prior_log_variance, labels_mask):


#   kl = prior_log_variance - posterior_log_variance - 0.5
#   kl += 0.5 * ((prior_latents - prior_means) ** 2) * torch.exp(-2.0 * prior_log_variance)
#   kl = torch.sum(kl * labels_mask)
#   loss = kl / torch.sum(labels_mask)
#   return loss

def get_state_grad_loss(k1=True,
                             mel=True,
                             duration=True,
                             generator=True,
                             discriminator=True):
                             return {'k1':k1,'mel':mel,'duration':duration,'generator':generator,'discriminator':discriminator}


def clip_grad_value_(parameters, clip_value, norm_type=2):
  if isinstance(parameters, torch.Tensor):
    parameters = [parameters]
  parameters = list(filter(lambda p: p.grad is not None, parameters))
  norm_type = float(norm_type)
  if clip_value is not None:
    clip_value = float(clip_value)

  total_norm = 0
  for p in parameters:
    param_norm = p.grad.data.norm(norm_type)
    total_norm += param_norm.item() ** norm_type
    if clip_value is not None:
      p.grad.data.clamp_(min=-clip_value, max=clip_value)
  total_norm = total_norm ** (1. / norm_type)
  return total_norm


class VitsModel(VitsPreTrainedModel):

    def __init__(self, config: VitsConfig):
        super().__init__(config)

        self.config = config
        self.text_encoder = VitsTextEncoder(config)
        self.flow = VitsResidualCouplingBlock(config)
        self.decoder = VitsHifiGan(config)



        if config.use_stochastic_duration_prediction:
            self.duration_predictor = VitsStochasticDurationPredictor(config)
        else:
            self.duration_predictor = VitsDurationPredictor(config)

        if config.num_speakers > 1:
            self.embed_speaker = nn.Embedding(config.num_speakers, config.speaker_embedding_size)

        # This is used only for training.
        self.posterior_encoder = VitsPosteriorEncoder(config)
        self.discriminator = VitsDiscriminator(config)

        # These parameters control the synthesised speech properties
        self.speaking_rate = config.speaking_rate
        self.noise_scale = config.noise_scale
        self.noise_scale_duration = config.noise_scale_duration
        self.segment_size = self.config.segment_size // self.config.hop_length

        # Initialize weights and apply final processing
        self.post_init()
        self.monotonic_alignment_function=self.monotonic_align_max_path



    #....................................
    def setMfA(self,fn):
      self.monotonic_alignment_function=fn



    def monotonic_align_max_path(self,log_likelihoods, mask):
        # used for training - awfully slow
        # an alternative is proposed in examples/pytorch/text-to-speech/run_vits_finetuning.py
        path = torch.zeros_like(log_likelihoods)

        text_length_maxs = mask.sum(1)[:, 0]
        latent_length_maxs = mask.sum(2)[:, 0]

        indexes = latent_length_maxs - 1

        max_neg_val = -1e9

        for batch_id in range(len(path)):
            index = int(indexes[batch_id].item())
            text_length_max = int(text_length_maxs[batch_id].item())
            latent_length_max = int(latent_length_maxs[batch_id].item())

            for y in range(text_length_max):
                for x in range(max(0, latent_length_max + y - text_length_max), min(latent_length_max, y + 1)):
                    if x == y:
                        v_cur = max_neg_val
                    else:
                        v_cur = log_likelihoods[batch_id, y - 1, x]
                    if x == 0:
                        if y == 0:
                            v_prev = 0.0
                        else:
                            v_prev = max_neg_val
                    else:
                        v_prev = log_likelihoods[batch_id, y - 1, x - 1]
                    log_likelihoods[batch_id, y, x] += max(v_prev, v_cur)

            for y in range(text_length_max - 1, -1, -1):
                path[batch_id, y, index] = 1
                if index != 0 and (
                    index == y or log_likelihoods[batch_id, y - 1, index] < log_likelihoods[batch_id, y - 1, index - 1]
                ):
                    index = index - 1
        return path

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

    def resize_speaker_embeddings(
        self,
        new_num_speakers: int,
        speaker_embedding_size: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = 2,
    ):
        if pad_to_multiple_of is not None:
            new_num_speakers = ((new_num_speakers + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

        # first, take care of embed_speaker
        if self.config.num_speakers <= 1:
            if speaker_embedding_size is None:
                raise ValueError(
                    "The current model had no previous speaker embedding, but `speaker_embedding_size` is not specified. Pass `speaker_embedding_size` to this method."
                )
            # create new embedding layer
            new_embeddings = nn.Embedding(
                new_num_speakers,
                speaker_embedding_size,
                device=self.device,
            )
            # initialize all new embeddings
            self._init_weights(new_embeddings)
        else:
            new_embeddings = self._get_resized_embeddings(self.embed_speaker, new_num_speakers)

        self.embed_speaker = new_embeddings

        # then take care of sub-models
        self.flow.resize_speaker_embeddings(speaker_embedding_size)
        for flow in self.flow.flows:
            self._init_weights(flow.wavenet.cond_layer)

        self.decoder.resize_speaker_embedding(speaker_embedding_size)
        self._init_weights(self.decoder.cond)

        self.duration_predictor.resize_speaker_embeddings(speaker_embedding_size)
        self._init_weights(self.duration_predictor.cond)

        self.posterior_encoder.resize_speaker_embeddings(speaker_embedding_size)
        self._init_weights(self.posterior_encoder.wavenet.cond_layer)

        self.config.num_speakers = new_num_speakers
        self.config.speaker_embedding_size = speaker_embedding_size

    #....................................

    def get_input_embeddings(self):
        return self.text_encoder.get_input_embeddings()

    #....................................

    def set_input_embeddings(self, value):
        self.text_encoder.set_input_embeddings(value)

    #....................................

    def apply_weight_norm(self):
        self.decoder.apply_weight_norm()
        self.flow.apply_weight_norm()
        self.posterior_encoder.apply_weight_norm()

    #....................................

    def remove_weight_norm(self):
        self.decoder.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.posterior_encoder.remove_weight_norm()

    #....................................

    def discriminate(self, hidden_states):
        return self.discriminator(hidden_states)

    #....................................

    def get_encoder(self):
        return self.text_encoder

    #....................................

    def _inference_forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        speaker_embeddings: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        text_encoder_output = self.text_encoder(
            input_ids=input_ids,
            padding_mask=padding_mask,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = text_encoder_output[0] if not return_dict else text_encoder_output.last_hidden_state
        hidden_states = hidden_states.transpose(1, 2)
        input_padding_mask = padding_mask.transpose(1, 2)

        prior_means = text_encoder_output[1] if not return_dict else text_encoder_output.prior_means
        prior_log_variances = text_encoder_output[2] if not return_dict else text_encoder_output.prior_log_variances

        if self.config.use_stochastic_duration_prediction:
            log_duration = self.duration_predictor(
                hidden_states,
                input_padding_mask,
                speaker_embeddings,
                reverse=True,
                noise_scale=self.noise_scale_duration,
            )
        else:
            log_duration = self.duration_predictor(hidden_states, input_padding_mask, speaker_embeddings)

        length_scale = 1.0 / self.speaking_rate
        duration = torch.ceil(torch.exp(log_duration) * input_padding_mask * length_scale)
        predicted_lengths = torch.clamp_min(torch.sum(duration, [1, 2]), 1).long()


        # Create a padding mask for the output lengths of shape (batch, 1, max_output_length)
        indices = torch.arange(predicted_lengths.max(), dtype=predicted_lengths.dtype, device=predicted_lengths.device)
        output_padding_mask = indices.unsqueeze(0) < predicted_lengths.unsqueeze(1)
        output_padding_mask = output_padding_mask.unsqueeze(1).to(input_padding_mask.dtype)

        # Reconstruct an attention tensor of shape (batch, 1, out_length, in_length)
        attn_mask = torch.unsqueeze(input_padding_mask, 2) * torch.unsqueeze(output_padding_mask, -1)
        batch_size, _, output_length, input_length = attn_mask.shape
        cum_duration = torch.cumsum(duration, -1).view(batch_size * input_length, 1)
        indices = torch.arange(output_length, dtype=duration.dtype, device=duration.device)
        valid_indices = indices.unsqueeze(0) < cum_duration
        valid_indices = valid_indices.to(attn_mask.dtype).view(batch_size, input_length, output_length)
        padded_indices = valid_indices - nn.functional.pad(valid_indices, [0, 0, 1, 0, 0, 0])[:, :-1]
        attn = padded_indices.unsqueeze(1).transpose(2, 3) * attn_mask

        # Expand prior distribution
        prior_means = torch.matmul(attn.squeeze(1), prior_means).transpose(1, 2)
        prior_log_variances = torch.matmul(attn.squeeze(1), prior_log_variances).transpose(1, 2)

        prior_latents = prior_means + torch.randn_like(prior_means) * torch.exp(prior_log_variances) * self.noise_scale
        latents = self.flow(prior_latents, output_padding_mask, speaker_embeddings, reverse=True)

        spectrogram = latents * output_padding_mask
        waveform = self.decoder(spectrogram, speaker_embeddings)
        waveform = waveform.squeeze(1)
        sequence_lengths = predicted_lengths * np.prod(self.config.upsample_rates)

        if not return_dict:
            outputs = (waveform, sequence_lengths, spectrogram) + text_encoder_output[3:]
            return outputs

        return VitsModelOutput(
            waveform=waveform,
            sequence_lengths=sequence_lengths,
            spectrogram=spectrogram,
            hidden_states=text_encoder_output.hidden_states,
            attentions=text_encoder_output.attentions,
        )

    #....................................

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        speaker_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.FloatTensor] = None,
        labels_attention_mask: Optional[torch.Tensor] = None,
        monotonic_alignment_function: Optional[Callable] = None,
    ) -> Union[Tuple[Any], VitsModelOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        monotonic_alignment_function = (
            self.monotonic_align_max_path if monotonic_alignment_function is None else monotonic_alignment_function
        )

        if attention_mask is not None:
            input_padding_mask = attention_mask.unsqueeze(-1).float()
        else:
            input_padding_mask = torch.ones_like(input_ids).unsqueeze(-1).float()

        if self.config.num_speakers > 1 and speaker_id is not None:
            if isinstance(speaker_id, int):
                speaker_id = torch.full(size=(1,), fill_value=speaker_id, device=self.device)
            elif isinstance(speaker_id, (list, tuple, np.ndarray)):
                speaker_id = torch.tensor(speaker_id, device=self.device)

            if not ((0 <= speaker_id).all() and (speaker_id < self.config.num_speakers).all()).item():
                raise ValueError(f"Set `speaker_id` in the range 0-{self.config.num_speakers - 1}.")
            if not (len(speaker_id) == 1 or len(speaker_id == len(input_ids))):
                raise ValueError(
                    f"You passed {len(speaker_id)} `speaker_id` but you should either pass one speaker id or `batch_size` `speaker_id`."
                )

            speaker_embeddings = self.embed_speaker(speaker_id).unsqueeze(-1)
        else:
            speaker_embeddings = None

        # if inference, return inference forward of VitsModel
        if labels is None:
            return self._inference_forward(
                input_ids,
                attention_mask,
                speaker_embeddings,
                output_attentions,
                output_hidden_states,
                return_dict,
                input_padding_mask,
            )

        if labels_attention_mask is not None:
            labels_padding_mask = labels_attention_mask.unsqueeze(1).float()
        else:
            labels_attention_mask = torch.ones((labels.shape[0], labels.shape[2])).float().to(self.device)
            labels_padding_mask = labels_attention_mask.unsqueeze(1)

        text_encoder_output = self.text_encoder(
            input_ids=input_ids,
            padding_mask=input_padding_mask,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = text_encoder_output[0] if not return_dict else text_encoder_output.last_hidden_state
        hidden_states = hidden_states.transpose(1, 2)
        input_padding_mask = input_padding_mask.transpose(1, 2)
        prior_means = text_encoder_output[1] if not return_dict else text_encoder_output.prior_means
        prior_log_variances = text_encoder_output[2] if not return_dict else text_encoder_output.prior_log_variances

        latents, posterior_means, posterior_log_variances = self.posterior_encoder(
            labels, labels_padding_mask, speaker_embeddings
        )
        prior_latents = self.flow(latents, labels_padding_mask, speaker_embeddings, reverse=False)

        prior_means, prior_log_variances = prior_means.transpose(1, 2), prior_log_variances.transpose(1, 2)
        with torch.no_grad():
            # negative cross-entropy

            # [batch_size, d, latent_length]
            prior_variances = torch.exp(-2 * prior_log_variances)
            # [batch_size, 1, latent_length]
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - prior_log_variances, [1], keepdim=True)
            # [batch_size, text_length, d] x [batch_size, d, latent_length] = [batch_size, text_length, latent_length]
            neg_cent2 = torch.matmul(-0.5 * (prior_latents**2).transpose(1, 2), prior_variances)
            # [batch_size, text_length, d] x [batch_size, d, latent_length] = [batch_size, text_length, latent_length]
            neg_cent3 = torch.matmul(prior_latents.transpose(1, 2), (prior_means * prior_variances))
            # [batch_size, 1, latent_length]
            neg_cent4 = torch.sum(-0.5 * (prior_means**2) * prior_variances, [1], keepdim=True)

            # [batch_size, text_length, latent_length]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            attn_mask = torch.unsqueeze(input_padding_mask, 2) * torch.unsqueeze(labels_padding_mask, -1)

            attn = monotonic_alignment_function(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

        durations = attn.sum(2)

        if self.config.use_stochastic_duration_prediction:
            log_duration = self.duration_predictor(
                hidden_states, input_padding_mask, speaker_embeddings, durations=durations, reverse=False
            )
            log_duration = log_duration / torch.sum(input_padding_mask)
        else:
            log_duration_padded = torch.log(durations + 1e-6) * input_padding_mask
            log_duration = self.duration_predictor(hidden_states, input_padding_mask, speaker_embeddings)
            log_duration = torch.sum((log_duration - log_duration_padded) ** 2, [1, 2]) / torch.sum(input_padding_mask)

        # expand priors
        prior_means = torch.matmul(attn.squeeze(1), prior_means.transpose(1, 2)).transpose(1, 2)
        prior_log_variances = torch.matmul(attn.squeeze(1), prior_log_variances.transpose(1, 2)).transpose(1, 2)

        label_lengths = labels_attention_mask.sum(dim=1)
        latents_slice, ids_slice = self.rand_slice_segments(latents, label_lengths, segment_size=self.segment_size)

        waveform = self.decoder(latents_slice, speaker_embeddings)

        if not return_dict:
            outputs = (
                waveform,
                log_duration,
                attn,
                ids_slice,
                input_padding_mask,
                labels_padding_mask,
                latents,
                prior_latents,
                prior_means,
                prior_log_variances,
                posterior_means,
                posterior_log_variances,
            )
            return outputs

        return VitsTrainingOutput(
            waveform=waveform,
            log_duration=log_duration,
            attn=attn,
            ids_slice=ids_slice,
            input_padding_mask=input_padding_mask,
            labels_padding_mask=labels_padding_mask,
            latents=latents,
            prior_latents=prior_latents,
            prior_means=prior_means,
            prior_log_variances=prior_log_variances,
            posterior_means=posterior_means,
            posterior_log_variances=posterior_log_variances,
        )
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

    def forward_k(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        speaker_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.FloatTensor] = None,
        labels_attention_mask: Optional[torch.Tensor] = None,
        text_encoder_output=None,
        posterior_encode_output=None,
        monotonic_alignment_function: Optional[Callable] = None,
    ) -> Union[Tuple[Any], VitsModelOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        monotonic_alignment_function = (
            self.monotonic_align_max_path if monotonic_alignment_function is None else monotonic_alignment_function
        )

        if attention_mask is not None:
            input_padding_mask = attention_mask.unsqueeze(-1).float()
        else:
            input_padding_mask = torch.ones_like(input_ids).unsqueeze(-1).float()

        if self.config.num_speakers > 1 and speaker_id is not None:
            if isinstance(speaker_id, int):
                speaker_id = torch.full(size=(1,), fill_value=speaker_id, device=self.device)
            elif isinstance(speaker_id, (list, tuple, np.ndarray)):
                speaker_id = torch.tensor(speaker_id, device=self.device)

            if not ((0 <= speaker_id).all() and (speaker_id < self.config.num_speakers).all()).item():
                raise ValueError(f"Set `speaker_id` in the range 0-{self.config.num_speakers - 1}.")
            if not (len(speaker_id) == 1 or len(speaker_id == len(input_ids))):
                raise ValueError(
                    f"You passed {len(speaker_id)} `speaker_id` but you should either pass one speaker id or `batch_size` `speaker_id`."
                )

            speaker_embeddings = self.embed_speaker(speaker_id).unsqueeze(-1)
        else:
            speaker_embeddings = None

        # if inference, return inference forward of VitsModel
        if labels is None:
            return self._inference_forward(
                input_ids,
                attention_mask,
                speaker_embeddings,
                output_attentions,
                output_hidden_states,
                return_dict,
                input_padding_mask,
            )

        if labels_attention_mask is not None:
            labels_padding_mask = labels_attention_mask.unsqueeze(1).float()
        else:
            labels_attention_mask = torch.ones((labels.shape[0], labels.shape[2])).float().to(self.device)
            labels_padding_mask = labels_attention_mask.unsqueeze(1)
        if text_encoder_output is None:
            text_encoder_output = self.text_encoder(
                input_ids=input_ids,
                padding_mask=input_padding_mask,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        hidden_states = text_encoder_output[0] if not return_dict else text_encoder_output.last_hidden_state
        hidden_states = hidden_states.transpose(1, 2)
        input_padding_mask = input_padding_mask.transpose(1, 2)
        prior_means = text_encoder_output[1] if not return_dict else text_encoder_output.prior_means
        prior_log_variances = text_encoder_output[2] if not return_dict else text_encoder_output.prior_log_variances
        if posterior_encode_output is None:
            latents, posterior_means, posterior_log_variances = self.posterior_encoder(
                labels, labels_padding_mask, speaker_embeddings
            )
        else:
          latents=posterior_encode_output['posterior_latents']
          posterior_means=posterior_encode_output['posterior_means']
          posterior_log_variances=posterior_encode_output['posterior_log_variances']

        prior_latents = self.flow(latents, labels_padding_mask, speaker_embeddings, reverse=False)

        prior_means, prior_log_variances = prior_means.transpose(1, 2), prior_log_variances.transpose(1, 2)
        with torch.no_grad():
            # negative cross-entropy

            # [batch_size, d, latent_length]
            prior_variances = torch.exp(-2 * prior_log_variances)
            # [batch_size, 1, latent_length]
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - prior_log_variances, [1], keepdim=True)
            # [batch_size, text_length, d] x [batch_size, d, latent_length] = [batch_size, text_length, latent_length]
            neg_cent2 = torch.matmul(-0.5 * (prior_latents**2).transpose(1, 2), prior_variances)
            # [batch_size, text_length, d] x [batch_size, d, latent_length] = [batch_size, text_length, latent_length]
            neg_cent3 = torch.matmul(prior_latents.transpose(1, 2), (prior_means * prior_variances))
            # [batch_size, 1, latent_length]
            neg_cent4 = torch.sum(-0.5 * (prior_means**2) * prior_variances, [1], keepdim=True)

            # [batch_size, text_length, latent_length]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            attn_mask = torch.unsqueeze(input_padding_mask, 2) * torch.unsqueeze(labels_padding_mask, -1)

            attn = monotonic_alignment_function(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

        durations = attn.sum(2)

        if self.config.use_stochastic_duration_prediction:
            log_duration = self.duration_predictor(
                hidden_states, input_padding_mask, speaker_embeddings, durations=durations, reverse=False
            )
            log_duration = log_duration / torch.sum(input_padding_mask)
        else:
            log_duration_padded = torch.log(durations + 1e-6) * input_padding_mask
            log_duration = self.duration_predictor(hidden_states, input_padding_mask, speaker_embeddings)
            log_duration = torch.sum((log_duration - log_duration_padded) ** 2, [1, 2]) / torch.sum(input_padding_mask)

        # expand priors
        prior_means = torch.matmul(attn.squeeze(1), prior_means.transpose(1, 2)).transpose(1, 2)
        prior_log_variances = torch.matmul(attn.squeeze(1), prior_log_variances.transpose(1, 2)).transpose(1, 2)

        label_lengths = labels_attention_mask.sum(dim=1)
        latents_slice, ids_slice = self.rand_slice_segments(latents, label_lengths, segment_size=self.segment_size)

        waveform = self.decoder(latents_slice, speaker_embeddings)

        if not return_dict:
            outputs = (
                waveform,
                log_duration,
                attn,
                ids_slice,
                input_padding_mask,
                labels_padding_mask,
                latents,
                prior_latents,
                prior_means,
                prior_log_variances,
                posterior_means,
                posterior_log_variances,
            )
            return outputs

        return VitsTrainingOutput(
            waveform=waveform,
            log_duration=log_duration,
            attn=attn,
            ids_slice=ids_slice,
            input_padding_mask=input_padding_mask,
            labels_padding_mask=labels_padding_mask,
            latents=latents,
            prior_latents=prior_latents,
            prior_means=prior_means,
            prior_log_variances=prior_log_variances,
            posterior_means=posterior_means,
            posterior_log_variances=posterior_log_variances,
        )

    def trainer(self,
              train_dataset_dir = None,
              eval_dataset_dir = None,
              full_generation_dir = None,
              feature_extractor = VitsFeatureExtractor(),
              training_args = None,
              full_generation_sample_index= 0,
              project_name = "Posterior_Decoder_Finetuning",
              wandbKey = "782b6a6e82bbb5a5348de0d3c7d40d1e76351e79",
              is_used_text_encoder=True,
              is_used_posterior_encode=True,
              dict_state_grad_loss=None,
              nk=1,
              path_save_model='./',
              maf=None


              ):


        os.makedirs(training_args.output_dir,exist_ok=True)
        logger = logging.getLogger(f"{__name__} Training")
        log_level = training_args.get_process_log_level()
        logger.setLevel(log_level)

        wandb.login(key= wandbKey)
        wandb.init(project= project_name,config = training_args.to_dict())
        if dict_state_grad_loss is None:
            dict_state_grad_loss=get_state_grad_loss()


        set_seed(training_args.seed)
        # Apply Weight Norm Decoder
        # self.apply_weight_norm()
        # Save Config
        self.config.save_pretrained(training_args.output_dir)

        train_dataset = FeaturesCollectionDataset(dataset_dir = train_dataset_dir,
                                                  device = self.device
                                                  )

        eval_dataset = None
        if training_args.do_eval:
            eval_dataset = FeaturesCollectionDataset(dataset_dir = eval_dataset_dir,
                                                     device = self.device
                                                     )

        full_generation_dataset = FeaturesCollectionDataset(dataset_dir = full_generation_dir,
                                                            device = self.device
                                                            )
        self.full_generation_sample = full_generation_dataset[full_generation_sample_index]

        # init optimizer, lr_scheduler

        optimizer = torch.optim.AdamW(
            self.parameters(),
            training_args.learning_rate,
            betas=[training_args.adam_beta1, training_args.adam_beta2],
            eps=training_args.adam_epsilon,
        )

         # hack to be able to train on multiple device


        # disc_optimizer = torch.optim.AdamW(
        #         self.discriminator.parameters(),
        #         training_args.learning_rate,
        #         betas=[training_args.adam_beta1, training_args.adam_beta2],
        #         eps=training_args.adam_epsilon,
        #     )
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=training_args.lr_decay, last_epoch=-1
            )
        # disc_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #                disc_optimizer, gamma=training_args.lr_decay, last_epoch=-1)


        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {training_args.num_train_epochs}")


        #.......................loop training............................

        global_step = 0

        for epoch in range(training_args.num_train_epochs):
            train_losses_sum = 0
            lr_scheduler.step()
          #  disc_lr_scheduler.step()
            print(f"  Num Epochs = {epoch}")
            if epoch%nk==0:
              print('Save checkpoints Model :',int(epoch/nk))
              self.save_pretrained(path_save_model)




            for step, batch in enumerate(train_dataset):

                # forward through model
                # outputs = self.forward(
                #     labels=batch["labels"],
                #     labels_attention_mask=batch["labels_attention_mask"],
                #     speaker_id=batch["speaker_id"]
                #     )
                #if step==10:break

                model_outputs = self.forward_k(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    labels_attention_mask=batch["labels_attention_mask"],
                    speaker_id=batch["speaker_id"],
                    text_encoder_output =None  if is_used_text_encoder else batch['text_encoder_output'],
                    posterior_encode_output=None  if is_used_posterior_encode else batch['posterior_encode_output'],
                    return_dict=True,
                    monotonic_alignment_function=maf,
                )

                mel_scaled_labels = batch["mel_scaled_input_features"]
                mel_scaled_target = self.slice_segments(mel_scaled_labels, model_outputs.ids_slice,self.segment_size)
                mel_scaled_generation = feature_extractor._torch_extract_fbank_features(model_outputs.waveform.squeeze(1))[1]

                target_waveform = batch["waveform"].transpose(1, 2)
                target_waveform = self.slice_segments(
                                    target_waveform,
                                    model_outputs.ids_slice * feature_extractor.hop_length,
                                    self.config.segment_size
                                )
                optimizer.zero_grad()

                displayloss={}
                # backpropagate
                if dict_state_grad_loss['k1']:
                    loss_kl = kl_loss(
                        model_outputs.prior_latents,
                        model_outputs.posterior_log_variances,
                        model_outputs.prior_means,
                        model_outputs.prior_log_variances,
                        model_outputs.labels_padding_mask,
                    )
                    loss_kl=loss_kl*training_args.weight_kl
                    displayloss['loss_kl']=loss_kl.detach().item()
                    #if displayloss['loss_kl']>=0:
                      #  loss_kl.backward()

                if dict_state_grad_loss['mel']:
                    loss_mel = torch.nn.functional.l1_loss(mel_scaled_target, mel_scaled_generation)
                    displayloss['loss_mel'] = loss_mel.detach().item()
                    train_losses_sum = train_losses_sum + displayloss['loss_mel']
                   # if displayloss['loss_mel']>=0:
                      #  loss_mel.backward()

                if dict_state_grad_loss['duration']:
                    loss_duration=torch.sum(model_outputs.log_duration)*training_args.weight_duration
                    displayloss['loss_duration'] = loss_duration.detach().item()
                  #  if displayloss['loss_duration']>=0:
                     #   loss_duration.backward()

                discriminator_target, fmaps_target = self.discriminator(target_waveform)
                discriminator_candidate, fmaps_candidate = self.discriminator(model_outputs.waveform.detach())
                if dict_state_grad_loss['discriminator']:

                    loss_disc, loss_real_disc, loss_fake_disc = discriminator_loss(
                        discriminator_target, discriminator_candidate
                    )

                    dk={"step_loss_disc": loss_disc.detach().item(),
                        "step_loss_real_disc": loss_real_disc.detach().item(),
                        "step_loss_fake_disc": loss_fake_disc.detach().item()}
                    displayloss['dict_loss_discriminator']=dk
                    loss_dd = loss_disc# + loss_real_disc + loss_fake_disc

                    loss_dd.backward()
                discriminator_target, fmaps_target = self.discriminator(target_waveform)

                discriminator_candidate, fmaps_candidate = self.discriminator(model_outputs.waveform.detach())

                if dict_state_grad_loss['generator']:
                    loss_fmaps = feature_loss(fmaps_target, fmaps_candidate)
                    loss_gen, losses_gen = generator_loss(discriminator_candidate)
                    loss_gen=loss_gen * training_args.weight_gen
                    displayloss['loss_gen'] = loss_gen.detach().item()
                 #   loss_gen.backward(retain_graph=True)
                    loss_fmaps=loss_fmaps * training_args.weight_fmaps
                    displayloss['loss_fmaps'] = loss_fmaps.detach().item()
                 #   loss_fmaps.backward(retain_graph=True)
                    total_generator_loss = (
                        loss_duration
                        + loss_mel
                        + loss_kl
                        + loss_fmaps
                        + loss_gen
                    )
                    total_generator_loss.backward()





                optimizer.step()




                print(f"TRAINIG - batch {step}, waveform {(batch['waveform'].shape)}, lr {lr_scheduler.get_last_lr()[0]}... ")
                print(f"display loss function  enable  :{displayloss}")

                global_step +=1

                # validation

                do_eval = training_args.do_eval and (global_step % training_args.eval_steps == 0)
                if do_eval:
                    logger.info("Running validation... ")
                    eval_losses_sum = 0
                    cc=0;
                    for step, batch in enumerate(eval_dataset):
                        break
                        if cc>2: break
                        cc+=1
                        with torch.no_grad():
                            model_outputs = self.forward(
                                input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                            labels_attention_mask=batch["labels_attention_mask"],
                            speaker_id=batch["speaker_id"],


                            return_dict=True,
                            monotonic_alignment_function=None,
                                )

                        mel_scaled_labels = batch["mel_scaled_input_features"]
                        mel_scaled_target = self.slice_segments(mel_scaled_labels, model_outputs.ids_slice,self.segment_size)
                        mel_scaled_generation = feature_extractor._torch_extract_fbank_features(model_outputs.waveform.squeeze(1))[1]
                        loss = loss_mel.detach().item()
                        eval_losses_sum +=loss

                        loss_mel = torch.nn.functional.l1_loss(mel_scaled_target, mel_scaled_generation)
                        print(f"VALIDATION - batch {step}, waveform {(batch['waveform'].shape)}, step_loss_mel {loss} ... ")



                    with torch.no_grad():
                        full_generation_sample = self.full_generation_sample
                        full_generation =self.forward(
                          input_ids =full_generation_sample["input_ids"],
                          attention_mask=full_generation_sample["attention_mask"],
                          speaker_id=full_generation_sample["speaker_id"]
                          )

                        full_generation_waveform = full_generation.waveform.cpu().numpy()

                    wandb.log({
                    "eval_losses": eval_losses_sum,
                    "full generations samples": [
                        wandb.Audio(w.reshape(-1), caption=f"Full generation sample {epoch}", sample_rate=16000)
                        for w in full_generation_waveform],})

            wandb.log({"train_losses":train_losses_sum})

        # add weight norms
        # self.remove_weight_norm()

        try:
          torch.save(self.posterior_encoder.state_dict(), os.path.join(training_args.output_dir,"posterior_encoder.pt"))
          torch.save(self.decoder.state_dict(), os.path.join(training_args.output_dir,"decoder.pt"))
        except:pass


        logger.info("Running final full generations samples... ")


        with torch.no_grad():

            full_generation_sample = self.full_generation_sample
            full_generation = self.forward(
                    input_ids=full_generation_sample["labels"],
                    attention_mask=full_generation_sample["labels_attention_mask"],
                    speaker_id=full_generation_sample["speaker_id"]
                    )

            full_generation_waveform = full_generation.waveform.cpu().numpy()

        wandb.log({"eval_losses": eval_losses_sum,
                "full generations samples": [
                    wandb.Audio(w.reshape(-1), caption=f"Full generation sample {epoch}",
                                sample_rate=16000) for w in full_generation_waveform],
                })


        logger.info("***** Training / Inference Done *****")

    #....................................


    def trainer_to_cuda(self,
              train_dataset_dir = None,
              eval_dataset_dir = None,
              full_generation_dir = None,
              feature_extractor = VitsFeatureExtractor(),
              training_args = None,
              full_generation_sample_index= 0,
              project_name = "Posterior_Decoder_Finetuning",
              wandbKey = "782b6a6e82bbb5a5348de0d3c7d40d1e76351e79",
              is_used_text_encoder=True,
              is_used_posterior_encode=True,
              dict_state_grad_loss=None,
              nk=1,
              path_save_model='./',
                        maf=None


              ):


        os.makedirs(training_args.output_dir,exist_ok=True)
        logger = logging.getLogger(f"{__name__} Training")
        log_level = training_args.get_process_log_level()
        logger.setLevel(log_level)

        wandb.login(key= wandbKey)
        wandb.init(project= project_name,config = training_args.to_dict())
        if dict_state_grad_loss is None:
            dict_state_grad_loss=get_state_grad_loss()


        set_seed(training_args.seed)
        scaler = GradScaler(enabled=training_args.fp16)

        # Apply Weight Norm Decoder
        # self.apply_weight_norm()
        # Save Config
        self.config.save_pretrained(training_args.output_dir)

        train_dataset = FeaturesCollectionDataset(dataset_dir = train_dataset_dir,
                                                  device = self.device
                                                  )

        eval_dataset = None
        if training_args.do_eval:
            eval_dataset = FeaturesCollectionDataset(dataset_dir = eval_dataset_dir,
                                                     device = self.device
                                                     )

        full_generation_dataset = FeaturesCollectionDataset(dataset_dir = full_generation_dir,
                                                            device = self.device
                                                            )
        self.full_generation_sample = full_generation_dataset[full_generation_sample_index]

        # init optimizer, lr_scheduler
        discriminator=self.discriminator
        self.discriminator=None

        optimizer = torch.optim.AdamW(
            self.parameters(),
            training_args.learning_rate,
            betas=[training_args.adam_beta1, training_args.adam_beta2],
            eps=training_args.adam_epsilon,
        )

         # hack to be able to train on multiple device


        disc_optimizer = torch.optim.AdamW(
                discriminator.parameters(),
                training_args.d_learning_rate,
                betas=[training_args.d_adam_beta1, training_args.d_adam_beta2],
                eps=training_args.adam_epsilon,
            )
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=training_args.lr_decay, last_epoch=-1
            )
        disc_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                       disc_optimizer, gamma=training_args.lr_decay, last_epoch=-1)


        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {training_args.num_train_epochs}")


        #.......................loop training............................

        global_step = 0

        for epoch in range(training_args.num_train_epochs):
            train_losses_sum = 0
            lr_scheduler.step()

            disc_lr_scheduler.step()
            print(f"  Num Epochs = {epoch}")
            if (epoch+1)%nk==0:
              clear_output()
              print('Save checkpoints Model :',int(epoch/nk))
              self.discriminator=discriminator

              self.save_pretrained(path_save_model)
              self.discriminator=None




            for step, batch in enumerate(train_dataset):

                # forward through model
                # outputs = self.forward(
                #     labels=batch["labels"],
                #     labels_attention_mask=batch["labels_attention_mask"],
                #     speaker_id=batch["speaker_id"]
                #     )
                #if step==10:break
                batch=covert_cuda_batch(batch)

                with autocast(enabled=training_args.fp16):


                  model_outputs = self.forward_k(
                      input_ids=batch["input_ids"],
                      attention_mask=batch["attention_mask"],
                      labels=batch["labels"],
                      labels_attention_mask=batch["labels_attention_mask"],
                      speaker_id=batch["speaker_id"],
                      text_encoder_output =None  if is_used_text_encoder else batch['text_encoder_output'],
                      posterior_encode_output=None  if is_used_posterior_encode else batch['posterior_encode_output'],
                      return_dict=True,
                      monotonic_alignment_function= maf,
                  )

                  mel_scaled_labels = batch["mel_scaled_input_features"]
                  mel_scaled_target = self.slice_segments(mel_scaled_labels, model_outputs.ids_slice,self.segment_size)
                  mel_scaled_generation = feature_extractor._torch_extract_fbank_features(model_outputs.waveform.squeeze(1))[1]

                  target_waveform = batch["waveform"].transpose(1, 2)
                  target_waveform = self.slice_segments(
                                      target_waveform,
                                      model_outputs.ids_slice * feature_extractor.hop_length,
                                      self.config.segment_size
                                  )

                  discriminator_target, fmaps_target = discriminator(target_waveform)
                  discriminator_candidate, fmaps_candidate = discriminator(model_outputs.waveform.detach())
                  #with autocast(enabled=False):
                  if dict_state_grad_loss['discriminator']:


                        loss_disc, loss_real_disc, loss_fake_disc = discriminator_loss(
                            discriminator_target, discriminator_candidate
                        )

                        dk={"step_loss_disc": loss_disc.detach().item(),
                            "step_loss_real_disc": loss_real_disc.detach().item(),
                            "step_loss_fake_disc": loss_fake_disc.detach().item()}
                        displayloss['dict_loss_discriminator']=dk
                        loss_dd = loss_disc# + loss_real_disc + loss_fake_disc

                  #  loss_dd.backward()

                disc_optimizer.zero_grad()
                scaler.scale(loss_dd).backward()
                scaler.unscale_(disc_optimizer )
                grad_norm_d = clip_grad_value_(discriminator.parameters(), None)
                scaler.step(disc_optimizer)


                with autocast(enabled=training_args.fp16):

                  displayloss={}
                  # backpropagate
                  if dict_state_grad_loss['k1']:
                      loss_kl = kl_loss(
                          model_outputs.prior_latents,
                          model_outputs.posterior_log_variances,
                          model_outputs.prior_means,
                          model_outputs.prior_log_variances,
                          model_outputs.labels_padding_mask,
                      )
                      loss_kl=loss_kl*training_args.weight_kl
                      displayloss['loss_kl']=loss_kl.detach().item()
                      #if displayloss['loss_kl']>=0:
                        #  loss_kl.backward()

                  if dict_state_grad_loss['mel']:
                      loss_mel = torch.nn.functional.l1_loss(mel_scaled_target, mel_scaled_generation)
                      displayloss['loss_mel'] = loss_mel.detach().item()
                      train_losses_sum = train_losses_sum + displayloss['loss_mel']
                    # if displayloss['loss_mel']>=0:
                        #  loss_mel.backward()

                  if dict_state_grad_loss['duration']:
                      loss_duration=torch.sum(model_outputs.log_duration)*training_args.weight_duration
                      displayloss['loss_duration'] = loss_duration.detach().item()
                    #  if displayloss['loss_duration']>=0:
                      #   loss_duration.backward()




                  discriminator_target, fmaps_target = discriminator(target_waveform)

                  discriminator_candidate, fmaps_candidate = discriminator(model_outputs.waveform.detach())

                  if dict_state_grad_loss['generator']:
                      loss_fmaps = feature_loss(fmaps_target, fmaps_candidate)
                      loss_gen, losses_gen = generator_loss(discriminator_candidate)
                      loss_gen=loss_gen * training_args.weight_gen
                      displayloss['loss_gen'] = loss_gen.detach().item()
                  #   loss_gen.backward(retain_graph=True)
                      loss_fmaps=loss_fmaps * training_args.weight_fmaps
                      displayloss['loss_fmaps'] = loss_fmaps.detach().item()
                  #   loss_fmaps.backward(retain_graph=True)
                      total_generator_loss = (
                          loss_duration
                          + loss_mel
                          + loss_kl
                          + loss_fmaps
                          + loss_gen
                      )
                     # total_generator_loss.backward()
                optimizer.zero_grad()
                scaler.scale(total_generator_loss).backward()
                scaler.unscale_(optimizer)
                grad_norm_g = clip_grad_value_(self.parameters(), None)
                scaler.step(optimizer)
                scaler.update()






                # optimizer.step()




                print(f"TRAINIG - batch {step}, waveform {(batch['waveform'].shape)}, lr {lr_scheduler.get_last_lr()[0]}... ")
                print(f"display loss function  enable  :{displayloss}")

                global_step +=1

                # validation

                do_eval = training_args.do_eval and (global_step % training_args.eval_steps == 0)
                if do_eval:
                    logger.info("Running validation... ")
                    eval_losses_sum = 0
                    cc=0;
                    for step, batch in enumerate(eval_dataset):
                        break
                        if cc>2: break
                        cc+=1
                        with torch.no_grad():
                            model_outputs = self.forward(
                                input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                            labels_attention_mask=batch["labels_attention_mask"],
                            speaker_id=batch["speaker_id"],


                            return_dict=True,
                            monotonic_alignment_function=None,
                                )

                        mel_scaled_labels = batch["mel_scaled_input_features"]
                        mel_scaled_target = self.slice_segments(mel_scaled_labels, model_outputs.ids_slice,self.segment_size)
                        mel_scaled_generation = feature_extractor._torch_extract_fbank_features(model_outputs.waveform.squeeze(1))[1]
                        loss = loss_mel.detach().item()
                        eval_losses_sum +=loss

                        loss_mel = torch.nn.functional.l1_loss(mel_scaled_target, mel_scaled_generation)
                        print(f"VALIDATION - batch {step}, waveform {(batch['waveform'].shape)}, step_loss_mel {loss} ... ")



                    with torch.no_grad():
                        full_generation_sample = self.full_generation_sample
                        full_generation =self.forward(
                          input_ids =full_generation_sample["input_ids"],
                          attention_mask=full_generation_sample["attention_mask"],
                          speaker_id=full_generation_sample["speaker_id"]
                          )

                        full_generation_waveform = full_generation.waveform.cpu().numpy()

                    wandb.log({
                    "eval_losses": eval_losses_sum,
                    "full generations samples": [
                        wandb.Audio(w.reshape(-1), caption=f"Full generation sample {epoch}", sample_rate=16000)
                        for w in full_generation_waveform],})

            wandb.log({"train_losses":train_losses_sum})

        # add weight norms
        # self.remove_weight_norm()

        try:
          torch.save(self.posterior_encoder.state_dict(), os.path.join(training_args.output_dir,"posterior_encoder.pt"))
          torch.save(self.decoder.state_dict(), os.path.join(training_args.output_dir,"decoder.pt"))
        except:pass


        logger.info("Running final full generations samples... ")


        with torch.no_grad():

            full_generation_sample = self.full_generation_sample
            full_generation = self.forward(
                    input_ids=full_generation_sample["labels"],
                    attention_mask=full_generation_sample["labels_attention_mask"],
                    speaker_id=full_generation_sample["speaker_id"]
                    )

            full_generation_waveform = full_generation.waveform.cpu().numpy()

        wandb.log({"eval_losses": eval_losses_sum,
                "full generations samples": [
                    wandb.Audio(w.reshape(-1), caption=f"Full generation sample {epoch}",
                                sample_rate=16000) for w in full_generation_waveform],
                })


        logger.info("***** Training / Inference Done *****")

    #....................................
    def trainer_to_cuda(self,
              train_dataset_dir = None,
              eval_dataset_dir = None,
              full_generation_dir = None,
              feature_extractor = VitsFeatureExtractor(),
              training_args = None,
              full_generation_sample_index= 0,
              project_name = "Posterior_Decoder_Finetuning",
              wandbKey = "782b6a6e82bbb5a5348de0d3c7d40d1e76351e79",
              is_used_text_encoder=True,
              is_used_posterior_encode=True,
              dict_state_grad_loss=None,
              nk=1,
              path_save_model='./',
               maf=None


              ):


        os.makedirs(training_args.output_dir,exist_ok=True)
        logger = logging.getLogger(f"{__name__} Training")
        log_level = training_args.get_process_log_level()
        logger.setLevel(log_level)

        wandb.login(key= wandbKey)
        wandb.init(project= project_name,config = training_args.to_dict())
        if dict_state_grad_loss is None:
            dict_state_grad_loss=get_state_grad_loss()


        set_seed(training_args.seed)
        scaler = GradScaler(enabled=training_args.fp16)

        # Apply Weight Norm Decoder
        # self.apply_weight_norm()
        # Save Config
        self.config.save_pretrained(training_args.output_dir)

        train_dataset = FeaturesCollectionDataset(dataset_dir = train_dataset_dir,
                                                  device = self.device
                                                  )

        eval_dataset = None
        if training_args.do_eval:
            eval_dataset = FeaturesCollectionDataset(dataset_dir = eval_dataset_dir,
                                                     device = self.device
                                                     )

        full_generation_dataset = FeaturesCollectionDataset(dataset_dir = full_generation_dir,
                                                            device = self.device
                                                            )
        self.full_generation_sample = full_generation_dataset[full_generation_sample_index]

        # init optimizer, lr_scheduler

        optimizer = torch.optim.AdamW(
            self.parameters(),
            training_args.learning_rate,
            betas=[training_args.adam_beta1, training_args.adam_beta2],
            eps=training_args.adam_epsilon,
        )

         # hack to be able to train on multiple device


        # disc_optimizer = torch.optim.AdamW(
        #         self.discriminator.parameters(),
        #         training_args.d_learning_rate,
        #         betas=[training_args.d_adam_beta1, training_args.d_adam_beta2],
        #         eps=training_args.adam_epsilon,
        #     )
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=training_args.lr_decay, last_epoch=-1
            )
        # disc_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #                disc_optimizer, gamma=training_args.lr_decay, last_epoch=-1)


        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {training_args.num_train_epochs}")


        #.......................loop training............................

        global_step = 0

        for epoch in range(training_args.num_train_epochs):
            train_losses_sum = 0
            lr_scheduler.step()

            # disc_lr_scheduler.step()
            print(f"  Num Epochs = {epoch}")
            if (epoch+1)%nk==0:
             clear_output()
             print('Save checkpoints Model :',int(epoch/nk))
             self.save_pretrained(path_save_model)

            for step, batch in enumerate(train_dataset):

                # forward through model
                # outputs = self.forward(
                #     labels=batch["labels"],
                #     labels_attention_mask=batch["labels_attention_mask"],
                #     speaker_id=batch["speaker_id"]
                #     )
                #if step==10:break
                batch=covert_cuda_batch(batch)
                displayloss={}


                with autocast(enabled=training_args.fp16):


                  model_outputs = self.forward_k(
                      input_ids=batch["input_ids"],
                      attention_mask=batch["attention_mask"],
                      labels=batch["labels"],
                      labels_attention_mask=batch["labels_attention_mask"],
                      speaker_id=batch["speaker_id"],
                      text_encoder_output =None  if is_used_text_encoder else batch['text_encoder_output'],
                      posterior_encode_output=None  if is_used_posterior_encode else batch['posterior_encode_output'],
                      return_dict=True,
                      monotonic_alignment_function=maf,
                  )

                  mel_scaled_labels = batch["mel_scaled_input_features"]
                  mel_scaled_target = self.slice_segments(mel_scaled_labels, model_outputs.ids_slice,self.segment_size)
                  mel_scaled_generation = feature_extractor._torch_extract_fbank_features(model_outputs.waveform.squeeze(1))[1]

                  target_waveform = batch["waveform"].transpose(1, 2)
                  target_waveform = self.slice_segments(
                                      target_waveform,
                                      model_outputs.ids_slice * feature_extractor.hop_length,
                                      self.config.segment_size
                                  )

                  discriminator_target, fmaps_target = self.discriminator(target_waveform)
                  discriminator_candidate, fmaps_candidate = self.discriminator(model_outputs.waveform.detach())
                  with autocast(enabled=False):
                     if dict_state_grad_loss['discriminator']:
                      #  disc_optimizer.zero_grad()

                        loss_disc, loss_real_disc, loss_fake_disc = discriminator_loss(
                            discriminator_target, discriminator_candidate
                        )

                        dk={"step_loss_disc": loss_disc.detach().item(),
                            "step_loss_real_disc": loss_real_disc.detach().item(),
                            "step_loss_fake_disc": loss_fake_disc.detach().item()}
                        displayloss['dict_loss_discriminator']=dk
                        loss_dd = loss_disc# + loss_real_disc + loss_fake_disc

                  #  loss_dd.backward()
                optimizer.zero_grad()
                # disc_optimizer.zero_grad()
                scaler.scale(loss_dd).backward()
                # scaler.unscale_(disc_optimizer)
                #grad_norm_d = clip_grad_value_(self.discriminator.parameters(), None)
                # scaler.step(disc_optimizer)


                with autocast(enabled=training_args.fp16):




                  # backpropagate




                  discriminator_target, fmaps_target = self.discriminator(target_waveform)

                  discriminator_candidate, fmaps_candidate = self.discriminator(model_outputs.waveform.detach())
                  with autocast(enabled=False):
                    if dict_state_grad_loss['k1']:
                      loss_kl = kl_loss(
                          model_outputs.prior_latents,
                          model_outputs.posterior_log_variances,
                          model_outputs.prior_means,
                          model_outputs.prior_log_variances,
                          model_outputs.labels_padding_mask,
                      )
                      loss_kl=loss_kl*training_args.weight_kl
                      displayloss['loss_kl']=loss_kl.detach().item()
                      #if displayloss['loss_kl']>=0:
                        #  loss_kl.backward()

                    if dict_state_grad_loss['mel']:
                          loss_mel = torch.nn.functional.l1_loss(mel_scaled_target, mel_scaled_generation)
                          displayloss['loss_mel'] = loss_mel.detach().item()
                          train_losses_sum = train_losses_sum + displayloss['loss_mel']
                        # if displayloss['loss_mel']>=0:
                            #  loss_mel.backward()

                    if dict_state_grad_loss['duration']:
                          loss_duration=torch.sum(model_outputs.log_duration)*training_args.weight_duration
                          displayloss['loss_duration'] = loss_duration.detach().item()
                        #  if displayloss['loss_duration']>=0:
                          #   loss_duration.backward()

                    if dict_state_grad_loss['generator']:
                        loss_fmaps = feature_loss(fmaps_target, fmaps_candidate)
                        loss_gen, losses_gen = generator_loss(discriminator_candidate)
                        loss_gen=loss_gen * training_args.weight_gen
                        displayloss['loss_gen'] = loss_gen.detach().item()
                    #   loss_gen.backward(retain_graph=True)
                        loss_fmaps=loss_fmaps * training_args.weight_fmaps
                        displayloss['loss_fmaps'] = loss_fmaps.detach().item()
                    #   loss_fmaps.backward(retain_graph=True)
                        total_generator_loss = (
                            loss_duration
                            + loss_mel
                            + loss_kl
                            + loss_fmaps
                            + loss_gen
                        )
                     # total_generator_loss.backward()
                scaler.scale(total_generator_loss).backward()
                scaler.unscale_(optimizer)
                grad_norm_g = clip_grad_value_(self.parameters(), None)
                scaler.step(optimizer)
                scaler.update()






                # optimizer.step()




                print(f"TRAINIG - batch {step},Grad G{grad_norm_g}, waveform {(batch['waveform'].shape)}, lr {lr_scheduler.get_last_lr()[0]}... ")
                print(f"display loss function  enable  :{displayloss}")

                global_step +=1

                # validation

                do_eval = training_args.do_eval and (global_step % training_args.eval_steps == 0)
                if do_eval:
                    logger.info("Running validation... ")
                    eval_losses_sum = 0
                    cc=0;
                    for step, batch in enumerate(eval_dataset):
                        break
                        if cc>2: break
                        cc+=1
                        with torch.no_grad():
                            model_outputs = self.forward(
                                input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                            labels_attention_mask=batch["labels_attention_mask"],
                            speaker_id=batch["speaker_id"],


                            return_dict=True,
                            monotonic_alignment_function=None,
                                )

                        mel_scaled_labels = batch["mel_scaled_input_features"]
                        mel_scaled_target = self.slice_segments(mel_scaled_labels, model_outputs.ids_slice,self.segment_size)
                        mel_scaled_generation = feature_extractor._torch_extract_fbank_features(model_outputs.waveform.squeeze(1))[1]
                        loss = loss_mel.detach().item()
                        eval_losses_sum +=loss

                        loss_mel = torch.nn.functional.l1_loss(mel_scaled_target, mel_scaled_generation)
                        print(f"VALIDATION - batch {step}, waveform {(batch['waveform'].shape)}, step_loss_mel {loss} ... ")



                    with torch.no_grad():
                        full_generation_sample = self.full_generation_sample
                        full_generation =self.forward(
                          input_ids =full_generation_sample["input_ids"],
                          attention_mask=full_generation_sample["attention_mask"],
                          speaker_id=full_generation_sample["speaker_id"]
                          )

                        full_generation_waveform = full_generation.waveform.cpu().numpy()

                    wandb.log({
                    "eval_losses": eval_losses_sum,
                    "full generations samples": [
                        wandb.Audio(w.reshape(-1), caption=f"Full generation sample {epoch}", sample_rate=16000)
                        for w in full_generation_waveform],})

            wandb.log({"train_losses":train_losses_sum})

        # add weight norms
        # self.remove_weight_norm()

        try:
          torch.save(self.posterior_encoder.state_dict(), os.path.join(training_args.output_dir,"posterior_encoder.pt"))
          torch.save(self.decoder.state_dict(), os.path.join(training_args.output_dir,"decoder.pt"))
        except:pass


        logger.info("Running final full generations samples... ")


        with torch.no_grad():

            full_generation_sample = self.full_generation_sample
            full_generation = self.forward(
                    input_ids=full_generation_sample["labels"],
                    attention_mask=full_generation_sample["labels_attention_mask"],
                    speaker_id=full_generation_sample["speaker_id"]
                    )

            full_generation_waveform = full_generation.waveform.cpu().numpy()

        wandb.log({"eval_losses": eval_losses_sum,
                "full generations samples": [
                    wandb.Audio(w.reshape(-1), caption=f"Full generation sample {epoch}",
                                sample_rate=16000) for w in full_generation_waveform],
                })


        logger.info("***** Training / Inference Done *****")

    #....................................

    def trainer_to(self,
              train_dataset_dir = None,
              eval_dataset_dir = None,
              full_generation_dir = None,
              feature_extractor = VitsFeatureExtractor(),
              training_args = None,
              full_generation_sample_index= 0,
              project_name = "Posterior_Decoder_Finetuning",
              wandbKey = "782b6a6e82bbb5a5348de0d3c7d40d1e76351e79",
              is_used_text_encoder=True,
              is_used_posterior_encode=True,
              dict_state_grad_loss=None,
              nk=1,
              path_save_model='./',
               maf=None


              ):


        os.makedirs(training_args.output_dir,exist_ok=True)
        logger = logging.getLogger(f"{__name__} Training")
        log_level = training_args.get_process_log_level()
        logger.setLevel(log_level)

        wandb.login(key= wandbKey)
        wandb.init(project= project_name,config = training_args.to_dict())
        if dict_state_grad_loss is None:
            dict_state_grad_loss=get_state_grad_loss()


        set_seed(training_args.seed)
        # Apply Weight Norm Decoder
        # self.apply_weight_norm()
        # Save Config
        self.config.save_pretrained(training_args.output_dir)

        train_dataset = FeaturesCollectionDataset(dataset_dir = train_dataset_dir,
                                                  device = self.device
                                                  )

        eval_dataset = None
        if training_args.do_eval:
            eval_dataset = FeaturesCollectionDataset(dataset_dir = eval_dataset_dir,
                                                     device = self.device
                                                     )

        full_generation_dataset = FeaturesCollectionDataset(dataset_dir = full_generation_dir,
                                                            device = self.device
                                                            )
        self.full_generation_sample = full_generation_dataset[full_generation_sample_index]

        # init optimizer, lr_scheduler

        optimizer = torch.optim.AdamW(
            self.parameters(),
            training_args.learning_rate,
            betas=[training_args.adam_beta1, training_args.adam_beta2],
            eps=training_args.adam_epsilon,
        )

         # hack to be able to train on multiple device


        disc_optimizer = torch.optim.AdamW(
                self.discriminator.parameters(),
                training_args.d_learning_rate,
                betas=[training_args.d_adam_beta1, training_args.d_adam_beta2],
                eps=training_args.adam_epsilon,
            )
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=training_args.lr_decay, last_epoch=-1
            )
        disc_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                       disc_optimizer, gamma=training_args.lr_decay, last_epoch=-1)


        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {training_args.num_train_epochs}")


        #.......................loop training............................

        global_step = 0

        for epoch in range(training_args.num_train_epochs):
            train_losses_sum = 0
            lr_scheduler.step()

            disc_lr_scheduler.step()
            print(f"  Num Epochs = {epoch}")
            if epoch%nk==0:
              clear_output()
              print('')
              print('Save checkpoints Model :',int(epoch/nk))
              self.save_pretrained(path_save_model)




            for step, batch in enumerate(train_dataset):

                # forward through model
                # outputs = self.forward(
                #     labels=batch["labels"],
                #     labels_attention_mask=batch["labels_attention_mask"],
                #     speaker_id=batch["speaker_id"]
                #     )
                #if step==10:break
                batch=covert_cuda_batch(batch)

                waveform,ids_slice,log_duration,prior_latents,posterior_log_variances,prior_means,prior_log_variances,labels_padding_mask=self.forward_train(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    labels_attention_mask=batch["labels_attention_mask"],
                    speaker_id=batch["speaker_id"],
                    text_encoder_output =None , #if is_used_text_encoder else batch['text_encoder_output'],
                    posterior_encode_output=batch['posterior_encode_output'] ,#  if is_used_posterior_encode else ,
                    return_dict=True,
                    monotonic_alignment_function= maf,
                )

                mel_scaled_labels = batch["mel_scaled_input_features"]
                mel_scaled_target = self.slice_segments(mel_scaled_labels, ids_slice,self.segment_size)
                mel_scaled_generation = feature_extractor._torch_extract_fbank_features(waveform.squeeze(1))[1]

                target_waveform = batch["waveform"].transpose(1, 2)
                target_waveform = self.slice_segments(
                                    target_waveform,
                                    ids_slice * feature_extractor.hop_length,
                                    self.config.segment_size
                                )



                displayloss={}
                # backpropagate
                #if dict_state_grad_loss['k1']:
                loss_kl = kl_loss(
                    prior_latents,
                    posterior_log_variances,
                    prior_means,
                   prior_log_variances,
                   labels_padding_mask,
                )
                loss_kl=loss_kl*training_args.weight_kl
                displayloss['loss_kl']=loss_kl.detach().item()
                    #if displayloss['loss_kl']>=0:
                      #  loss_kl.backward()

              #  if dict_state_grad_loss['mel']:
                loss_mel = torch.nn.functional.l1_loss(mel_scaled_target, mel_scaled_generation)
                displayloss['loss_mel'] = loss_mel.detach().item()
                train_losses_sum = train_losses_sum + displayloss['loss_mel']
                # if displayloss['loss_mel']>=0:
                      #  loss_mel.backward()

                #if dict_state_grad_loss['duration']:
                loss_duration=torch.sum(log_duration)*training_args.weight_duration
                displayloss['loss_duration'] = loss_duration.detach().item()
              #  if displayloss['loss_duration']>=0:
                     #   loss_duration.backward()

                discriminator_target, fmaps_target = self.discriminator(target_waveform)
                discriminator_candidate, fmaps_candidate = self.discriminator(waveform.detach())
                #if dict_state_grad_loss['discriminator']:
                disc_optimizer.zero_grad()

                loss_disc, loss_real_disc, loss_fake_disc = discriminator_loss(
                    discriminator_target, discriminator_candidate
                )

                dk={"step_loss_disc": loss_disc.detach().item(),
                    "step_loss_real_disc": loss_real_disc.detach().item(),
                    "step_loss_fake_disc": loss_fake_disc.detach().item()}
                displayloss['dict_loss_discriminator']=dk
                loss_dd = loss_disc# + loss_real_disc + loss_fake_disc

                loss_dd.backward()
                disc_optimizer.step()


                discriminator_target, fmaps_target = self.discriminator(target_waveform)

                discriminator_candidate, fmaps_candidate = self.discriminator(waveform.detach())
                optimizer.zero_grad()
             #   if dict_state_grad_loss['generator']:
                loss_fmaps = feature_loss(fmaps_target, fmaps_candidate)
                loss_gen, losses_gen = generator_loss(discriminator_candidate)
                loss_gen=loss_gen * training_args.weight_gen
                displayloss['loss_gen'] = loss_gen.detach().item()
              #   loss_gen.backward(retain_graph=True)
                loss_fmaps=loss_fmaps * training_args.weight_fmaps
                displayloss['loss_fmaps'] = loss_fmaps.detach().item()
              #   loss_fmaps.backward(retain_graph=True)
                total_generator_loss = (
                    loss_duration
                    + loss_mel
                    + loss_kl
                    + loss_fmaps
                    + loss_gen
                )
                total_generator_loss.backward()





                optimizer.step()




                print(f"TRAINIG - batch {step}, waveform {(batch['waveform'].shape)}, lr {lr_scheduler.get_last_lr()[0]}... ")
                print(f"display loss function  enable  :{displayloss}")

                global_step +=1

                # validation

                do_eval = training_args.do_eval and (global_step % training_args.eval_steps == 0)
                if do_eval:
                    logger.info("Running validation... ")
                    eval_losses_sum = 0
                    cc=0;
                    for step, batch in enumerate(eval_dataset):
                        break
                        if cc>2: break
                        cc+=1
                        with torch.no_grad():
                            model_outputs = self.forward(
                                input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                            labels_attention_mask=batch["labels_attention_mask"],
                            speaker_id=batch["speaker_id"],


                            return_dict=True,
                            monotonic_alignment_function=None,
                                )

                        mel_scaled_labels = batch["mel_scaled_input_features"]
                        mel_scaled_target = self.slice_segments(mel_scaled_labels, model_outputs.ids_slice,self.segment_size)
                        mel_scaled_generation = feature_extractor._torch_extract_fbank_features(model_outputs.waveform.squeeze(1))[1]
                        loss = loss_mel.detach().item()
                        eval_losses_sum +=loss

                        loss_mel = torch.nn.functional.l1_loss(mel_scaled_target, mel_scaled_generation)
                        print(f"VALIDATION - batch {step}, waveform {(batch['waveform'].shape)}, step_loss_mel {loss} ... ")



                    with torch.no_grad():
                        full_generation_sample = self.full_generation_sample
                        full_generation =self.forward(
                          input_ids =full_generation_sample["input_ids"],
                          attention_mask=full_generation_sample["attention_mask"],
                          speaker_id=full_generation_sample["speaker_id"]
                          )

                        full_generation_waveform = full_generation.waveform.cpu().numpy()

                    wandb.log({
                    "eval_losses": eval_losses_sum,
                    "full generations samples": [
                        wandb.Audio(w.reshape(-1), caption=f"Full generation sample {epoch}", sample_rate=16000)
                        for w in full_generation_waveform],})

            wandb.log({"train_losses":train_losses_sum})

        # add weight norms
        # self.remove_weight_norm()

        try:
          torch.save(self.posterior_encoder.state_dict(), os.path.join(training_args.output_dir,"posterior_encoder.pt"))
          torch.save(self.decoder.state_dict(), os.path.join(training_args.output_dir,"decoder.pt"))
        except:pass


        logger.info("Running final full generations samples... ")


        with torch.no_grad():

            full_generation_sample = self.full_generation_sample
            full_generation = self.forward(
                    input_ids=full_generation_sample["labels"],
                    attention_mask=full_generation_sample["labels_attention_mask"],
                    speaker_id=full_generation_sample["speaker_id"]
                    )

            full_generation_waveform = full_generation.waveform.cpu().numpy()

        wandb.log({"eval_losses": eval_losses_sum,
                "full generations samples": [
                    wandb.Audio(w.reshape(-1), caption=f"Full generation sample {epoch}",
                                sample_rate=16000) for w in full_generation_waveform],
                })


        logger.info("***** Training / Inference Done *****")

    #....................................




    def trainer_to_cuda1(self,
              train_dataset_dir = None,
              eval_dataset_dir = None,
              full_generation_dir = None,
              feature_extractor = VitsFeatureExtractor(),
              training_args = None,
              full_generation_sample_index= 0,
              project_name = "Posterior_Decoder_Finetuning",
              wandbKey = "782b6a6e82bbb5a5348de0d3c7d40d1e76351e79",
              is_used_text_encoder=True,
              is_used_posterior_encode=True,
              dict_state_grad_loss=None,
              nk=1,
              path_save_model='./',
                        maf=None


              ):


        os.makedirs(training_args.output_dir,exist_ok=True)
        logger = logging.getLogger(f"{__name__} Training")
        log_level = training_args.get_process_log_level()
        logger.setLevel(log_level)

        wandb.login(key= wandbKey)
        wandb.init(project= project_name,config = training_args.to_dict())
        if dict_state_grad_loss is None:
            dict_state_grad_loss=get_state_grad_loss()


        set_seed(training_args.seed)
        scaler = GradScaler(enabled=training_args.fp16)

        # Apply Weight Norm Decoder
        # self.apply_weight_norm()
        # Save Config
        self.config.save_pretrained(training_args.output_dir)

        train_dataset = FeaturesCollectionDataset(dataset_dir = train_dataset_dir,
                                                  device = self.device
                                                  )

        eval_dataset = None
        if training_args.do_eval:
            eval_dataset = FeaturesCollectionDataset(dataset_dir = eval_dataset_dir,
                                                     device = self.device
                                                     )

        full_generation_dataset = FeaturesCollectionDataset(dataset_dir = full_generation_dir,
                                                            device = self.device
                                                            )
        self.full_generation_sample = full_generation_dataset[full_generation_sample_index]

        # init optimizer, lr_scheduler
        discriminator=self.discriminator
        self.discriminator=None

        optimizer = torch.optim.AdamW(
            self.parameters(),
            training_args.learning_rate,
            betas=[training_args.adam_beta1, training_args.adam_beta2],
            eps=training_args.adam_epsilon,
        )

         # hack to be able to train on multiple device


        disc_optimizer = torch.optim.AdamW(
                discriminator.parameters(),
                training_args.d_learning_rate,
                betas=[training_args.d_adam_beta1, training_args.d_adam_beta2],
                eps=training_args.adam_epsilon,
            )
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=training_args.lr_decay, last_epoch=-1
            )
        disc_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                       disc_optimizer, gamma=training_args.lr_decay, last_epoch=-1)


        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {training_args.num_train_epochs}")


        #.......................loop training............................

        global_step = 0

        for epoch in range(training_args.num_train_epochs):
            train_losses_sum = 0
            lr_scheduler.step()

            disc_lr_scheduler.step()
            print(f"  Num Epochs = {epoch}")
            if epoch%nk==0:
              clear_output()
              print('Save checkpoints Model :',int(epoch/nk))
              self.discriminator=discriminator

              self.save_pretrained(path_save_model)
              self.discriminator=None




            for step, batch in enumerate(train_dataset):

                # forward through model
                # outputs = self.forward(
                #     labels=batch["labels"],
                #     labels_attention_mask=batch["labels_attention_mask"],
                #     speaker_id=batch["speaker_id"]
                #     )
                #if step==10:break
                batch=covert_cuda_batch(batch)
                displayloss={}

                with autocast(enabled=training_args.fp16):


                  model_outputs = self.forward_k(
                      input_ids=batch["input_ids"],
                      attention_mask=batch["attention_mask"],
                      labels=batch["labels"],
                      labels_attention_mask=batch["labels_attention_mask"],
                      speaker_id=batch["speaker_id"],
                      text_encoder_output =None  if is_used_text_encoder else batch['text_encoder_output'],
                      posterior_encode_output=None  if is_used_posterior_encode else batch['posterior_encode_output'],
                      return_dict=True,
                      monotonic_alignment_function= maf,
                  )

                  mel_scaled_labels = batch["mel_scaled_input_features"]
                  mel_scaled_target = self.slice_segments(mel_scaled_labels, model_outputs.ids_slice,self.segment_size)
                  mel_scaled_generation = feature_extractor._torch_extract_fbank_features(model_outputs.waveform.squeeze(1))[1]

                  target_waveform = batch["waveform"].transpose(1, 2)
                  target_waveform = self.slice_segments(
                                      target_waveform,
                                      model_outputs.ids_slice * feature_extractor.hop_length,
                                      self.config.segment_size
                                  )

                  discriminator_target, fmaps_target = discriminator(target_waveform)
                  discriminator_candidate, fmaps_candidate = discriminator(model_outputs.waveform.detach())
                  #with autocast(enabled=False):
                  if dict_state_grad_loss['discriminator']:


                        loss_disc, loss_real_disc, loss_fake_disc = discriminator_loss(
                            discriminator_target, discriminator_candidate
                        )

                        dk={"step_loss_disc": loss_disc.detach().item(),
                            "step_loss_real_disc": loss_real_disc.detach().item(),
                            "step_loss_fake_disc": loss_fake_disc.detach().item()}
                        displayloss['dict_loss_discriminator']=dk
                        loss_dd = loss_disc# + loss_real_disc + loss_fake_disc

                disc_optimizer.zero_grad()
                loss_dd.backward()


                # scaler.scale(loss_dd).backward()
                # scaler.unscale_(disc_optimizer )
                grad_norm_d = clip_grad_value_(discriminator.parameters(), None)
                disc_optimizer.step()


                with autocast(enabled=training_args.fp16):


                  # backpropagate
                  if dict_state_grad_loss['k1']:
                      loss_kl = kl_loss(
                          model_outputs.prior_latents,
                          model_outputs.posterior_log_variances,
                          model_outputs.prior_means,
                          model_outputs.prior_log_variances,
                          model_outputs.labels_padding_mask,
                      )
                      loss_kl=loss_kl*training_args.weight_kl
                      displayloss['loss_kl']=loss_kl.detach().item()
                      #if displayloss['loss_kl']>=0:
                        #  loss_kl.backward()

                  if dict_state_grad_loss['mel']:
                      loss_mel = torch.nn.functional.l1_loss(mel_scaled_target, mel_scaled_generation)
                      displayloss['loss_mel'] = loss_mel.detach().item()
                      train_losses_sum = train_losses_sum + displayloss['loss_mel']
                    # if displayloss['loss_mel']>=0:
                        #  loss_mel.backward()

                  if dict_state_grad_loss['duration']:
                      loss_duration=torch.sum(model_outputs.log_duration)*training_args.weight_duration
                      displayloss['loss_duration'] = loss_duration.detach().item()
                    #  if displayloss['loss_duration']>=0:
                      #   loss_duration.backward()




                  discriminator_target, fmaps_target = discriminator(target_waveform)

                  discriminator_candidate, fmaps_candidate = discriminator(model_outputs.waveform.detach())

                  if dict_state_grad_loss['generator']:
                      loss_fmaps = feature_loss(fmaps_target, fmaps_candidate)
                      loss_gen, losses_gen = generator_loss(discriminator_candidate)
                      loss_gen=loss_gen * training_args.weight_gen
                      displayloss['loss_gen'] = loss_gen.detach().item()
                  #   loss_gen.backward(retain_graph=True)
                      loss_fmaps=loss_fmaps * training_args.weight_fmaps
                      displayloss['loss_fmaps'] = loss_fmaps.detach().item()
                  #   loss_fmaps.backward(retain_graph=True)
                      total_generator_loss = (
                          loss_duration
                          + loss_mel
                          + loss_kl
                          + loss_fmaps
                          + loss_gen
                      )

                optimizer.zero_grad()
                total_generator_loss.backward()
                # scaler.scale(total_generator_loss).backward()
                # scaler.unscale_(optimizer)
                grad_norm_g = clip_grad_value_(self.parameters(), None)
                optimizer.step()
                # scaler.update()






                # optimizer.step()




                print(f"TRAINIG - batch {step}, waveform {(batch['waveform'].shape)}, lr {lr_scheduler.get_last_lr()[0]}... ")
                print(f"display loss function  enable  :{displayloss}")

                global_step +=1

                # validation

                do_eval = training_args.do_eval and (global_step % training_args.eval_steps == 0)
                if do_eval:
                    logger.info("Running validation... ")
                    eval_losses_sum = 0
                    cc=0;
                    for step, batch in enumerate(eval_dataset):
                        break
                        if cc>2: break
                        cc+=1
                        with torch.no_grad():
                            model_outputs = self.forward(
                                input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                            labels_attention_mask=batch["labels_attention_mask"],
                            speaker_id=batch["speaker_id"],


                            return_dict=True,
                            monotonic_alignment_function=None,
                                )

                        mel_scaled_labels = batch["mel_scaled_input_features"]
                        mel_scaled_target = self.slice_segments(mel_scaled_labels, model_outputs.ids_slice,self.segment_size)
                        mel_scaled_generation = feature_extractor._torch_extract_fbank_features(model_outputs.waveform.squeeze(1))[1]
                        loss = loss_mel.detach().item()
                        eval_losses_sum +=loss

                        loss_mel = torch.nn.functional.l1_loss(mel_scaled_target, mel_scaled_generation)
                        print(f"VALIDATION - batch {step}, waveform {(batch['waveform'].shape)}, step_loss_mel {loss} ... ")



                    with torch.no_grad():
                        full_generation_sample = self.full_generation_sample
                        full_generation =self.forward(
                          input_ids =full_generation_sample["input_ids"],
                          attention_mask=full_generation_sample["attention_mask"],
                          speaker_id=full_generation_sample["speaker_id"]
                          )

                        full_generation_waveform = full_generation.waveform.cpu().numpy()

                    wandb.log({
                    "eval_losses": eval_losses_sum,
                    "full generations samples": [
                        wandb.Audio(w.reshape(-1), caption=f"Full generation sample {epoch}", sample_rate=16000)
                        for w in full_generation_waveform],})

            wandb.log({"train_losses":train_losses_sum})

        # add weight norms
        # self.remove_weight_norm()

        try:
          torch.save(self.posterior_encoder.state_dict(), os.path.join(training_args.output_dir,"posterior_encoder.pt"))
          torch.save(self.decoder.state_dict(), os.path.join(training_args.output_dir,"decoder.pt"))
        except:pass


        logger.info("Running final full generations samples... ")


        with torch.no_grad():

            full_generation_sample = self.full_generation_sample
            full_generation = self.forward(
                    input_ids=full_generation_sample["labels"],
                    attention_mask=full_generation_sample["labels_attention_mask"],
                    speaker_id=full_generation_sample["speaker_id"]
                    )

            full_generation_waveform = full_generation.waveform.cpu().numpy()

        wandb.log({"eval_losses": eval_losses_sum,
                "full generations samples": [
                    wandb.Audio(w.reshape(-1), caption=f"Full generation sample {epoch}",
                                sample_rate=16000) for w in full_generation_waveform],
                })


        logger.info("***** Training / Inference Done *****")

    def forward_train(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        speaker_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.FloatTensor] = None,
        labels_attention_mask: Optional[torch.Tensor] = None,
        text_encoder_output=None,
        posterior_encode_output=None,
        monotonic_alignment_function: Optional[Callable] = None,
        speaker_embeddings=None
    ) -> Union[Tuple[Any], VitsModelOutput]:

        #output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states# if output_hidden_states is not None else self.config.output_hidden_states
        )
       # return_dict = return_dict if return_dict is not None else self.config.use_return_dict


       # if attention_mask is not None:
        input_padding_mask = attention_mask.unsqueeze(-1).float()
        #else:
         #   input_padding_mask = torch.ones_like(input_ids).unsqueeze(-1).float()

        # speaker_embeddings=None
       # if labels_attention_mask is not None:
        labels_padding_mask = labels_attention_mask.unsqueeze(1).float()
        # else:
        #     labels_attention_mask = torch.ones((labels.shape[0], labels.shape[2])).float().to(self.device)
        #     labels_padding_mask = labels_attention_mask.unsqueeze(1)
        if text_encoder_output is None:
          text_encoder_output = self.text_encoder(
                  input_ids=input_ids,
                  padding_mask=input_padding_mask,
                  attention_mask=attention_mask,
                  output_attentions=output_attentions,
                  output_hidden_states=output_hidden_states,
                  return_dict=return_dict,
              )
        #hidden_states = text_encoder_output[0] #if not return_dict else text_encoder_output.last_hidden_state
        hidden_states = text_encoder_output[0].transpose(1, 2)
        input_padding_mask = input_padding_mask.transpose(1, 2)
        prior_means = text_encoder_output[1].transpose(1, 2) #if not return_dict else text_encoder_output.prior_means
        prior_log_variances = text_encoder_output[2].transpose(1, 2) #if not return_dict else text_encoder_output.prior_log_variances

        if posterior_encode_output is None:
             latents, posterior_means, posterior_log_variances = self.posterior_encoder(
                labels, labels_padding_mask, speaker_embeddings
            )
        else:
             latents=posterior_encode_output['posterior_latents']
             posterior_means=posterior_encode_output['posterior_means']
             posterior_log_variances=posterior_encode_output['posterior_log_variances']

        prior_latents = self.flow(latents, labels_padding_mask, speaker_embeddings, reverse=False)

        # prior_means, prior_log_variances = prior_means.transpose(1, 2), prior_log_variances.transpose(1, 2)
        with torch.no_grad():
            # negative cross-entropy

            # [batch_size, d, latent_length]
            prior_variances = torch.exp(-2 * prior_log_variances)
            # [batch_size, 1, latent_length]
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - prior_log_variances, [1], keepdim=True)
            # [batch_size, text_length, d] x [batch_size, d, latent_length] = [batch_size, text_length, latent_length]
            neg_cent2 = torch.matmul(-0.5 * (prior_latents**2).transpose(1, 2), prior_variances)
            # [batch_size, text_length, d] x [batch_size, d, latent_length] = [batch_size, text_length, latent_length]
            neg_cent3 = torch.matmul(prior_latents.transpose(1, 2), (prior_means * prior_variances))
            # [batch_size, 1, latent_length]
            neg_cent4 = torch.sum(-0.5 * (prior_means**2) * prior_variances, [1], keepdim=True)

            # [batch_size, text_length, latent_length]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            attn_mask = torch.unsqueeze(input_padding_mask, 2) * torch.unsqueeze(labels_padding_mask, -1)

            attn = monotonic_alignment_function(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

        durations = attn.sum(2)

        #if self.config.use_stochastic_duration_prediction:
        log_duration = self.duration_predictor(
                hidden_states, input_padding_mask, speaker_embeddings, durations=durations, reverse=False
            )
        log_duration = log_duration / torch.sum(input_padding_mask)
        # else:
        #     log_duration_padded = torch.log(durations + 1e-6) * input_padding_mask
        #     log_duration = self.duration_predictor(hidden_states, input_padding_mask, speaker_embeddings)
        #     log_duration = torch.sum((log_duration - log_duration_padded) ** 2, [1, 2]) / torch.sum(input_padding_mask)

        # expand priors
        prior_means = torch.matmul(attn.squeeze(1), prior_means.transpose(1, 2)).transpose(1, 2)
        prior_log_variances = torch.matmul(attn.squeeze(1), prior_log_variances.transpose(1, 2)).transpose(1, 2)

        label_lengths = labels_attention_mask.sum(dim=1)
        latents_slice, ids_slice = self.rand_slice_segments(latents, label_lengths, segment_size=self.segment_size)
        waveform = self.decoder(latents_slice, speaker_embeddings)
        return waveform,ids_slice,log_duration,prior_latents,posterior_log_variances,prior_means,prior_log_variances,labels_padding_mask
