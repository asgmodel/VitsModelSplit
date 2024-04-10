from typing import Any, Optional, Tuple, Union,List,Dict
import torch
from dataclasses import dataclass
from transformers.modeling_outputs import (
    BaseModelOutput,
    ModelOutput,
)
#.............................................



@dataclass
class PosteriorDecoderModelOutput(ModelOutput):
    labels_padding_mask: torch.FloatTensor = None
    posterior_latents: torch.FloatTensor = None
    posterior_means: torch.FloatTensor = None
    posterior_log_variances: torch.FloatTensor = None
    latents_slice : torch.FloatTensor = None
    ids_slice: torch.FloatTensor = None
    waveform: torch.FloatTensor = None
    
#.............................................................................................    
    
    
@dataclass
class VitsModelOutput(ModelOutput):
    waveform: torch.FloatTensor = None
    sequence_lengths: torch.FloatTensor = None
    spectrogram: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

#.............................................................................................

@dataclass
class VitsTrainingOutput(ModelOutput):
    waveform: torch.FloatTensor = None
    log_duration: torch.FloatTensor = None
    attn: torch.FloatTensor = None
    ids_slice: torch.FloatTensor = None
    input_padding_mask: torch.FloatTensor = None
    labels_padding_mask: torch.FloatTensor = None
    latents: torch.FloatTensor = None
    prior_latents: torch.FloatTensor = None
    prior_means: torch.FloatTensor = None
    prior_log_variances: torch.FloatTensor = None
    posterior_means: torch.FloatTensor = None
    posterior_log_variances: torch.FloatTensor = None


#.............................................................................................

@dataclass
class VitsTextEncoderOutput(ModelOutput):
    """
    Describes the outputs for the VITS text encoder model, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        prior_means (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            The predicted mean values of the prior distribution for the latent text variables.
        prior_log_variances (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            The predicted log-variance values of the prior distribution for the latent text variables.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attention weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    prior_means: torch.FloatTensor = None
    prior_log_variances: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

#.............................................................................................
