from typing import Any, Optional, Tuple, Union,List,Dict
import torch
from dataclasses import dataclass
from transformers.modeling_outputs import (
    BaseModelOutput,
    ModelOutput,
)
#.............................................



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
