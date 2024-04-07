import math
from typing import Optional
import torch
from torch import nn
from .vits_config import VitsConfig
from .flow import VitsWaveNet

#.............................................



class VitsPosteriorEncoder(nn.Module):
    def __init__(self, config: VitsConfig):
        super().__init__()
        self.out_channels = config.flow_size

        self.conv_pre = nn.Conv1d(config.spectrogram_bins, config.hidden_size, 1)
        self.wavenet = VitsWaveNet(config, num_layers=config.posterior_encoder_num_wavenet_layers)
        self.conv_proj = nn.Conv1d(config.hidden_size, self.out_channels * 2, 1)

    def forward(self, inputs, padding_mask, global_conditioning=None):
        inputs = self.conv_pre(inputs) * padding_mask
        inputs = self.wavenet(inputs, padding_mask, global_conditioning)
        stats = self.conv_proj(inputs) * padding_mask
        mean, log_stddev = torch.split(stats, self.out_channels, dim=1)
        sampled = (mean + torch.randn_like(mean) * torch.exp(log_stddev)) * padding_mask
        return sampled, mean, log_stddev

    def apply_weight_norm(self):
        self.wavenet.apply_weight_norm()

    def remove_weight_norm(self):
        self.wavenet.remove_weight_norm()

    def resize_speaker_embeddings(self, speaker_embedding_size: Optional[int] = None):
        self.wavenet.speaker_embedding_size = speaker_embedding_size
        hidden_size = self.wavenet.hidden_size
        num_layers = self.wavenet.num_layers

        cond_layer = torch.nn.Conv1d(speaker_embedding_size, 2 * hidden_size * num_layers, 1)
        self.wavenet.cond_layer = nn.utils.weight_norm(cond_layer, name="weight")
        nn.init.kaiming_normal_(self.wavenet.cond_layer.weight)
        if self.wavenet.cond_layer.bias is not None:
            k = math.sqrt(
                self.wavenet.cond_layer.groups
                / (self.wavenet.cond_layer.in_channels * self.wavenet.cond_layer.kernel_size[0])
            )
            nn.init.uniform_(self.wavenet.cond_layer.bias, a=-k, b=k)

#.............................................................................................