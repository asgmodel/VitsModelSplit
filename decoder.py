import math
from typing import Optional
import numpy as np
import torch
from torch import nn
from .vits_config import VitsConfig

#.............................................





# Copied from transformers.models.speecht5.modeling_speecht5.HifiGanResidualBlock
class HifiGanResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), leaky_relu_slope=0.1):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope

        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation[i],
                    padding=self.get_padding(kernel_size, dilation[i]),
                )
                for i in range(len(dilation))
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=1,
                    padding=self.get_padding(kernel_size, 1),
                )
                for _ in range(len(dilation))
            ]
        )

    def get_padding(self, kernel_size, dilation=1):
        return (kernel_size * dilation - dilation) // 2

    def apply_weight_norm(self):
        for layer in self.convs1:
            nn.utils.weight_norm(layer)
        for layer in self.convs2:
            nn.utils.weight_norm(layer)

    def remove_weight_norm(self):
        for layer in self.convs1:
            nn.utils.remove_weight_norm(layer)
        for layer in self.convs2:
            nn.utils.remove_weight_norm(layer)

    def forward(self, hidden_states):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = hidden_states
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv1(hidden_states)
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv2(hidden_states)
            hidden_states = hidden_states + residual
        return hidden_states


#.............................................................................................


class VitsHifiGan(nn.Module):
    def __init__(self, config: VitsConfig):
        super().__init__()
        self.config = config
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.conv_pre = nn.Conv1d(
            config.flow_size,
            config.upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.upsampler = nn.ModuleList()
        for i, (upsample_rate, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.upsampler.append(
                nn.ConvTranspose1d(
                    config.upsample_initial_channel // (2**i),
                    config.upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=kernel_size,
                    stride=upsample_rate,
                    padding=(kernel_size - upsample_rate) // 2,
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.upsampler)):
            channels = config.upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(HifiGanResidualBlock(channels, kernel_size, dilation, config.leaky_relu_slope))

        self.conv_post = nn.Conv1d(channels, 1, kernel_size=7, stride=1, padding=3, bias=False)

        if config.speaker_embedding_size != 0:
            self.cond = nn.Conv1d(config.speaker_embedding_size, config.upsample_initial_channel, 1)

    def resize_speaker_embedding(self, speaker_embedding_size):
        self.config.speaker_embedding_size = speaker_embedding_size
        self.cond = nn.Conv1d(speaker_embedding_size, self.config.upsample_initial_channel, 1)
        nn.init.kaiming_normal_(self.cond.weight)
        if self.cond.bias is not None:
            k = math.sqrt(self.cond.groups / (self.cond.in_channels * self.cond.kernel_size[0]))
            nn.init.uniform_(self.cond.bias, a=-k, b=k)

    def apply_weight_norm(self):
        for layer in self.upsampler:
            nn.utils.weight_norm(layer)
        for layer in self.resblocks:
            layer.apply_weight_norm()

    def remove_weight_norm(self):
        for layer in self.upsampler:
            nn.utils.remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()

    def forward(
        self, spectrogram: torch.FloatTensor, global_conditioning: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        r"""
        Converts a spectrogram into a speech waveform.

        Args:
            spectrogram (`torch.FloatTensor` of shape `(batch_size, config.spectrogram_bins, sequence_length)`):
                Tensor containing the spectrograms.
            global_conditioning (`torch.FloatTensor` of shape `(batch_size, config.speaker_embedding_size, 1)`, *optional*):
                Tensor containing speaker embeddings, for multispeaker models.

        Returns:
            `torch.FloatTensor`: Tensor of shape shape `(batch_size, 1, num_frames)` containing the speech waveform.
        """
        hidden_states = self.conv_pre(spectrogram)

        if global_conditioning is not None:
            hidden_states = hidden_states + self.cond(global_conditioning)

        for i in range(self.num_upsamples):
            hidden_states = nn.functional.leaky_relu(hidden_states, self.config.leaky_relu_slope)
            hidden_states = self.upsampler[i](hidden_states)

            res_state = self.resblocks[i * self.num_kernels](hidden_states)
            for j in range(1, self.num_kernels):
                res_state += self.resblocks[i * self.num_kernels + j](hidden_states)
            hidden_states = res_state / self.num_kernels

        hidden_states = nn.functional.leaky_relu(hidden_states)
        hidden_states = self.conv_post(hidden_states)
        waveform = torch.tanh(hidden_states)
        return waveform


#.............................................................................................
