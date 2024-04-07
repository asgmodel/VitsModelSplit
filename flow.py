
import torch
from torch import nn
from vits_config import VitsConfig
from typing import  Optional

#.............................................

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, num_channels):
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :num_channels, :])
    s_act = torch.sigmoid(in_act[:, num_channels:, :])
    acts = t_act * s_act
    return acts



#.............................................

class VitsWaveNet(torch.nn.Module):
    def __init__(self, config: VitsConfig, num_layers: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_layers = num_layers
        self.speaker_embedding_size = config.speaker_embedding_size

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.dropout = nn.Dropout(config.wavenet_dropout)

        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm
        else:
            weight_norm = nn.utils.weight_norm

        if config.speaker_embedding_size != 0:
            cond_layer = torch.nn.Conv1d(config.speaker_embedding_size, 2 * config.hidden_size * num_layers, 1)
            self.cond_layer = weight_norm(cond_layer, name="weight")

        for i in range(num_layers):
            dilation = config.wavenet_dilation_rate**i
            padding = (config.wavenet_kernel_size * dilation - dilation) // 2
            in_layer = torch.nn.Conv1d(
                in_channels=config.hidden_size,
                out_channels=2 * config.hidden_size,
                kernel_size=config.wavenet_kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < num_layers - 1:
                res_skip_channels = 2 * config.hidden_size
            else:
                res_skip_channels = config.hidden_size

            res_skip_layer = torch.nn.Conv1d(config.hidden_size, res_skip_channels, 1)
            res_skip_layer = weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, inputs, padding_mask, global_conditioning=None):
        outputs = torch.zeros_like(inputs)
        num_channels_tensor = torch.IntTensor([self.hidden_size])

        if global_conditioning is not None:
            global_conditioning = self.cond_layer(global_conditioning)

        for i in range(self.num_layers):
            hidden_states = self.in_layers[i](inputs)

            if global_conditioning is not None:
                cond_offset = i * 2 * self.hidden_size
                global_states = global_conditioning[:, cond_offset : cond_offset + 2 * self.hidden_size, :]
            else:
                global_states = torch.zeros_like(hidden_states)

            acts = fused_add_tanh_sigmoid_multiply(hidden_states, global_states, num_channels_tensor[0])
            acts = self.dropout(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.num_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_size, :]
                inputs = (inputs + res_acts) * padding_mask
                outputs = outputs + res_skip_acts[:, self.hidden_size :, :]
            else:
                outputs = outputs + res_skip_acts

        return outputs * padding_mask

    def remove_weight_norm(self):
        if self.speaker_embedding_size != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for layer in self.in_layers:
            torch.nn.utils.remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(layer)

    def apply_weight_norm(self):
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm
        else:
            weight_norm = nn.utils.weight_norm

        if self.speaker_embedding_size != 0:
            weight_norm(self.cond_layer)
        for layer in self.in_layers:
            weight_norm(layer)
        for layer in self.res_skip_layers:
            weight_norm(layer)


#.............................................................................................

class VitsResidualCouplingLayer(nn.Module):
    def __init__(self, config: VitsConfig):
        super().__init__()
        self.half_channels = config.flow_size // 2

        self.conv_pre = nn.Conv1d(self.half_channels, config.hidden_size, 1)
        self.wavenet = VitsWaveNet(config, num_layers=config.prior_encoder_num_wavenet_layers)
        self.conv_post = nn.Conv1d(config.hidden_size, self.half_channels, 1)

    def forward(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        first_half, second_half = torch.split(inputs, [self.half_channels] * 2, dim=1)
        hidden_states = self.conv_pre(first_half) * padding_mask
        hidden_states = self.wavenet(hidden_states, padding_mask, global_conditioning)
        mean = self.conv_post(hidden_states) * padding_mask
        log_stddev = torch.zeros_like(mean)

        if not reverse:
            second_half = mean + second_half * torch.exp(log_stddev) * padding_mask
            outputs = torch.cat([first_half, second_half], dim=1)
            log_determinant = torch.sum(log_stddev, [1, 2])
            return outputs, log_determinant
        else:
            second_half = (second_half - mean) * torch.exp(-log_stddev) * padding_mask
            outputs = torch.cat([first_half, second_half], dim=1)
            return outputs, None

    def apply_weight_norm(self):
        nn.utils.weight_norm(self.conv_pre)
        self.wavenet.apply_weight_norm()
        nn.utils.weight_norm(self.conv_post)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv_pre)
        self.wavenet.remove_weight_norm()
        nn.utils.remove_weight_norm(self.conv_post)



#.............................................................................................

class VitsResidualCouplingBlock(nn.Module):
    def __init__(self, config: VitsConfig):
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(config.prior_encoder_num_flows):
            self.flows.append(VitsResidualCouplingLayer(config))

    def forward(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                inputs, _ = flow(inputs, padding_mask, global_conditioning)
                inputs = torch.flip(inputs, [1])
        else:
            for flow in reversed(self.flows):
                inputs = torch.flip(inputs, [1])
                inputs, _ = flow(inputs, padding_mask, global_conditioning, reverse=True)
        return inputs

    def apply_weight_norm(self):
        for flow in self.flows:
            flow.apply_weight_norm()

    def remove_weight_norm(self):
        for flow in self.flows:
            flow.remove_weight_norm()

    def resize_speaker_embeddings(self, speaker_embedding_size: Optional[int] = None):
        for flow in self.flows:
            flow.wavenet.speaker_embedding_size = speaker_embedding_size
            hidden_size = flow.wavenet.hidden_size
            num_layers = flow.wavenet.num_layers

            cond_layer = torch.nn.Conv1d(speaker_embedding_size, 2 * hidden_size * num_layers, 1)
            flow.wavenet.cond_layer = nn.utils.weight_norm(cond_layer, name="weight")

