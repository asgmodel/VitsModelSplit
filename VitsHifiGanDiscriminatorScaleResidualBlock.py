import torch
from torch import nn

#.............................................

class VitsHifiGanDiscriminatorScaleResidualBlock(nn.Module):
    def __init__(self, discriminator_scale_channels, leaky_relu_slope=0.1):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope

        in_channels, out_channels = discriminator_scale_channels[:2]
        self.convs = nn.ModuleList([nn.Conv1d(in_channels, out_channels, 15, 1, padding=7)])

        groups = 4
        for in_channels, out_channels in zip(discriminator_scale_channels[1:-1], discriminator_scale_channels[2:]):
            self.convs.append(nn.Conv1d(in_channels, out_channels, 41, 4, groups=groups, padding=20))
            groups = groups * 4

        channel_size = discriminator_scale_channels[-1]
        self.convs.append(nn.Conv1d(channel_size, channel_size, 41, 4, groups=groups, padding=20))
        self.convs.append(nn.Conv1d(channel_size, channel_size, 5, 1, padding=2))
        self.final_conv = nn.Conv1d(channel_size, 1, 3, 1, padding=1)

    def apply_weight_norm(self):
        for layer in self.convs:
            nn.utils.weight_norm(layer)
        nn.utils.weight_norm(self.final_conv)

    def remove_weight_norm(self):
        for layer in self.convs:
            nn.utils.remove_weight_norm(layer)
        nn.utils.remove_weight_norm(self.final_conv)

    def forward(self, hidden_states):
        fmap = []

        for conv in self.convs:
            hidden_states = conv(hidden_states)
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            fmap.append(hidden_states)

        hidden_states = self.final_conv(hidden_states)
        fmap.append(hidden_states)
        hidden_states = torch.flatten(hidden_states, 1, -1)

        return hidden_states, fmap

#.............................................................................................

