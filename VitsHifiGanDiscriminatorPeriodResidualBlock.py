import torch
from torch import nn

#............................................


class VitsHifiGanDiscriminatorPeriodResidualBlock(nn.Module):
    
    def __init__(self, discriminator_period_channels, period, kernel_size=5, stride=3, leaky_relu_slope=0.1):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope
        self.period = period

        self.convs = nn.ModuleList()
        for in_channels, out_channels in zip(discriminator_period_channels[:-1], discriminator_period_channels[1:]):
            self.convs.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    (kernel_size, 1),
                    (stride, 1),
                    padding=(self.get_padding(kernel_size, 1), 0),
                )
            )

        channel_size = discriminator_period_channels[-1]
        self.convs.append(
            nn.Conv2d(channel_size, channel_size, (kernel_size, 1), 1, padding=(self.get_padding(kernel_size, 1), 0))
        )
        self.final_conv = nn.Conv2d(channel_size, 1, (3, 1), 1, padding=(1, 0))

    def get_padding(self, kernel_size, dilation=1):
        return (kernel_size * dilation - dilation) // 2

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

        # from 1D to 2D
        batch_size, channels, length = hidden_states.shape
        if length % self.period != 0:
            # pad first
            n_pad = self.period - (length % self.period)
            hidden_states = nn.functional.pad(hidden_states, (0, n_pad), "reflect")
            length = length + n_pad
        hidden_states = hidden_states.view(batch_size, channels, length // self.period, self.period)

        for conv in self.convs:
            hidden_states = conv(hidden_states)
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            fmap.append(hidden_states)

        hidden_states = self.final_conv(hidden_states)
        fmap.append(hidden_states)
        hidden_states = torch.flatten(hidden_states, 1, -1)

        return hidden_states, fmap


#.............................................................................................