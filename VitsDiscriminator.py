import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from VitsHifiGanDiscriminatorPeriodResidualBlock import VitsHifiGanDiscriminatorPeriodResidualBlock
from VitsHifiGanDiscriminatorScaleResidualBlock import VitsHifiGanDiscriminatorScaleResidualBlock
from VitsPreTrainedModel import VitsPreTrainedModel

#.............................................


class VitsDiscriminator(VitsPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        if config.discriminator_scale_channels is not None:
            self.discriminators = nn.ModuleList(
                [VitsHifiGanDiscriminatorScaleResidualBlock(config.discriminator_scale_channels, config.leaky_relu_slope)]
            )
        else:
            self.discriminators = nn.ModuleList([])
        
        self.discriminators.extend(
            [
                VitsHifiGanDiscriminatorPeriodResidualBlock(
                    config.discriminator_period_channels,
                    period,
                    config.discriminator_kernel_size,
                    config.discriminator_stride,
                    config.leaky_relu_slope,
                )
                for period in config.discriminator_periods
            ]
        )

    def forward(self, hidden_states):
        fmaps = []
        discriminated_hidden_states_list = []

        for discriminator in self.discriminators:
            discriminated_hidden_states, fmap = discriminator(hidden_states)
            fmaps.append(fmap)
            discriminated_hidden_states_list.append(discriminated_hidden_states)

        return discriminated_hidden_states_list, fmaps

    def apply_weight_norm(self):
        for disc in self.discriminators:
            disc.apply_weight_norm()

    def remove_weight_norm(self):
        for disc in self.discriminators:
            disc.remove_weight_norm()


#.............................................................................................