import math
import numpy as np
import torch
from torch import nn

from .vits_config import VitsConfig

#.............................................


def _rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    reverse,
    tail_bound,
    min_bin_width,
    min_bin_height,
    min_derivative,
):
    """
    This transformation represents a monotonically increasing piecewise rational quadratic function. Unlike the
    function `_unconstrained_rational_quadratic_spline`, the function behaves the same across the `tail_bound`.

    Args:
        inputs (`torch.FloatTensor` of shape `(batch_size, channels, seq_len)`:
            Second half of the hidden-states input to the Vits convolutional flow module.
        unnormalized_widths (`torch.FloatTensor` of shape `(batch_size, channels, seq_len, duration_predictor_flow_bins)`):
            First `duration_predictor_flow_bins` of the hidden-states from the output of the convolution projection
            layer in the convolutional flow module
        unnormalized_heights (`torch.FloatTensor` of shape `(batch_size, channels, seq_len, duration_predictor_flow_bins)`):
            Second `duration_predictor_flow_bins` of the hidden-states from the output of the convolution projection
            layer in the convolutional flow module
        unnormalized_derivatives (`torch.FloatTensor` of shape `(batch_size, channels, seq_len, duration_predictor_flow_bins)`):
            Third `duration_predictor_flow_bins` of the hidden-states from the output of the convolution projection
            layer in the convolutional flow module
        reverse (`bool`):
            Whether the model is being run in reverse mode.
        tail_bound (`float`):
            Upper and lower limit bound for the rational quadratic function. Outside of this `tail_bound`, the
            transform behaves as an identity function.
        min_bin_width (`float`):
            Minimum bin value across the width dimension for the piecewise rational quadratic function.
        min_bin_height (`float`):
            Minimum bin value across the height dimension for the piecewise rational quadratic function.
        min_derivative (`float`):
            Minimum bin value across the derivatives for the piecewise rational quadratic function.
    Returns:
        outputs (`torch.FloatTensor` of shape `(batch_size, channels, seq_len)`:
            Hidden-states as transformed by the piecewise rational quadratic function.
        log_abs_det (`torch.FloatTensor` of shape `(batch_size, channels, seq_len)`:
            Logarithm of the absolute value of the determinants corresponding to the `outputs`.
    """
    upper_bound = tail_bound
    lower_bound = -tail_bound
    if torch.min(inputs) < lower_bound or torch.max(inputs) > upper_bound:
        raise ValueError("Input to a transform is not within its domain")

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError(f"Minimal bin width {min_bin_width} too large for the number of bins {num_bins}")
    if min_bin_height * num_bins > 1.0:
        raise ValueError(f"Minimal bin height {min_bin_height} too large for the number of bins {num_bins}")

    widths = nn.functional.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = nn.functional.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (upper_bound - lower_bound) * cumwidths + lower_bound
    cumwidths[..., 0] = lower_bound
    cumwidths[..., -1] = upper_bound
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + nn.functional.softplus(unnormalized_derivatives)

    heights = nn.functional.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = nn.functional.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (upper_bound - lower_bound) * cumheights + lower_bound
    cumheights[..., 0] = lower_bound
    cumheights[..., -1] = upper_bound
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    bin_locations = cumheights if reverse else cumwidths
    bin_locations[..., -1] += 1e-6
    bin_idx = torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1
    bin_idx = bin_idx[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    intermediate1 = input_derivatives + input_derivatives_plus_one - 2 * input_delta
    if not reverse:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + intermediate1 * theta_one_minus_theta
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        log_abs_det = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, log_abs_det
    else:
        # find the roots of a quadratic equation
        intermediate2 = inputs - input_cumheights
        intermediate3 = intermediate2 * intermediate1
        a = input_heights * (input_delta - input_derivatives) + intermediate3
        b = input_heights * input_derivatives - intermediate3
        c = -input_delta * intermediate2

        discriminant = b.pow(2) - 4 * a * c
        if not (discriminant >= 0).all():
            raise RuntimeError(f"invalid discriminant {discriminant}")

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + intermediate1 * theta_one_minus_theta
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        log_abs_det = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, -log_abs_det

#.............................................

def _unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    reverse=False,
    tail_bound=5.0,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3,
):
    """
    This transformation represents a monotonically increasing piecewise rational quadratic function. Outside of the
    `tail_bound`, the transform behaves as an identity function.

    Args:
        inputs (`torch.FloatTensor` of shape `(batch_size, channels, seq_len)`:
            Second half of the hidden-states input to the Vits convolutional flow module.
        unnormalized_widths (`torch.FloatTensor` of shape `(batch_size, channels, seq_len, duration_predictor_flow_bins)`):
            First `duration_predictor_flow_bins` of the hidden-states from the output of the convolution projection
            layer in the convolutional flow module
        unnormalized_heights (`torch.FloatTensor` of shape `(batch_size, channels, seq_len, duration_predictor_flow_bins)`):
            Second `duration_predictor_flow_bins` of the hidden-states from the output of the convolution projection
            layer in the convolutional flow module
        unnormalized_derivatives (`torch.FloatTensor` of shape `(batch_size, channels, seq_len, duration_predictor_flow_bins)`):
            Third `duration_predictor_flow_bins` of the hidden-states from the output of the convolution projection
            layer in the convolutional flow module
        reverse (`bool`, *optional*, defaults to `False`):
            Whether the model is being run in reverse mode.
        tail_bound (`float`, *optional* defaults to 5):
            Upper and lower limit bound for the rational quadratic function. Outside of this `tail_bound`, the
            transform behaves as an identity function.
        min_bin_width (`float`, *optional*, defaults to 1e-3):
            Minimum bin value across the width dimension for the piecewise rational quadratic function.
        min_bin_height (`float`, *optional*, defaults to 1e-3):
            Minimum bin value across the height dimension for the piecewise rational quadratic function.
        min_derivative (`float`, *optional*, defaults to 1e-3):
            Minimum bin value across the derivatives for the piecewise rational quadratic function.
    Returns:
        outputs (`torch.FloatTensor` of shape `(batch_size, channels, seq_len)`:
            Hidden-states as transformed by the piecewise rational quadratic function with the `tail_bound` limits
            applied.
        log_abs_det (`torch.FloatTensor` of shape `(batch_size, channels, seq_len)`:
            Logarithm of the absolute value of the determinants corresponding to the `outputs` with the `tail_bound`
            limits applied.
    """
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    log_abs_det = torch.zeros_like(inputs)
    constant = np.log(np.exp(1 - min_derivative) - 1)

    unnormalized_derivatives = nn.functional.pad(unnormalized_derivatives, pad=(1, 1))
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    log_abs_det[outside_interval_mask] = 0.0

    outputs[inside_interval_mask], log_abs_det[inside_interval_mask] = _rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        reverse=reverse,
        tail_bound=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )
    return outputs, log_abs_det


#.............................................................................................

class VitsConvFlow(nn.Module):
    def __init__(self, config: VitsConfig):
        super().__init__()
        self.filter_channels = config.hidden_size
        self.half_channels = config.depth_separable_channels // 2
        self.num_bins = config.duration_predictor_flow_bins
        self.tail_bound = config.duration_predictor_tail_bound

        self.conv_pre = nn.Conv1d(self.half_channels, self.filter_channels, 1)
        self.conv_dds = VitsDilatedDepthSeparableConv(config)
        self.conv_proj = nn.Conv1d(self.filter_channels, self.half_channels * (self.num_bins * 3 - 1), 1)

    def forward(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        first_half, second_half = torch.split(inputs, [self.half_channels] * 2, dim=1)

        hidden_states = self.conv_pre(first_half)
        hidden_states = self.conv_dds(hidden_states, padding_mask, global_conditioning)
        hidden_states = self.conv_proj(hidden_states) * padding_mask

        batch_size, channels, length = first_half.shape
        hidden_states = hidden_states.reshape(batch_size, channels, -1, length).permute(0, 1, 3, 2)

        unnormalized_widths = hidden_states[..., : self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_heights = hidden_states[..., self.num_bins : 2 * self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_derivatives = hidden_states[..., 2 * self.num_bins :]

        second_half, log_abs_det = _unconstrained_rational_quadratic_spline(
            second_half,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            reverse=reverse,
            tail_bound=self.tail_bound,
        )

        outputs = torch.cat([first_half, second_half], dim=1) * padding_mask
        if not reverse:
            log_determinant = torch.sum(log_abs_det * padding_mask, [1, 2])
            return outputs, log_determinant
        else:
            return outputs, None


#.............................................................................................

class VitsElementwiseAffine(nn.Module):
    def __init__(self, config: VitsConfig):
        super().__init__()
        self.channels = config.depth_separable_channels
        self.translate = nn.Parameter(torch.zeros(self.channels, 1))
        self.log_scale = nn.Parameter(torch.zeros(self.channels, 1))

    def forward(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        if not reverse:
            outputs = self.translate + torch.exp(self.log_scale) * inputs
            outputs = outputs * padding_mask
            log_determinant = torch.sum(self.log_scale * padding_mask, [1, 2])
            return outputs, log_determinant
        else:
            outputs = (inputs - self.translate) * torch.exp(-self.log_scale) * padding_mask
            return outputs, None

#.............................................................................................

class VitsDilatedDepthSeparableConv(nn.Module):
    def __init__(self, config: VitsConfig, dropout_rate=0.0):
        super().__init__()
        kernel_size = config.duration_predictor_kernel_size
        channels = config.hidden_size
        self.num_layers = config.depth_separable_num_layers

        self.dropout = nn.Dropout(dropout_rate)
        self.convs_dilated = nn.ModuleList()
        self.convs_pointwise = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for i in range(self.num_layers):
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_dilated.append(
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    groups=channels,
                    dilation=dilation,
                    padding=padding,
                )
            )
            self.convs_pointwise.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(nn.LayerNorm(channels))
            self.norms_2.append(nn.LayerNorm(channels))

    def forward(self, inputs, padding_mask, global_conditioning=None):
        if global_conditioning is not None:
            inputs = inputs + global_conditioning

        for i in range(self.num_layers):
            hidden_states = self.convs_dilated[i](inputs * padding_mask)
            hidden_states = self.norms_1[i](hidden_states.transpose(1, -1)).transpose(1, -1)
            hidden_states = nn.functional.gelu(hidden_states)
            hidden_states = self.convs_pointwise[i](hidden_states)
            hidden_states = self.norms_2[i](hidden_states.transpose(1, -1)).transpose(1, -1)
            hidden_states = nn.functional.gelu(hidden_states)
            hidden_states = self.dropout(hidden_states)
            inputs = inputs + hidden_states

        return inputs * padding_mask

#.............................................................................................

class VitsStochasticDurationPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.speaker_embedding_size
        filter_channels = config.hidden_size

        self.conv_pre = nn.Conv1d(filter_channels, filter_channels, 1)
        self.conv_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.conv_dds = VitsDilatedDepthSeparableConv(
            config,
            dropout_rate=config.duration_predictor_dropout,
        )

        if embed_dim != 0:
            self.cond = nn.Conv1d(embed_dim, filter_channels, 1)

        self.flows = nn.ModuleList()
        self.flows.append(VitsElementwiseAffine(config))
        for _ in range(config.duration_predictor_num_flows):
            self.flows.append(VitsConvFlow(config))

        self.post_conv_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_conv_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_conv_dds = VitsDilatedDepthSeparableConv(
            config,
            dropout_rate=config.duration_predictor_dropout,
        )

        self.post_flows = nn.ModuleList()
        self.post_flows.append(VitsElementwiseAffine(config))
        for _ in range(config.duration_predictor_num_flows):
            self.post_flows.append(VitsConvFlow(config))

        self.filter_channels = filter_channels

    def resize_speaker_embeddings(self, speaker_embedding_size):
        self.cond = nn.Conv1d(speaker_embedding_size, self.filter_channels, 1)

    def forward(self, inputs, padding_mask, global_conditioning=None, durations=None, reverse=False, noise_scale=1.0):
        inputs = torch.detach(inputs)
        inputs = self.conv_pre(inputs)

        if global_conditioning is not None:
            global_conditioning = torch.detach(global_conditioning)
            inputs = inputs + self.cond(global_conditioning)

        inputs = self.conv_dds(inputs, padding_mask)
        inputs = self.conv_proj(inputs) * padding_mask

        if not reverse:
            hidden_states = self.post_conv_pre(durations)
            hidden_states = self.post_conv_dds(hidden_states, padding_mask)
            hidden_states = self.post_conv_proj(hidden_states) * padding_mask

            random_posterior = (
                torch.randn(durations.size(0), 2, durations.size(2)).to(device=inputs.device, dtype=inputs.dtype)
                * padding_mask
            )
            latents_posterior = random_posterior

            latents_posterior, log_determinant = self.post_flows[0](
                latents_posterior, padding_mask, global_conditioning=inputs + hidden_states
            )
            log_determinant_posterior_sum = log_determinant

            for flow in self.post_flows[1:]:
                latents_posterior, log_determinant = flow(
                    latents_posterior, padding_mask, global_conditioning=inputs + hidden_states
                )
                latents_posterior = torch.flip(latents_posterior, [1])
                log_determinant_posterior_sum += log_determinant

            first_half, second_half = torch.split(latents_posterior, [1, 1], dim=1)

            log_determinant_posterior_sum += torch.sum(
                (nn.functional.logsigmoid(first_half) + nn.functional.logsigmoid(-first_half)) * padding_mask, [1, 2]
            )
            logq = (
                torch.sum(-0.5 * (math.log(2 * math.pi) + (random_posterior**2)) * padding_mask, [1, 2])
                - log_determinant_posterior_sum
            )

            first_half = (durations - torch.sigmoid(first_half)) * padding_mask
            first_half = torch.log(torch.clamp_min(first_half, 1e-5)) * padding_mask
            log_determinant_sum = torch.sum(-first_half, [1, 2])

            latents = torch.cat([first_half, second_half], dim=1)
            latents, log_determinant = self.flows[0](latents, padding_mask, global_conditioning=inputs)

            log_determinant_sum += log_determinant
            for flow in self.flows[1:]:
                latents, log_determinant = flow(latents, padding_mask, global_conditioning=inputs)
                latents = torch.flip(latents, [1])
                log_determinant_sum += log_determinant

            nll = torch.sum(0.5 * (math.log(2 * math.pi) + (latents**2)) * padding_mask, [1, 2]) - log_determinant_sum
            return nll + logq
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow

            latents = (
                torch.randn(inputs.size(0), 2, inputs.size(2)).to(device=inputs.device, dtype=inputs.dtype)
                * noise_scale
            )
            for flow in flows:
                latents = torch.flip(latents, [1])
                latents, _ = flow(latents, padding_mask, global_conditioning=inputs, reverse=True)

            log_duration, _ = torch.split(latents, [1, 1], dim=1)
            return log_duration

#.............................................................................................

class VitsDurationPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        kernel_size = config.duration_predictor_kernel_size
        filter_channels = config.duration_predictor_filter_channels

        self.dropout = nn.Dropout(config.duration_predictor_dropout)
        self.conv_1 = nn.Conv1d(config.hidden_size, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = nn.LayerNorm(filter_channels, eps=config.layer_norm_eps)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = nn.LayerNorm(filter_channels, eps=config.layer_norm_eps)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if config.speaker_embedding_size != 0:
            self.cond = nn.Conv1d(config.speaker_embedding_size, config.hidden_size, 1)

        self.hidden_size = config.hidden_size

    def resize_speaker_embeddings(self, speaker_embedding_size):
        self.cond = nn.Conv1d(speaker_embedding_size, self.hidden_size, 1)

    def forward(self, inputs, padding_mask, global_conditioning=None):
        inputs = torch.detach(inputs)

        if global_conditioning is not None:
            global_conditioning = torch.detach(global_conditioning)
            inputs = inputs + self.cond(global_conditioning)

        inputs = self.conv_1(inputs * padding_mask)
        inputs = torch.relu(inputs)
        inputs = self.norm_1(inputs.transpose(1, -1)).transpose(1, -1)
        inputs = self.dropout(inputs)

        inputs = self.conv_2(inputs * padding_mask)
        inputs = torch.relu(inputs)
        inputs = self.norm_2(inputs.transpose(1, -1)).transpose(1, -1)
        inputs = self.dropout(inputs)

        inputs = self.proj(inputs * padding_mask)
        return inputs * padding_mask

#.............................................................................................