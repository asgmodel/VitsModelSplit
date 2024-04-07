import math
from typing import Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutput  

from .vits_config import VitsConfig
from .vits_output import VitsTextEncoderOutput


#....................................................





class VitsFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_1 = nn.Conv1d(config.hidden_size, config.ffn_dim, config.ffn_kernel_size)
        self.conv_2 = nn.Conv1d(config.ffn_dim, config.hidden_size, config.ffn_kernel_size)
        self.dropout = nn.Dropout(config.activation_dropout)

        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act

        if config.ffn_kernel_size > 1:
            pad_left = (config.ffn_kernel_size - 1) // 2
            pad_right = config.ffn_kernel_size // 2
            self.padding = [pad_left, pad_right, 0, 0, 0, 0]
        else:
            self.padding = None

    def forward(self, hidden_states, padding_mask):
        hidden_states = hidden_states.permute(0, 2, 1)
        padding_mask = padding_mask.permute(0, 2, 1)

        hidden_states = hidden_states * padding_mask
        if self.padding is not None:
            hidden_states = nn.functional.pad(hidden_states, self.padding)

        hidden_states = self.conv_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states * padding_mask
        if self.padding is not None:
            hidden_states = nn.functional.pad(hidden_states, self.padding)

        hidden_states = self.conv_2(hidden_states)
        hidden_states = hidden_states * padding_mask

        hidden_states = hidden_states.permute(0, 2, 1)
        return hidden_states


#.............................................................................................

class VitsAttention(nn.Module):
    """Multi-headed attention with relative positional representation."""

    def __init__(self, config: VitsConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        self.window_size = config.window_size

        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads (got `hidden_size`: {self.embed_dim}"
                f" and `num_attention_heads`: {self.num_heads})."
            )

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias)

        if self.window_size:
            self.emb_rel_k = nn.Parameter(torch.randn(1, self.window_size * 2 + 1, self.head_dim) * self.scaling)
            self.emb_rel_v = nn.Parameter(torch.randn(1, self.window_size * 2 + 1, self.head_dim) * self.scaling)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling

        # self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if self.window_size is not None:
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, src_len)
            relative_logits = torch.matmul(query_states, key_relative_embeddings.transpose(-2, -1))
            rel_pos_bias = self._relative_position_to_absolute_position(relative_logits)
            attn_weights += rel_pos_bias

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        if self.window_size is not None:
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, src_len)
            relative_weights = self._absolute_position_to_relative_position(attn_probs)
            rel_pos_bias = torch.matmul(relative_weights, value_relative_embeddings)
            attn_output += rel_pos_bias

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped

    def _get_relative_embeddings(self, relative_embeddings, length):
        pad_length = max(length - (self.window_size + 1), 0)
        if pad_length > 0:
            relative_embeddings = nn.functional.pad(relative_embeddings, [0, 0, pad_length, pad_length, 0, 0])

        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        return relative_embeddings[:, slice_start_position:slice_end_position]

    def _relative_position_to_absolute_position(self, x):
        batch_heads, length, _ = x.size()

        # Concat columns of pad to shift from relative to absolute indexing.
        x = nn.functional.pad(x, [0, 1, 0, 0, 0, 0])

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch_heads, length * 2 * length])
        x_flat = nn.functional.pad(x_flat, [0, length - 1, 0, 0])

        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch_heads, length + 1, 2 * length - 1])
        x_final = x_final[:, :length, length - 1 :]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        batch_heads, length, _ = x.size()

        # Pad along column
        x = nn.functional.pad(x, [0, length - 1, 0, 0, 0, 0])
        x_flat = x.view([batch_heads, length**2 + length * (length - 1)])

        # Add 0's in the beginning that will skew the elements after reshape
        x_flat = nn.functional.pad(x_flat, [length, 0, 0, 0])
        x_final = x_flat.view([batch_heads, length, 2 * length])[:, :, 1:]
        return x_final


#.............................................................................................

class VitsEncoderLayer(nn.Module):
    def __init__(self, config: VitsConfig):
        super().__init__()
        self.attention = VitsAttention(config)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = VitsFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        residual = hidden_states
        hidden_states, attn_weights = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(residual + hidden_states)

        residual = hidden_states
        hidden_states = self.feed_forward(hidden_states, padding_mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.final_layer_norm(residual + hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

#.............................................................................................

class VitsEncoder(nn.Module):
    def __init__(self, config: VitsConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([VitsEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        self.layerdrop = config.layerdrop

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        padding_mask: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        hidden_states = hidden_states * padding_mask

        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)

            skip_the_layer = self.training and (dropout_probability < self.layerdrop)
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        padding_mask,
                        attention_mask,
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        padding_mask=padding_mask,
                        output_attentions=output_attentions,
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = hidden_states * padding_mask

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
        
#.............................................................................................

class VitsTextEncoder(nn.Module):
    """
    Transformer encoder that uses relative positional representation instead of absolute positional encoding.
    """

    def __init__(self, config: VitsConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        
        self.encoder = VitsEncoder(config)
        self.project = nn.Conv1d(config.hidden_size, config.flow_size * 2, kernel_size=1)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], VitsTextEncoderOutput]:
        hidden_states = self.embed_tokens(input_ids) * math.sqrt(self.config.hidden_size)

        encoder_outputs = self.encoder(
            hidden_states=hidden_states,
            padding_mask=padding_mask,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0] if not return_dict else encoder_outputs.last_hidden_state

        stats = self.project(last_hidden_state.transpose(1, 2)).transpose(1, 2) * padding_mask
        prior_means, prior_log_variances = torch.split(stats, self.config.flow_size, dim=2)

        if not return_dict:
            outputs = (last_hidden_state, prior_means, prior_log_variances) + encoder_outputs[1:]
            return outputs

        return VitsTextEncoderOutput(
            last_hidden_state=last_hidden_state,
            prior_means=prior_means,
            prior_log_variances=prior_log_variances,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        
#.............................................................................................
