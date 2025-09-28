import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithCrossAttentions,
)
from transformers.modeling_utils import PreTrainedModel

from .config import HypersphereGPTConfig
from variations.norm_variations import HyperSphereNorm
from variations.position_encoding_variations import RotaryEmbedding


def _make_causal_mask(
    input_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
) -> torch.Tensor:
    batch_size, target_length = input_shape
    mask = torch.full((target_length, target_length), float("-inf"), dtype=dtype, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask = mask.masked_fill(mask_cond.unsqueeze(0) <= mask_cond.unsqueeze(-1), 0.0)
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, target_length, target_length)

    if past_key_values_length > 0:
        past_mask = torch.zeros((target_length, past_key_values_length), dtype=dtype, device=device)
        mask = torch.cat([past_mask.unsqueeze(0).unsqueeze(0), mask], dim=-1)

    return mask.expand(batch_size, -1, -1, -1)


def _expand_mask(
    mask: torch.Tensor,
    dtype: torch.dtype,
    target_length: Optional[int] = None,
) -> torch.Tensor:
    batch_size, source_length = mask.size()
    target_length = target_length if target_length is not None else source_length

    expanded_mask = mask[:, None, None, :].expand(batch_size, 1, target_length, source_length).to(dtype)
    inverted_mask = (1.0 - expanded_mask) * torch.finfo(dtype).min
    return inverted_mask


class HypersphereGPTPreTrainedModel(PreTrainedModel):
    config_class = HypersphereGPTConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HypersphereBlock"]

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)


class HypersphereSelfAttention(nn.Module):
    def __init__(self, config: HypersphereGPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.use_qk_norm = config.use_qk_norm
        self.use_qk_norm_scale = config.use_qk_norm_scale
        if self.use_qk_norm_scale:
            self.qk_norm_factor = nn.Parameter(torch.tensor(self.scale))
        else:
            self.register_parameter("qk_norm_factor", None)

        self.use_rotary = config.use_rotary_embeddings
        if self.use_rotary:
            self.rotary_q = RotaryEmbedding(config=config, size=self.head_dim)
            self.rotary_k = RotaryEmbedding(config=config, size=self.head_dim)
        else:
            self.rotary_q = None
            self.rotary_k = None

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)),
            persistent=False,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int) -> torch.Tensor:
        return tensor.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_length, _ = hidden_states.size()

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._shape(query, seq_length, batch_size)
        key = self._shape(key, seq_length, batch_size)
        value = self._shape(value, seq_length, batch_size)

        if self.use_rotary:
            query = self.rotary_q(query)
            key = self.rotary_k(key)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)
        present = (key, value) if use_cache else None

        if self.use_qk_norm:
            query = query / (query.norm(dim=-1, keepdim=True) + 1e-6)
            key = key / (key.norm(dim=-1, keepdim=True) + 1e-6)

        attn_weights = torch.matmul(query, key.transpose(-2, -1))
        if self.use_qk_norm_scale:
            attn_weights = attn_weights * self.qk_norm_factor
        else:
            attn_weights = attn_weights * self.scale

        causal_mask = self.bias[: seq_length, : key.size(-2)]
        causal_mask = causal_mask.view(1, 1, seq_length, key.size(-2))
        attn_weights = attn_weights.masked_fill(~causal_mask, float("-inf"))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        return attn_output, present


class HypersphereMLP(nn.Module):
    def __init__(self, config: HypersphereGPTConfig) -> None:
        super().__init__()
        self.fc_in = nn.Linear(config.n_embd, config.n_inner, bias=False)
        self.act = nn.GELU()
        self.fc_out = nn.Linear(config.n_inner, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class HypersphereBlock(nn.Module):
    def __init__(self, config: HypersphereGPTConfig) -> None:
        super().__init__()
        self.pre_attn_norm = HyperSphereNorm(config)
        self.attn = HypersphereSelfAttention(config)
        self.peri_attn_norm = HyperSphereNorm(config) if config.use_peri_ln_attn else nn.Identity()

        self.pre_mlp_norm = HyperSphereNorm(config)
        self.mlp = HypersphereMLP(config)
        self.peri_mlp_norm = HyperSphereNorm(config) if config.use_peri_ln_mlp else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        normed_states = self.pre_attn_norm(hidden_states)
        attn_output, present = self.attn(normed_states, attention_mask=attention_mask, layer_past=layer_past, use_cache=use_cache)
        attn_output = self.peri_attn_norm(attn_output) if not isinstance(self.peri_attn_norm, nn.Identity) else attn_output
        hidden_states = residual + attn_output

        residual = hidden_states
        normed_states = self.pre_mlp_norm(hidden_states)
        mlp_output = self.mlp(normed_states)
        mlp_output = self.peri_mlp_norm(mlp_output) if not isinstance(self.peri_mlp_norm, nn.Identity) else mlp_output
        hidden_states = residual + mlp_output

        return hidden_states, present


class HypersphereGPTModel(HypersphereGPTPreTrainedModel):
    def __init__(self, config: HypersphereGPTConfig) -> None:
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([HypersphereBlock(config) for _ in range(config.n_layer)])
        self.ln_f = HyperSphereNorm(config)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPast]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot provide both input_ids and inputs_embeds")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You must provide input_ids or inputs_embeds")

        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        else:
            input_shape = inputs_embeds.size()[:-1]

        batch_size = input_shape[0]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        inputs_embeds = self.wte(input_ids) if inputs_embeds is None else inputs_embeds

        hidden_states = self.drop(inputs_embeds)

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else True

        all_hidden_states = () if output_hidden_states else None
        presents = () if use_cache else None

        past_key_values = past_key_values or [None] * len(self.h)
        past_key_values_length = past_key_values[0][0].size(-2) if past_key_values[0] is not None else 0

        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            attn_mask = _expand_mask(attention_mask, hidden_states.dtype, target_length=hidden_states.size(1))
            if past_key_values_length > 0:
                zeros = torch.zeros(
                    batch_size,
                    1,
                    hidden_states.size(1),
                    past_key_values_length,
                    dtype=hidden_states.dtype,
                    device=device,
                )
                attn_mask = torch.cat([zeros, attn_mask], dim=-1)
        else:
            attn_mask = None

        causal_mask = _make_causal_mask(input_shape, hidden_states.dtype, device, past_key_values_length)
        attention_mask_combined = causal_mask if attn_mask is None else (causal_mask + attn_mask)

        for block_idx, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states, present = block(
                hidden_states,
                attention_mask=attention_mask_combined,
                layer_past=layer_past,
                use_cache=use_cache,
            )

            if use_cache:
                presents = presents + (present,)

        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            outputs = (hidden_states, presents, all_hidden_states)
            return tuple(v for v in outputs if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
        )


class HypersphereGPTForCausalLM(HypersphereGPTPreTrainedModel):
    def __init__(self, config: HypersphereGPTConfig) -> None:
        super().__init__(config)
        self.transformer = HypersphereGPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.post_init()

        # Weight tying
        self.lm_head.weight = self.transformer.wte.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.transformer.wte

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.transformer.wte = value
        self.lm_head.weight = value.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], CausalLMOutputWithCrossAttentions]:
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=None,
            cross_attentions=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
        }


AutoConfig.register("hypersphere-gpt", HypersphereGPTConfig)
AutoModelForCausalLM.register(HypersphereGPTConfig, HypersphereGPTForCausalLM)
