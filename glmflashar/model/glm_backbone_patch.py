# -*- coding: utf-8 -*-
"""Runtime patch of GLM-Image's inner AR transformer (`GlmImageTextModel`).

FlashAR needs the backbone forward to:
  (a) accept a caller-supplied 4-D additive attention mask (eager attn) instead
      of always building a causal mask,
  (b) capture the hidden state *before* a chosen layer index,
  (c) optionally stop after N layers (split-backbone / vertical-branch reuse),
  (d) return an output object exposing both FlashAR fields and the stock
      .hidden_states / .attentions attributes read by GlmImageModel wrappers.

We patch in place (rather than vendoring a fork) so we stay on upstream weights.
`GlmImageTextModel` layers return the hidden tensor directly and update the
DynamicCache in-place (unlike Emu3 which returns (hidden, cache) tuples).
"""
from __future__ import annotations

import torch
from transformers.models.glm_image import modeling_glm_image as _mg


class BaseModelOutputWithCapture:
    """Lightweight output compatible with stock GLM wrapper expectations."""

    __slots__ = (
        "last_hidden_state",
        "past_key_values",
        "captured_hidden_state",
        "hidden_states",
        "attentions",
    )

    def __init__(
        self,
        last_hidden_state,
        past_key_values=None,
        captured_hidden_state=None,
        hidden_states=None,
        attentions=None,
    ):
        self.last_hidden_state = last_hidden_state
        self.past_key_values = past_key_values
        self.captured_hidden_state = captured_hidden_state
        self.hidden_states = hidden_states
        self.attentions = attentions


def _flashar_text_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    output_hidden_states=None,
    output_attentions=None,
    capture_hidden_before_layer: int | None = None,
    stop_after_layer: int | None = None,
    **kwargs,
):
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("Specify exactly one of input_ids / inputs_embeds")

    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = _mg.DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # ---- position ids (3D M-RoPE: T,H,W) ----------------------------------
    if position_ids is None:
        past = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past
        position_ids = position_ids.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = None

    # ---- attention mask ----------------------------------------------------
    # If the caller passes a 4-D float additive mask, use it verbatim (FlashAR
    # proximity / diagonal masks). Otherwise fall back to upstream causal mask.
    if attention_mask is not None and attention_mask.dim() == 4:
        causal_mask = attention_mask
    else:
        causal_mask = _mg.create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

    captured = None
    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None
    n_layers = len(self.layers)
    for i, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if capture_hidden_before_layer is not None and i == capture_hidden_before_layer:
            captured = hidden_states
        if stop_after_layer is not None and i >= stop_after_layer:
            # return pre-norm hidden (caller applies norm / continues)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            return BaseModelOutputWithCapture(
                hidden_states,
                past_key_values,
                captured,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
            )
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    if capture_hidden_before_layer is not None and capture_hidden_before_layer >= n_layers:
        captured = hidden_states

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    return BaseModelOutputWithCapture(
        hidden_states,
        past_key_values,
        captured,
        hidden_states=all_hidden_states,
        attentions=all_attentions,
    )


_PATCHED = False


def apply_glm_backbone_patch():
    """Idempotently install the FlashAR-capable forward on GlmImageTextModel."""
    global _PATCHED
    if _PATCHED:
        return
    _mg.GlmImageTextModel._original_forward = _mg.GlmImageTextModel.forward
    _mg.GlmImageTextModel.forward = _flashar_text_forward
    _PATCHED = True


def set_layers_non_causal(text_model, non_causal: bool = True):
    """Toggle self_attn.is_causal on decoder layers.

    Diagonal decode needs FlashAR's explicit proximity mask to be authoritative,
    not an implicit attention-kernel causal mask. Accept either a GLM text model
    exposing ``.layers`` or a layer iterable such as ``vertical_block``.
    """
    layers = getattr(text_model, "layers", text_model)
    for layer in layers:
        if hasattr(layer, "self_attn"):
            layer.self_attn.is_causal = not non_causal
