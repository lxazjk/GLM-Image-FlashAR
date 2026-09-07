# -*- coding: utf-8 -*-
"""Build the GLM-Image text prefix ids for FlashAR training.

The FlashAR wrapper (:class:`glmflashar.model.modeling_glm_flashar.GlmImageFlashAR`)
conditions on ``text_input_ids`` -- the GLM prompt prefix that precedes the
large (32x32) image-token grid it predicts -- plus a ``text_attention_mask``.

GLM-Image builds the t2i prompt inside ``GlmImageProcessor`` (see
``transformers/models/glm_image/processing_glm_image.py::_build_prompt_with_target_shape``).
For a text-to-image target of size ``target_h x target_w`` the templated prompt is::

    f"{caption}{grid_bos}{token_h} {token_w}{grid_eos}"
    f"{grid_bos}{prev_h} {prev_w}{grid_eos}{bos}"

where ``factor = 32``, ``token_h = target_h // 32``, ``token_w = target_w // 32``
(so 1024x1024 -> token_h = token_w = 32), and ``prev_h/prev_w`` describe a small
preview grid (16x16 for a square target). ``grid_bos = <sop>``,
``grid_eos = <eop>`` and ``bos = <|dit_token_16384|>``. We obtain exactly this
token sequence by calling ``processor.apply_chat_template(...)`` with
``target_h/target_w`` -- the same code path the diffusers ``GlmImagePipeline``
uses at inference time.

Training-data note:
    At inference GLM generates the *preview* grid (256 tokens for a square
    target) BETWEEN this prefix and the large 32x32 grid -- i.e. the true
    conditioning context for the large grid also contains those 256 sampled
    preview tokens (``_compute_generation_params`` reports
    ``large_image_start_offset = prod(preview_grid) = 256``). Those preview
    tokens are sampled at runtime and are NOT part of the image-VQ pretokenized
    dataset (which stores only the d32 large grid). Training batches therefore
    still build the text + shape-declaration prefix here. Evaluation/generation
    separately generates the preview grid with stock AR and passes those preview
    tokens as image-prefix conditioning to FlashAR.

Padding note:
    The wrapper's ``_split_hidden_states`` gathers the hidden state at index
    ``valid_len - 1`` per sample, i.e. it assumes RIGHT padding (valid tokens
    first, pad tokens appended at the end). We therefore tokenize each caption
    individually and right-pad the batch here (rather than using the
    processor's left-padded batch mode).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


def build_glm_processor(model_path: str, subfolder: str = "processor"):
    """Load the GLM-Image processor used to build t2i prompts."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        model_path, subfolder=subfolder, trust_remote_code=True
    )


def _resolve_pad_id(processor, pad_token_id: Optional[int]) -> int:
    if pad_token_id is not None:
        return int(pad_token_id)
    tok = processor.tokenizer
    pid = getattr(tok, "pad_token_id", None)
    if pid is None:
        # Fall back to the DiT bos token id; it is masked out anyway.
        pid = tok.convert_tokens_to_ids(processor.bos_token)
    return int(pid)


def encode_glm_prefix_ids(
    processor,
    caption: str,
    target_h: int = 1024,
    target_w: int = 1024,
) -> torch.Tensor:
    """Return the 1-D text-prefix token ids for a single caption.

    Uses ``apply_chat_template`` with ``target_h/target_w`` so the returned ids
    match GLM-Image's t2i prompt (caption + shape declarations + ``<bos>``).
    """
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": caption}]}
    ]
    out = processor.apply_chat_template(
        conversation,
        tokenize=True,
        target_h=target_h,
        target_w=target_w,
        return_dict=True,
        return_tensors="pt",
    )
    return out["input_ids"][0].long()


def pad_prefix_ids(
    prefix_ids: List[torch.Tensor], pad_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of 1-D id tensors into ``(B, T)`` ids + attention mask."""
    max_len = max(int(ids.numel()) for ids in prefix_ids)
    padded = []
    mask = []
    for ids in prefix_ids:
        valid = int(ids.numel())
        pad_len = max_len - valid
        if pad_len > 0:
            pad = torch.full((pad_len,), int(pad_id), dtype=ids.dtype)
            ids = torch.cat([ids, pad], dim=0)
        padded.append(ids)
        attn = torch.zeros((max_len,), dtype=torch.long)
        attn[:valid] = 1
        mask.append(attn)
    return torch.stack(padded, dim=0), torch.stack(mask, dim=0)


def build_glm_text_prefix(
    processor,
    captions: List[str],
    target_h: int = 1024,
    target_w: int = 1024,
    pad_token_id: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build ``(text_input_ids, text_attention_mask)`` for a batch of captions.

    Args:
        processor: a ``GlmImageProcessor`` (see :func:`build_glm_processor`).
        captions: list of ``B`` caption strings.
        target_h, target_w: target image size (1024 -> a 32x32 large grid).
        pad_token_id: id used for right padding; defaults to the tokenizer pad id.
        device: optional device to move the returned tensors to.

    Returns:
        text_input_ids: ``(B, T)`` LongTensor prefix ids (right-padded).
        text_attention_mask: ``(B, T)`` LongTensor, 1 for valid tokens.
    """
    if not captions:
        raise ValueError("captions must be a non-empty list.")
    pad_id = _resolve_pad_id(processor, pad_token_id)
    prefix_ids = [
        encode_glm_prefix_ids(processor, c, target_h=target_h, target_w=target_w)
        for c in captions
    ]
    text_input_ids, text_attention_mask = pad_prefix_ids(prefix_ids, pad_id)
    if device is not None:
        text_input_ids = text_input_ids.to(device)
        text_attention_mask = text_attention_mask.to(device)
    return text_input_ids, text_attention_mask


__all__ = [
    "build_glm_processor",
    "encode_glm_prefix_ids",
    "pad_prefix_ids",
    "build_glm_text_prefix",
]
