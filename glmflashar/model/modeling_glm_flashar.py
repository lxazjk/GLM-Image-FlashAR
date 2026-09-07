# -*- coding: utf-8 -*-
"""FlashAR wrapper for GLM-Image.

Ports the Emu3.5 FlashAR architecture onto GLM-Image's inner AR
transformer (``GlmImageTextModel``).

FlashAR accelerates raster-scan AR image-token generation by anti-diagonal
parallel decoding: a vertical prediction head + a small vertical transformer
block + a learned fusion gate let an H*W token grid be produced in H+W-1
diagonal steps instead of H*W autoregressive steps. It is trained teacher-forced
under an anti-diagonal proximity attention mask.

GLM-specific deviations vs. the Emu reference (documented inline):
  * Decoder layers return the hidden tensor directly (not ``(hidden, cache)``)
    and update the ``DynamicCache`` in place.
  * Layers consume ``position_embeddings=(cos, sin)`` (3D M-RoPE) rather than a
    1-D ``position_ids``; we recompute them via ``language_model.rotary_emb``.
  * Image grid cell (row, col) gets M-RoPE position (T=P, H=P+row, W=P+col)
    where P = each sample's valid prefix length plus the skipped preview-grid
    extent (matches ``GlmImageModel.get_rope_index`` large-grid decode).
  * ``GlmImageRMSNorm`` replaces ``Emu3RMSNorm``.
  * Sampled logits are restricted to codebook ids ``[0, codebook_size)``.

Usage:
    from glmflashar.model.glm_backbone_patch import apply_glm_backbone_patch
    apply_glm_backbone_patch()
    wrapper = GlmImageFlashAR(model.model.language_model, vocab_size=16512,
                              hidden_size=4096, lm_head=model.lm_head)
    # Training:
    out = wrapper(input_ids=image_ids, height=H, width=W, text_input_ids=text_ids)
    out["loss"].backward()
    # Generation:
    grid = wrapper.generate(height=H, width=W, device=device, text_input_ids=text_ids)
"""

from __future__ import annotations

import copy
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache
from transformers.models.glm_image.modeling_glm_image import GlmImageRMSNorm
from glmflashar.utils.preview import stock_preview_shape_from_large_grid


# ============================================================================
# Sampling helpers (identical to the Emu reference)
# ============================================================================

def _top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


def _sample_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    sample_logits: bool = True,
) -> torch.Tensor:
    idx = []
    for i in range(logits.shape[1]):
        one_logits = logits[:, i, :] / max(temperature, 1e-5)
        if top_k > 0 or top_p < 1.0:
            one_logits = _top_k_top_p_filtering(one_logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(one_logits.float(), dim=-1)
        if sample_logits:
            one_idx = torch.multinomial(probs, num_samples=1)
        else:
            _, one_idx = torch.topk(probs, k=1, dim=-1)
        idx.append(one_idx.view(-1, 1))
    return torch.cat(idx, dim=-1)


def _validate_sampling_args(temperature: float, top_k: int, top_p: float) -> None:
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError(f"temperature must be finite and > 0, got {temperature}.")
    if int(top_k) < 0:
        raise ValueError(f"top_k must be >= 0, got {top_k}.")
    if not math.isfinite(float(top_p)) or not (0.0 < float(top_p) <= 1.0):
        raise ValueError(f"top_p must be finite and in (0, 1], got {top_p}.")


def _validate_model_sizes(vocab_size: int, hidden_size: int, codebook_size: int, mask_token_id) -> None:
    if int(vocab_size) <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}.")
    if int(hidden_size) <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}.")
    if int(codebook_size) <= 0:
        raise ValueError(f"codebook_size must be positive, got {codebook_size}.")
    if int(codebook_size) > int(vocab_size):
        raise ValueError(
            f"codebook_size must be <= vocab_size, got {codebook_size} > {vocab_size}."
        )
    if mask_token_id is not None and not (0 <= int(mask_token_id) < int(codebook_size)):
        raise ValueError(
            f"mask_token_id must be in [0, codebook_size), got {mask_token_id}."
        )


# ============================================================================
# Anti-diagonal proximity mask builders (model-agnostic)
# ============================================================================

def _build_step_id(height: int, width: int, device: torch.device) -> torch.Tensor:
    """Anti-diagonal step id (row + col) for each position in the H*W grid."""
    rows = torch.arange(height, device=device).unsqueeze(1).expand(height, width)
    cols = torch.arange(width, device=device).unsqueeze(0).expand(height, width)
    return (rows + cols).reshape(-1)


def _build_proximity_allow(
    height: int, width: int, device: torch.device
) -> torch.Tensor:
    """Boolean allow matrix: allow[q, k] = True iff cell k may be attended by q.

    A cell attends every cell with step_id <= its own step_id.
    """
    total = height * width
    allow = torch.zeros((total, total), device=device, dtype=torch.bool)
    previous: List[int] = []
    for c in range(height + width - 1):
        current: List[int] = []
        for h in range(height):
            w = c - h
            if 0 <= w < width:
                idx = h * width + w
                current.append(idx)
                previous.append(idx)
        for idx in current:
            allow[idx, previous] = True
    return allow


def build_t2i_flashar_mask(
    prefix_ids: Optional[torch.Tensor],
    height: int,
    width: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    text_attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a 4-D additive mask for pure T2I:
      - text prefix uses a causal mask
      - image tokens attend all valid prefix tokens
      - image block uses the anti-diagonal proximity mask
    Returns (attn_mask [B,1,S,S] with 0 allowed / -inf blocked, step_id [H*W]).
    """
    if prefix_ids is None:
        prefix_ids = torch.empty((batch_size, 0), device=device, dtype=torch.long)
    if prefix_ids.dim() == 1:
        prefix_ids = prefix_ids.unsqueeze(0)
    if prefix_ids.size(0) != batch_size:
        raise ValueError(
            f"prefix_ids batch size mismatch: expected {batch_size}, got {prefix_ids.size(0)}."
        )
    prefix_len = prefix_ids.size(1)
    if text_attention_mask is not None:
        if text_attention_mask.dim() == 1:
            text_attention_mask = text_attention_mask.unsqueeze(0)
        if text_attention_mask.size(0) != batch_size:
            raise ValueError(
                "text_attention_mask batch size mismatch: "
                f"expected {batch_size}, got {text_attention_mask.size(0)}."
            )
        if text_attention_mask.size(1) < prefix_len:
            raise ValueError("text_attention_mask length is smaller than prefix length.")
        text_attention_mask = text_attention_mask[:, :prefix_len]

    t_steps = height * width
    total = prefix_len + t_steps
    attn = torch.full(
        (batch_size, 1, total, total), float("-inf"), device=device, dtype=dtype
    )

    if prefix_len > 0:
        causal = torch.tril(
            torch.ones(prefix_len, prefix_len, device=device, dtype=torch.bool)
        )
        diag = torch.eye(prefix_len, device=device, dtype=torch.bool)
        for b in range(batch_size):
            allow = causal.clone()
            if text_attention_mask is not None:
                valid = text_attention_mask[b].to(device=device).bool()
                allow &= valid.unsqueeze(0) & valid.unsqueeze(1)
                allow |= diag
                attn[b, 0, prefix_len:, :prefix_len] = torch.where(
                    valid.unsqueeze(0), 0.0, float("-inf")
                )
            else:
                attn[b, 0, prefix_len:, :prefix_len] = 0.0
            attn[b, 0, :prefix_len, :prefix_len].masked_fill_(allow, 0.0)

    allowed = _build_proximity_allow(height, width, device)
    attn[:, 0, prefix_len:, prefix_len:].masked_fill_(allowed, 0.0)
    step_id = _build_step_id(height, width, device)
    return attn, step_id


# ============================================================================
# GlmImageFlashAR
# ============================================================================

class GlmImageFlashAR(nn.Module):
    """Wrap the GLM-Image AR backbone for FlashAR training and diagonal decoding.

    Architecture:
        backbone[:vertical_start_layer]  -> captured hidden (shared)
        backbone[vertical_start_layer:]  -> horizontal_head -> h_logits (right nb)
        vertical_block (copy of top layers on captured hidden)
                                         -> vertical_head   -> v_logits (down nb)
        fused_logits = shift-and-merge(h_logits, v_logits) via learned gate
    """

    def __init__(
        self,
        language_model: nn.Module,
        vocab_size: int = 16512,
        hidden_size: int = 4096,
        pad_token_id: int = -100,
        mask_token_id: Optional[int] = None,
        codebook_size: int = 16384,
        use_vertical_block: bool = True,
        vertical_layers: int = 4,
        vertical_start_layer: int = -1,
        lm_head: Optional[nn.Linear] = None,
    ) -> None:
        super().__init__()
        _validate_model_sizes(vocab_size, hidden_size, codebook_size, mask_token_id)
        self.backbone = language_model
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.pad_token_id = pad_token_id
        self.mask_token_id = (
            mask_token_id if mask_token_id is not None else (codebook_size - 1)
        )
        self.codebook_size = int(codebook_size)
        self.use_vertical_block = use_vertical_block

        cfg = getattr(language_model, "config", None)
        backbone_layers = getattr(language_model, "layers", None)
        self.backbone_num_layers = int(len(backbone_layers)) if backbone_layers is not None else 0
        if use_vertical_block and int(vertical_layers) <= 0:
            raise ValueError(
                "vertical_layers must be positive when use_vertical_block=True; "
                "pass use_vertical_block=False to disable the vertical block."
            )
        self.vertical_layers = int(vertical_layers) if use_vertical_block else 0
        self.vertical_start_layer = self.backbone_num_layers
        if use_vertical_block:
            if backbone_layers is None or self.backbone_num_layers <= 0:
                raise ValueError("language_model.layers is required for vertical_block.")
            if vertical_start_layer < 0:
                vertical_start_layer = self.backbone_num_layers - self.vertical_layers
            self.vertical_start_layer = int(vertical_start_layer)
            if self.vertical_start_layer < 0:
                raise ValueError(
                    f"vertical_start_layer must be >= 0, got {self.vertical_start_layer}."
                )
            if self.vertical_start_layer + self.vertical_layers > self.backbone_num_layers:
                raise ValueError(
                    "vertical_start_layer + vertical_layers exceeds backbone depth: "
                    f"start={self.vertical_start_layer} layers={self.vertical_layers} "
                    f"backbone_num_layers={self.backbone_num_layers}."
                )

        # --- prediction heads ---
        self.horizontal_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.vertical_head = nn.Linear(hidden_size, vocab_size, bias=False)

        if use_vertical_block:
            # Reuse GLM transformer parameters by deep-copying the top decoder
            # layers. Re-index their KV cache slots to 0..vertical_layers-1 so the
            # vertical DynamicCache is contiguous and independent of the backbone.
            self.vertical_block = nn.ModuleList(
                [
                    copy.deepcopy(layer)
                    for layer in backbone_layers[
                        self.vertical_start_layer : self.vertical_start_layer + self.vertical_layers
                    ]
                ]
            )
            for i, layer in enumerate(self.vertical_block):
                if hasattr(layer, "self_attn"):
                    if hasattr(layer.self_attn, "is_causal"):
                        layer.self_attn.is_causal = False
                    if hasattr(layer.self_attn, "layer_idx"):
                        layer.self_attn.layer_idx = i
            rms_eps = float(getattr(cfg, "rms_norm_eps", 1e-5))
            self.vertical_norm = GlmImageRMSNorm(hidden_size, eps=rms_eps)
        else:
            self.vertical_block = None
            self.vertical_norm = None

        # Per-position h/v gate on the SAME target-position features (keeps train
        # and inference consistent).
        gate_proj_dim = max(64, hidden_size // 8)
        self.hv_gate_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, gate_proj_dim, bias=False),
            nn.SiLU(),
            nn.Linear(gate_proj_dim, 1, bias=True),
        )
        self.hv_gate_corner = nn.Linear(hidden_size, 1, bias=True)
        # Start from a symmetric mix (sigmoid(0) = 0.5).
        nn.init.zeros_(self.hv_gate_mlp[-1].weight)
        nn.init.zeros_(self.hv_gate_mlp[-1].bias)
        nn.init.zeros_(self.hv_gate_corner.weight)
        nn.init.zeros_(self.hv_gate_corner.bias)

        # initialise heads from lm_head if provided
        if lm_head is not None:
            with torch.no_grad():
                self.horizontal_head.weight.copy_(lm_head.weight)
                self.vertical_head.weight.copy_(lm_head.weight)

        self._sync_dtype_with_backbone()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sync_dtype_with_backbone(self) -> None:
        try:
            backbone_dtype = next(self.backbone.parameters()).dtype
        except StopIteration:
            return
        self.horizontal_head.to(dtype=backbone_dtype)
        self.vertical_head.to(dtype=backbone_dtype)
        if self.vertical_block is not None:
            self.vertical_block.to(dtype=backbone_dtype)
        if self.vertical_norm is not None:
            self.vertical_norm.to(dtype=backbone_dtype)

    def _vertical_capture_layer(self) -> Optional[int]:
        if self.vertical_block is None:
            return None
        if self.vertical_start_layer >= self.backbone_num_layers:
            return None
        return int(self.vertical_start_layer)

    def _reshape_grid(self, seq: torch.Tensor, height: int, width: int) -> torch.Tensor:
        bsz, seq_len = seq.shape[:2]
        if seq_len != height * width:
            raise ValueError(f"Expected seq_len={height * width}, got {seq_len}")
        return seq.view(bsz, height, width, -1)

    @staticmethod
    def _validate_image_ids_grid(input_ids: torch.Tensor, height: int, width: int) -> None:
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2-D (B, H*W), got shape {tuple(input_ids.shape)}.")
        if int(height) <= 0 or int(width) <= 0:
            raise ValueError(f"height and width must be positive, got {height}x{width}.")
        expected = int(height) * int(width)
        if int(input_ids.size(1)) != expected:
            raise ValueError(
                f"input_ids length must equal height*width={expected}, got {int(input_ids.size(1))}."
            )

    # ---- M-RoPE position id builders (GLM-specific) -------------------

    def _image_position_ids(
        self,
        prefix_len,
        rows: torch.Tensor,
        cols: torch.Tensor,
        batch_size: int,
        device: torch.device,
        image_offset: int = 0,
    ) -> torch.Tensor:
        """3-D M-RoPE position ids (3, B, N) for a set of image grid cells.

        Follows GlmImageModel.get_rope_index decode convention:
        cell (row, col) -> (T=base, H=base+row, W=base+col), where base is the
        per-sample effective prefix length plus the skipped preview grid's
        max(H,W) extent for GLM T2I.
        """
        rows = rows.to(device=device, dtype=torch.long)
        cols = cols.to(device=device, dtype=torch.long)
        n = rows.numel()
        pos = torch.empty((3, batch_size, n), device=device, dtype=torch.long)
        base = torch.as_tensor(prefix_len, device=device, dtype=torch.long)
        if base.ndim == 0:
            base = base.view(1).expand(batch_size)
        elif base.numel() != batch_size:
            raise ValueError(
                f"prefix_len tensor must have batch_size={batch_size} values, got {base.numel()}."
            )
        base = base.view(batch_size, 1) + int(image_offset)
        pos[0] = base.expand(batch_size, n)
        pos[1] = base + rows.view(1, n)
        pos[2] = base + cols.view(1, n)
        return pos

    @staticmethod
    def _effective_prefix_lens(
        prefix_len: int,
        batch_size: int,
        device: torch.device,
        text_attention_mask: Optional[torch.Tensor] = None,
    ):
        """Return per-sample prefix lengths for image RoPE bases.

        Training right-pads text prefixes before appending image tokens. GLM's
        image M-RoPE base should follow the valid text prefix, not the batch's
        padded max length; otherwise shorter captions train at positions that
        do not match single-sample inference.
        """
        if prefix_len <= 0:
            return 0
        if text_attention_mask is None:
            return int(prefix_len)
        if text_attention_mask.dim() == 1:
            text_attention_mask = text_attention_mask.unsqueeze(0)
        if text_attention_mask.size(0) != batch_size:
            raise ValueError("text_attention_mask batch size mismatch.")
        if text_attention_mask.size(1) < prefix_len:
            raise ValueError("text_attention_mask length is smaller than prefix length.")
        valid_lens = text_attention_mask[:, :prefix_len].to(device=device).long().sum(dim=1)
        if (valid_lens <= 0).any():
            raise ValueError("text_attention_mask must contain at least one valid prefix token per sample.")
        return valid_lens.clamp(min=1, max=prefix_len)

    @staticmethod
    def _preview_grid_shape(height: int, width: int) -> Tuple[int, int]:
        """Return stock GLM preview grid dims for a d32 large grid.

        ``GlmImageProcessor._build_prompt_with_target_shape`` computes preview
        dimensions from the large-grid aspect ratio, not by simple halving:
        ``int(sqrt(token_h/token_w) * 16)`` and its reciprocal.
        """
        return stock_preview_shape_from_large_grid(height, width)

    @staticmethod
    def _skipped_preview_offset(height: int, width: int) -> int:
        """M-RoPE offset of the skipped preview grid before the large grid.

        Stock GLM T2I declares image grids as [large, preview] but generates
        preview first; get_rope_index advances decode_pos by max(preview_h,
        preview_w), not by preview token count.
        """
        preview_h, preview_w = GlmImageFlashAR._preview_grid_shape(height, width)
        return max(preview_h, preview_w)

    def _full_position_ids(
        self,
        prefix_len: int,
        height: int,
        width: int,
        batch_size: int,
        device: torch.device,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """3-D M-RoPE position ids (3, B, prefix_len + H*W) for the training path."""
        total = prefix_len + height * width
        pos = torch.zeros((3, batch_size, total), device=device, dtype=torch.long)
        if prefix_len > 0:
            idx = torch.arange(prefix_len, device=device, dtype=torch.long)
            pos[:, :, :prefix_len] = idx.view(1, 1, prefix_len)
        rows = torch.arange(height, device=device).unsqueeze(1).expand(height, width).reshape(-1)
        cols = torch.arange(width, device=device).unsqueeze(0).expand(height, width).reshape(-1)
        prefix_base = self._effective_prefix_lens(
            prefix_len, batch_size, device, text_attention_mask=text_attention_mask
        )
        img = self._image_position_ids(
            prefix_base,
            rows,
            cols,
            batch_size,
            device,
            image_offset=self._skipped_preview_offset(height, width),
        )
        pos[:, :, prefix_len:] = img
        return pos

    # ---- gates --------------------------------------------------------

    def _hv_gate_from_pair(
        self, h_feat: torch.Tensor, v_feat: torch.Tensor, out_dtype: torch.dtype
    ) -> torch.Tensor:
        if h_feat.shape != v_feat.shape:
            raise ValueError("h_feat and v_feat must share shape.")
        if h_feat.numel() == 0:
            shape = (*h_feat.shape[:-1], 1)
            return torch.full(shape, 0.5, device=h_feat.device, dtype=out_dtype)
        gate_dtype = self.hv_gate_mlp[0].weight.dtype
        feat = torch.cat([h_feat, v_feat], dim=-1).to(gate_dtype)
        gate = torch.sigmoid(self.hv_gate_mlp(feat))
        return gate.to(dtype=out_dtype)

    def _hv_gate_corner(self, cond_hidden: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        gate_dtype = self.hv_gate_corner.weight.dtype
        gate = torch.sigmoid(self.hv_gate_corner(cond_hidden.to(gate_dtype)))
        return gate.to(dtype=out_dtype)

    @staticmethod
    def _binary_gate_entropy(prob: torch.Tensor) -> torch.Tensor:
        prob = prob.clamp(1e-6, 1.0 - 1e-6)
        entropy = -(prob * prob.log() + (1.0 - prob) * (1.0 - prob).log())
        return entropy / 0.6931471805599453

    @staticmethod
    def _build_step_mask_2d(step_id: torch.Tensor) -> torch.Tensor:
        """Bool mask where True means blocked attention (step_id[k] > step_id[q])."""
        allowed = step_id[None, :] <= step_id[:, None]
        return ~allowed

    @staticmethod
    def _build_step_attn_4d(
        step_mask_2d: torch.Tensor, batch_size: int, dtype: torch.dtype
    ) -> torch.Tensor:
        seq_len = int(step_mask_2d.size(0))
        mask = torch.zeros(
            (batch_size, 1, seq_len, seq_len), device=step_mask_2d.device, dtype=dtype
        )
        if step_mask_2d.any():
            mask_val = torch.finfo(dtype).min
            mask.masked_fill_(step_mask_2d.view(1, 1, seq_len, seq_len), mask_val)
        return mask

    def _fuse_logits(
        self,
        h_logits: torch.Tensor,
        v_logits: torch.Tensor,
        cond_h_logits: torch.Tensor,
        cond_v_logits: torch.Tensor,
        h_hidden: torch.Tensor,
        v_hidden: torch.Tensor,
        cond_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Shift-and-merge horizontal/vertical logits into per-position predictions.

        h_logits[b, h, w] predicts (h, w+1); after roll -> position (h, w).
        v_logits[b, h, w] predicts (h+1, w); after roll -> position (h, w).
        """
        rw_corner = self._hv_gate_corner(cond_hidden, out_dtype=h_logits.dtype)  # [B,1,1]
        cond_logits = (
            rw_corner[:, 0, :] * cond_h_logits[:, 0, :]
            + (1.0 - rw_corner[:, 0, :]) * cond_v_logits[:, 0, :]
        )
        h_shift = h_logits.roll(shifts=1, dims=2)  # prediction for right neighbor
        v_shift = v_logits.roll(shifts=1, dims=1)  # prediction for bottom neighbor
        fused = torch.zeros_like(v_shift)
        fused[:, 0, 0, :] = cond_logits            # corner: corner gate
        fused[:, 0, 1:, :] = h_shift[:, 0, 1:, :]  # first row: horizontal only
        fused[:, 1:, 0, :] = v_shift[:, 1:, 0, :]  # first col: vertical only
        if h_shift.size(1) > 1 and h_shift.size(2) > 1:
            h_feat_shift = h_hidden.roll(shifts=1, dims=2)
            v_feat_shift = v_hidden.roll(shifts=1, dims=1)
            rw_interior = self._hv_gate_from_pair(
                h_feat_shift[:, 1:, 1:, :],
                v_feat_shift[:, 1:, 1:, :],
                out_dtype=h_logits.dtype,
            )
            fused[:, 1:, 1:, :] = (
                rw_interior * h_shift[:, 1:, 1:, :]
                + (1.0 - rw_interior) * v_shift[:, 1:, 1:, :]
            )
            gate_mean = rw_interior.mean().detach().reshape(())
            gate_reg = self._hv_gate_from_pair(
                h_feat_shift[:, 1:, 1:, :].detach(),
                v_feat_shift[:, 1:, 1:, :].detach(),
                out_dtype=h_logits.dtype,
            )
            gate_entropy = self._binary_gate_entropy(gate_reg).mean().reshape(())
        else:
            gate_mean = rw_corner.mean().detach().reshape(())
            gate_reg = self._hv_gate_corner(cond_hidden.detach(), out_dtype=h_logits.dtype)
            gate_entropy = self._binary_gate_entropy(gate_reg).mean().reshape(())
        corner_mean = rw_corner.mean().detach().reshape(())
        gate_stats = {
            "hv_gate_h": gate_mean,
            "hv_gate_v": (1.0 - gate_mean).detach().reshape(()),
            "hv_gate_h_corner": corner_mean,
            "hv_gate_v_corner": (1.0 - corner_mean).detach().reshape(()),
            "hv_gate_entropy": gate_entropy.detach().reshape(()),
            "loss_gate_collapse": (1.0 - gate_entropy).reshape(()),
        }
        return fused, gate_stats

    def _apply_vertical_block(
        self,
        image_hidden: torch.Tensor,
        step_mask_2d: torch.Tensor,
        image_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run the vertical transformer block over image tokens (full grid).

        GLM deviation: decoder layers take position_embeddings=(cos, sin) and
        return the hidden tensor directly.
        """
        if self.vertical_block is None:
            return image_hidden
        bsz = image_hidden.size(0)
        attn_mask = self._build_step_attn_4d(step_mask_2d, bsz, image_hidden.dtype)
        pos_emb = self.backbone.rotary_emb(image_hidden, image_position_ids)
        v_hidden = image_hidden
        for layer in self.vertical_block:
            v_hidden = layer(
                v_hidden,
                position_embeddings=pos_emb,
                attention_mask=attn_mask,
                position_ids=None,
                use_cache=False,
            )
        return self.vertical_norm(v_hidden)

    def _compute_logits(
        self,
        cond_horizontal_hidden: torch.Tensor,
        cond_vertical_hidden: torch.Tensor,
        horizontal_hidden: torch.Tensor,
        vertical_input_hidden: torch.Tensor,
        height: int,
        width: int,
        step_mask_2d: torch.Tensor,
        image_position_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        cond_h_logits = self.horizontal_head(cond_horizontal_hidden)
        cond_v_logits = self.vertical_head(cond_vertical_hidden)
        h_grid = self._reshape_grid(horizontal_hidden, height, width)
        h_logits = self.horizontal_head(h_grid)

        v_hidden = self._apply_vertical_block(
            vertical_input_hidden, step_mask_2d, image_position_ids
        )
        v_grid = self._reshape_grid(v_hidden, height, width)
        v_logits = self.vertical_head(v_grid)

        fused, gate_stats = self._fuse_logits(
            h_logits,
            v_logits,
            cond_h_logits,
            cond_v_logits,
            h_grid,
            v_grid,
            cond_horizontal_hidden,
        )
        return {"fused": fused, "h_logits": h_logits, "v_logits": v_logits, **gate_stats}

    def _run_backbone(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        capture_layer = self._vertical_capture_layer()
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            capture_hidden_before_layer=capture_layer,
        )
        last_hidden = outputs.last_hidden_state
        vertical_input_hidden = (
            outputs.captured_hidden_state if capture_layer is not None else last_hidden
        )
        if vertical_input_hidden is None:
            raise RuntimeError(
                "backbone did not return captured_hidden_state for vertical branch."
            )
        return last_hidden, vertical_input_hidden

    def _split_hidden_states(
        self,
        hidden: torch.Tensor,
        prefix_len: int,
        text_attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split backbone hidden into (cond corner state, image states)."""
        if prefix_len <= 0:
            return hidden[:, :1, :], hidden

        prefix_hidden = hidden[:, :prefix_len, :]
        image_hidden = hidden[:, prefix_len:, :]
        if text_attention_mask is None:
            return prefix_hidden[:, -1:, :], image_hidden

        if text_attention_mask.dim() == 1:
            text_attention_mask = text_attention_mask.unsqueeze(0)
        if text_attention_mask.size(0) != hidden.size(0):
            raise ValueError("text_attention_mask batch size mismatch.")
        if text_attention_mask.size(1) < prefix_len:
            raise ValueError("text_attention_mask length is smaller than prefix length.")

        valid_lens = text_attention_mask[:, :prefix_len].to(device=hidden.device).long().sum(dim=1)
        if (valid_lens <= 0).any():
            raise ValueError("text_attention_mask must contain at least one valid prefix token per sample.")
        valid_lens = valid_lens.clamp(min=1, max=prefix_len)
        gather_idx = (valid_lens - 1).view(-1, 1, 1).expand(-1, 1, prefix_hidden.size(-1))
        cond_hidden = prefix_hidden.gather(dim=1, index=gather_idx)
        return cond_hidden, image_hidden

    def _cross_entropy_4d(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        chunked_loss: bool,
        chunk_rows: int = 4,
    ) -> torch.Tensor:
        """Cross entropy over [B, H, W, V] logits, optionally row-chunked."""
        # GLM image targets are codebook ids in [0, codebook_size). The extra
        # vocabulary slots should not compete in the training softmax.
        logits = logits[..., : self.codebook_size]
        if not chunked_loss:
            return F.cross_entropy(
                logits.reshape(-1, self.codebook_size).float(),
                targets.reshape(-1),
                ignore_index=self.pad_token_id,
            )

        height = int(logits.size(1))
        loss_sum = logits.new_zeros((), dtype=torch.float32)
        token_count = 0
        chunk_rows = max(1, int(chunk_rows))
        for row_start in range(0, height, chunk_rows):
            row_end = min(row_start + chunk_rows, height)
            chunk_logits = logits[:, row_start:row_end, :, :].reshape(-1, self.codebook_size)
            chunk_targets = targets[:, row_start:row_end, :].reshape(-1)
            valid = chunk_targets.ne(self.pad_token_id)
            valid_tokens = int(valid.sum().item())
            if valid_tokens == 0:
                continue
            chunk_loss = F.cross_entropy(
                chunk_logits.float(),
                chunk_targets,
                ignore_index=self.pad_token_id,
                reduction="sum",
            )
            loss_sum = loss_sum + chunk_loss
            token_count += valid_tokens

        if token_count == 0:
            return logits.new_zeros(())
        return loss_sum / float(token_count)

    # ------------------------------------------------------------------
    # Mask helper
    # ------------------------------------------------------------------

    def build_mask(
        self,
        image_ids: torch.Tensor,
        height: int,
        width: int,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        mask_dtype = next(self.backbone.parameters()).dtype
        if text_input_ids is not None:
            if text_input_ids.dim() == 1:
                text_input_ids = text_input_ids.unsqueeze(0)
            if text_input_ids.size(0) != image_ids.size(0):
                raise ValueError(
                    f"text_input_ids batch size mismatch: expected {image_ids.size(0)}, "
                    f"got {text_input_ids.size(0)}."
                )
            prefix_len = text_input_ids.size(1)
            full_input_ids = torch.cat([text_input_ids, image_ids], dim=1)
            attn_mask, step_id = build_t2i_flashar_mask(
                prefix_ids=text_input_ids,
                height=height,
                width=width,
                batch_size=image_ids.size(0),
                device=image_ids.device,
                dtype=mask_dtype,
                text_attention_mask=text_attention_mask,
            )
        else:
            prefix_len = 0
            full_input_ids = image_ids
            attn_mask, step_id = build_t2i_flashar_mask(
                prefix_ids=None,
                height=height,
                width=width,
                batch_size=image_ids.size(0),
                device=image_ids.device,
                dtype=mask_dtype,
            )
        step_mask_2d = self._build_step_mask_2d(step_id)
        return full_input_ids, prefix_len, attn_mask, step_mask_2d

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        height: int,
        width: int,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        preview_input_ids: Optional[torch.Tensor] = None,
        preview_height: int = 0,
        preview_width: int = 0,
        chunked_loss: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Compute the fused FlashAR loss.

        Args:
            input_ids: (B, H*W) flattened image token ids (codebook ids).
            height, width: spatial dims of the image grid.
            text_input_ids: (B, T) optional text prefix.
            text_attention_mask: (B, T) optional prefix padding mask.
        """
        self._validate_image_ids_grid(input_ids, height, width)
        bsz = input_ids.size(0)
        device = input_ids.device
        if text_input_ids is not None:
            if text_input_ids.dim() == 1:
                text_input_ids = text_input_ids.unsqueeze(0)
            text_input_ids = text_input_ids.to(device=device, dtype=torch.long)
        if text_attention_mask is not None:
            if text_attention_mask.dim() == 1:
                text_attention_mask = text_attention_mask.unsqueeze(0)
            text_attention_mask = text_attention_mask.to(device=device, dtype=torch.long)

        preview_len = 0
        if preview_input_ids is not None:
            if text_input_ids is None:
                raise ValueError("preview_input_ids requires text_input_ids.")
            if text_attention_mask is None:
                text_attention_mask = torch.ones_like(text_input_ids, dtype=torch.long)
            if preview_input_ids.dim() == 3:
                preview_height = int(preview_input_ids.size(1))
                preview_width = int(preview_input_ids.size(2))
                preview_input_ids = preview_input_ids.reshape(preview_input_ids.size(0), -1)
            elif preview_input_ids.dim() == 1:
                preview_input_ids = preview_input_ids.unsqueeze(0)
            elif preview_input_ids.dim() != 2:
                raise ValueError(
                    f"preview_input_ids must be 1-D, 2-D or 3-D, got {tuple(preview_input_ids.shape)}."
                )
            preview_input_ids = preview_input_ids.to(device=device, dtype=torch.long)
            if preview_input_ids.size(0) != bsz:
                raise ValueError("preview_input_ids batch size mismatch.")
            preview_len = int(preview_input_ids.size(1))
            expected_preview = int(preview_height) * int(preview_width)
            if expected_preview <= 0 or preview_len != expected_preview:
                raise ValueError(
                    f"preview_input_ids length must equal preview_height*preview_width="
                    f"{expected_preview}, got {preview_len}."
                )
            text_prefix_len = int(text_input_ids.size(1))
            full_input_ids = torch.cat([text_input_ids, preview_input_ids, input_ids], dim=1)
            known_positions = torch.arange(height * width, device=device, dtype=torch.long)
            prefix_base = self._effective_prefix_lens(
                text_prefix_len, bsz, device, text_attention_mask=text_attention_mask
            )
            attn_mask = self._build_compact_generation_mask(
                text_attention_mask=text_attention_mask.to(device=device),
                known_positions=known_positions,
                width=width,
                dtype=next(self.backbone.parameters()).dtype,
                preview_len=preview_len,
            )
            position_ids = self._compact_position_ids_with_preview(
                text_prefix_len=text_prefix_len,
                known_positions=known_positions,
                width=width,
                batch_size=bsz,
                device=device,
                image_offset=self._skipped_preview_offset(height, width),
                text_prefix_base=prefix_base,
                preview_height=int(preview_height),
                preview_width=int(preview_width),
            )
            step_mask_2d = self._build_step_mask_2d(_build_step_id(height, width, device))
            total_prefix_len = text_prefix_len + preview_len
        else:
            full_input_ids, prefix_len, attn_mask, step_mask_2d = self.build_mask(
                image_ids=input_ids,
                height=height,
                width=width,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
            )
            position_ids = self._full_position_ids(
                prefix_len,
                height,
                width,
                bsz,
                device,
                text_attention_mask=text_attention_mask,
            )
            text_prefix_len = prefix_len
            total_prefix_len = prefix_len

        hidden, vertical_input_hidden = self._run_backbone(
            full_input_ids, attn_mask, position_ids
        )
        if preview_len > 0:
            cond_horizontal_hidden, image_hidden = self._split_generation_hidden_states(
                hidden, text_prefix_len, total_prefix_len, text_attention_mask
            )
            cond_vertical_hidden, vertical_image_hidden = self._split_generation_hidden_states(
                vertical_input_hidden, text_prefix_len, total_prefix_len, text_attention_mask
            )
        else:
            cond_horizontal_hidden, image_hidden = self._split_hidden_states(
                hidden=hidden, prefix_len=text_prefix_len, text_attention_mask=text_attention_mask
            )
            cond_vertical_hidden, vertical_image_hidden = self._split_hidden_states(
                hidden=vertical_input_hidden,
                prefix_len=text_prefix_len,
                text_attention_mask=text_attention_mask,
            )

        prefix_base = self._effective_prefix_lens(
            text_prefix_len, bsz, device, text_attention_mask=text_attention_mask
        )
        image_position_ids = self._image_position_ids(
            prefix_base,
            torch.arange(height, device=device).unsqueeze(1).expand(height, width).reshape(-1),
            torch.arange(width, device=device).unsqueeze(0).expand(height, width).reshape(-1),
            bsz,
            device,
            image_offset=self._skipped_preview_offset(height, width),
        )

        logits = self._compute_logits(
            cond_horizontal_hidden,
            cond_vertical_hidden,
            image_hidden,
            vertical_image_hidden,
            height,
            width,
            step_mask_2d,
            image_position_ids,
        )
        fused = logits["fused"]
        h_logits = logits["h_logits"]
        v_logits = logits["v_logits"]

        target_grid = input_ids.view(bsz, height, width)
        loss = self._cross_entropy_4d(fused, target_grid, chunked_loss=chunked_loss)

        if width > 1:
            loss_h = self._cross_entropy_4d(
                h_logits[:, :, :-1, :], target_grid[:, :, 1:], chunked_loss=chunked_loss
            )
        else:
            loss_h = torch.zeros((), device=device, dtype=loss.dtype)

        if height > 1:
            loss_v = self._cross_entropy_4d(
                v_logits[:, :-1, :, :], target_grid[:, 1:, :], chunked_loss=chunked_loss
            )
        else:
            loss_v = torch.zeros((), device=device, dtype=loss.dtype)

        return {
            "loss": loss,
            "loss_h": loss_h,
            "loss_v": loss_v,
            "logits": fused,
            "logits_h": h_logits,
            "logits_v": v_logits,
            "hv_gate_h": logits["hv_gate_h"],
            "hv_gate_v": logits["hv_gate_v"],
            "hv_gate_h_corner": logits["hv_gate_h_corner"],
            "hv_gate_v_corner": logits["hv_gate_v_corner"],
            "hv_gate_entropy": logits["hv_gate_entropy"],
            "loss_gate_collapse": logits["loss_gate_collapse"],
        }

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_text_batch(
        text_input_ids: Optional[torch.Tensor],
        text_attention_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if text_input_ids is None:
            return None, None
        if text_input_ids.dim() == 1:
            text_input_ids = text_input_ids.unsqueeze(0)
        if text_attention_mask is None:
            text_attention_mask = torch.ones_like(text_input_ids, dtype=torch.long)
        elif text_attention_mask.dim() == 1:
            text_attention_mask = text_attention_mask.unsqueeze(0)
        if text_attention_mask.size(0) != text_input_ids.size(0):
            raise ValueError("text_attention_mask batch size mismatch.")
        if text_attention_mask.size(1) < text_input_ids.size(1):
            raise ValueError("text_attention_mask length is smaller than text_input_ids.")
        text_attention_mask = text_attention_mask.to(
            device=text_input_ids.device, dtype=torch.long
        )[:, : text_input_ids.size(1)]
        if text_attention_mask.long().sum(dim=1).le(0).any():
            raise ValueError("text_attention_mask must contain at least one valid prefix token per sample.")
        return text_input_ids, text_attention_mask

    @staticmethod
    def _gather_last_valid_hidden(
        hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states[:, -1:, :]
        valid_lens = attention_mask.long().sum(dim=1)
        if (valid_lens <= 0).any():
            raise ValueError("attention_mask must contain at least one valid token per sample.")
        valid_lens = valid_lens.clamp(min=1, max=hidden_states.size(1))
        gather_idx = (valid_lens - 1).view(-1, 1, 1).expand(-1, 1, hidden_states.size(-1))
        return hidden_states.gather(dim=1, index=gather_idx)

    def _build_prefix_causal_mask_4d(
        self,
        prefix_attention_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """4-D additive causal mask for the text prefix (respecting padding)."""
        device = prefix_attention_mask.device
        bsz, prefix_len = prefix_attention_mask.shape
        causal = torch.tril(
            torch.ones(prefix_len, prefix_len, device=device, dtype=torch.bool)
        )
        attn = torch.full(
            (bsz, 1, prefix_len, prefix_len), torch.finfo(dtype).min, device=device, dtype=dtype
        )
        eye = torch.eye(prefix_len, device=device, dtype=torch.bool)
        for b in range(bsz):
            valid = prefix_attention_mask[b].to(device=device).bool()
            allow = causal & valid.unsqueeze(0) & valid.unsqueeze(1)
            allow |= eye
            attn[b, 0].masked_fill_(allow, 0.0)
        return attn

    def _build_kv_attention_mask(
        self,
        batch_size: int,
        current_len: int,
        past_len: int,
        device: torch.device,
        dtype: torch.dtype,
        prefix_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Additive mask (B,1,current_len,total_kv). All cached tokens are allowed
        (proximity holds since cache only contains step_id <= current); only padded
        prefix positions are blocked."""
        prefix_len = 0 if prefix_attention_mask is None else int(prefix_attention_mask.size(1))
        total_kv = prefix_len + past_len + current_len
        attn_mask = torch.zeros(
            (batch_size, 1, current_len, total_kv), device=device, dtype=dtype
        )
        if prefix_len > 0:
            invalid_prefix = ~prefix_attention_mask.to(device=device).bool()
            if invalid_prefix.any():
                attn_mask[:, :, :, :prefix_len] = attn_mask[:, :, :, :prefix_len].masked_fill(
                    invalid_prefix[:, None, None, :], torch.finfo(dtype).min
                )
        return attn_mask

    def _build_compact_generation_mask(
        self,
        text_attention_mask: torch.Tensor,
        known_positions: torch.Tensor,
        width: int,
        dtype: torch.dtype,
        preview_len: int = 0,
    ) -> torch.Tensor:
        """Mask for text prefix + optional preview prefix + known large tokens."""
        device = text_attention_mask.device
        bsz, text_len = text_attention_mask.shape
        preview_len = int(preview_len)
        n_img = int(known_positions.numel())
        prefix_len = text_len + preview_len
        total = prefix_len + n_img
        attn = torch.full(
            (bsz, 1, total, total),
            torch.finfo(dtype).min,
            device=device,
            dtype=dtype,
        )
        text_causal = torch.tril(torch.ones(text_len, text_len, device=device, dtype=torch.bool))
        text_eye = torch.eye(text_len, device=device, dtype=torch.bool)
        for b in range(bsz):
            valid = text_attention_mask[b].to(device=device).bool()
            allow = text_causal & valid.unsqueeze(0) & valid.unsqueeze(1)
            allow |= text_eye
            attn[b, 0, :text_len, :text_len].masked_fill_(allow, 0.0)
            if preview_len > 0:
                preview = slice(text_len, prefix_len)
                preview_causal = torch.tril(
                    torch.ones(preview_len, preview_len, device=device, dtype=torch.bool)
                )
                attn[b, 0, preview, :text_len].masked_fill_(valid.view(1, -1), 0.0)
                attn[b, 0, preview, preview].masked_fill_(preview_causal, 0.0)
            if n_img > 0:
                attn[b, 0, prefix_len:, :text_len].masked_fill_(valid.view(1, -1), 0.0)
                if preview_len > 0:
                    attn[b, 0, prefix_len:, text_len:prefix_len] = 0.0

        if n_img > 0:
            rows = known_positions // width
            cols = known_positions % width
            step = rows + cols
            allow_img = step[None, :] <= step[:, None]
            attn[:, :, prefix_len:, prefix_len:].masked_fill_(
                allow_img.view(1, 1, n_img, n_img), 0.0
            )
        return attn

    def _preview_position_ids(
        self,
        text_prefix_len,
        preview_height: int,
        preview_width: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        rows = torch.arange(preview_height, device=device).unsqueeze(1).expand(
            preview_height, preview_width
        ).reshape(-1)
        cols = torch.arange(preview_width, device=device).unsqueeze(0).expand(
            preview_height, preview_width
        ).reshape(-1)
        return self._image_position_ids(
            text_prefix_len,
            rows,
            cols,
            batch_size,
            device,
            image_offset=0,
        )

    def _compact_position_ids(
        self,
        prefix_len: int,
        known_positions: torch.Tensor,
        width: int,
        batch_size: int,
        device: torch.device,
        image_offset: int = 0,
        image_prefix_len=None,
    ) -> torch.Tensor:
        prefix_pos = torch.arange(prefix_len, device=device).view(1, -1).expand(batch_size, -1)
        prefix_pos = prefix_pos.unsqueeze(0).expand(3, -1, -1)
        if known_positions.numel() == 0:
            return prefix_pos
        rows = known_positions // width
        cols = known_positions % width
        image_pos = self._image_position_ids(
            prefix_len if image_prefix_len is None else image_prefix_len,
            rows,
            cols,
            batch_size,
            device,
            image_offset=image_offset,
        )
        return torch.cat([prefix_pos, image_pos], dim=2)

    def _compact_position_ids_with_preview(
        self,
        text_prefix_len: int,
        known_positions: torch.Tensor,
        width: int,
        batch_size: int,
        device: torch.device,
        image_offset: int,
        text_prefix_base,
        preview_height: int = 0,
        preview_width: int = 0,
    ) -> torch.Tensor:
        text_pos = torch.arange(text_prefix_len, device=device).view(1, -1).expand(batch_size, -1)
        text_pos = text_pos.unsqueeze(0).expand(3, -1, -1)
        parts = [text_pos]
        if int(preview_height) > 0 and int(preview_width) > 0:
            parts.append(
                self._preview_position_ids(
                    text_prefix_base,
                    int(preview_height),
                    int(preview_width),
                    batch_size,
                    device,
                )
            )
        if known_positions.numel() > 0:
            rows = known_positions // width
            cols = known_positions % width
            parts.append(
                self._image_position_ids(
                    text_prefix_base,
                    rows,
                    cols,
                    batch_size,
                    device,
                    image_offset=image_offset,
                )
            )
        return torch.cat(parts, dim=2)

    def _split_generation_hidden_states(
        self,
        hidden: torch.Tensor,
        text_prefix_len: int,
        total_prefix_len: int,
        text_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prefix_hidden = hidden[:, :total_prefix_len, :]
        image_hidden = hidden[:, total_prefix_len:, :]
        if total_prefix_len > text_prefix_len:
            return prefix_hidden[:, -1:, :], image_hidden
        return self._split_hidden_states(hidden, text_prefix_len, text_attention_mask)

    @staticmethod
    def _compact_step_mask_2d(known_positions: torch.Tensor, width: int) -> torch.Tensor:
        if known_positions.numel() == 0:
            return torch.empty((0, 0), device=known_positions.device, dtype=torch.bool)
        rows = known_positions // width
        cols = known_positions % width
        step = rows + cols
        return ~(step[None, :] <= step[:, None])

    def _prefill_generation_prefix(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        preview_input_ids: Optional[torch.Tensor] = None,
        preview_height: int = 0,
        preview_width: int = 0,
        image_width: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, DynamicCache, int, torch.Tensor]:
        capture_layer = self._vertical_capture_layer()
        mask_dtype = next(self.backbone.parameters()).dtype
        prefix_cache = DynamicCache(config=getattr(self.backbone, "config", None))
        prefix_base = self._effective_prefix_lens(
            int(text_input_ids.size(1)),
            int(text_input_ids.size(0)),
            text_input_ids.device,
            text_attention_mask=text_attention_mask,
        )
        if preview_input_ids is not None:
            if preview_input_ids.dim() == 3:
                preview_height = int(preview_input_ids.size(1))
                preview_width = int(preview_input_ids.size(2))
                preview_input_ids = preview_input_ids.reshape(preview_input_ids.size(0), -1)
            if preview_input_ids.dim() == 1:
                preview_input_ids = preview_input_ids.unsqueeze(0)
            elif preview_input_ids.dim() != 2:
                raise ValueError(
                    f"preview_input_ids must be 1-D, 2-D or 3-D, got {tuple(preview_input_ids.shape)}."
                )
            preview_input_ids = preview_input_ids.to(device=text_input_ids.device, dtype=torch.long)
            if preview_input_ids.size(0) != text_input_ids.size(0):
                raise ValueError("preview_input_ids batch size mismatch.")
            preview_len = int(preview_input_ids.size(1))
            expected_preview = int(preview_height) * int(preview_width)
            if expected_preview <= 0 or preview_len != expected_preview:
                raise ValueError(
                    f"preview_input_ids length must equal preview_height*preview_width="
                    f"{expected_preview}, got {preview_len}."
                )
            full_prefix_ids = torch.cat([text_input_ids, preview_input_ids], dim=1)
            prefix_mask_4d = self._build_compact_generation_mask(
                text_attention_mask=text_attention_mask,
                known_positions=torch.empty((0,), device=text_input_ids.device, dtype=torch.long),
                width=max(1, int(image_width)),
                dtype=mask_dtype,
                preview_len=preview_len,
            )
            position_ids = self._compact_position_ids_with_preview(
                text_prefix_len=int(text_input_ids.size(1)),
                known_positions=torch.empty((0,), device=text_input_ids.device, dtype=torch.long),
                width=max(1, int(image_width)),
                batch_size=int(text_input_ids.size(0)),
                device=text_input_ids.device,
                image_offset=0,
                text_prefix_base=prefix_base,
                preview_height=int(preview_height),
                preview_width=int(preview_width),
            )
        else:
            full_prefix_ids = text_input_ids
            preview_len = 0
            prefix_mask_4d = self._build_prefix_causal_mask_4d(text_attention_mask, mask_dtype)
            position_ids = None
        outputs = self.backbone(
            input_ids=full_prefix_ids,
            attention_mask=prefix_mask_4d,
            position_ids=position_ids,
            past_key_values=prefix_cache,
            use_cache=True,
            capture_hidden_before_layer=capture_layer,
        )
        vertical_prefix_hidden = (
            outputs.captured_hidden_state if capture_layer is not None else outputs.last_hidden_state
        )
        if vertical_prefix_hidden is None:
            raise RuntimeError(
                "backbone did not return captured_hidden_state for vertical prefix branch."
            )
        if preview_len > 0:
            cond_horizontal_hidden = outputs.last_hidden_state[:, -1:, :]
            cond_vertical_hidden = vertical_prefix_hidden[:, -1:, :]
        else:
            cond_horizontal_hidden = self._gather_last_valid_hidden(
                outputs.last_hidden_state, text_attention_mask
            )
            cond_vertical_hidden = self._gather_last_valid_hidden(
                vertical_prefix_hidden, text_attention_mask
            )
        return (
            cond_horizontal_hidden,
            cond_vertical_hidden,
            outputs.past_key_values,
            int(text_input_ids.size(1)),
            text_attention_mask,
        )

    def _append_backbone_kv_step(
        self,
        step_token_ids: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        prefix_len: int,
        prefix_attention_mask: Optional[torch.Tensor],
        past_key_values: DynamicCache,
        past_image_len: int,
        image_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, DynamicCache]:
        batch_size = int(step_token_ids.size(0))
        current_len = int(step_token_ids.size(1))
        mask_dtype = next(self.backbone.parameters()).dtype
        attention_mask = self._build_kv_attention_mask(
            batch_size=batch_size,
            current_len=current_len,
            past_len=past_image_len,
            device=step_token_ids.device,
            dtype=mask_dtype,
            prefix_attention_mask=prefix_attention_mask,
        )
        position_ids = self._image_position_ids(
            prefix_len,
            rows,
            cols,
            batch_size,
            step_token_ids.device,
            image_offset=image_offset,
        )
        capture_layer = self._vertical_capture_layer()
        outputs = self.backbone(
            input_ids=step_token_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            capture_hidden_before_layer=capture_layer,
        )
        current_h_hidden = outputs.last_hidden_state
        current_v_input_hidden = (
            outputs.captured_hidden_state if capture_layer is not None else current_h_hidden
        )
        if current_v_input_hidden is None:
            raise RuntimeError(
                "backbone did not return captured_hidden_state for vertical KV branch."
            )
        return current_h_hidden, current_v_input_hidden, outputs.past_key_values

    def _append_vertical_kv_step(
        self,
        step_hidden: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        prefix_len: int,
        past_key_values: Optional[DynamicCache],
        past_image_len: int,
        image_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[DynamicCache]]:
        if self.vertical_block is None:
            return step_hidden, past_key_values

        cache = (
            past_key_values
            if past_key_values is not None
            else DynamicCache(config=getattr(self.backbone, "config", None))
        )
        batch_size = int(step_hidden.size(0))
        current_len = int(step_hidden.size(1))
        # Vertical block only ever sees image tokens (no text prefix); all cached
        # image tokens are allowed (proximity is preserved by decode order).
        attention_mask = self._build_kv_attention_mask(
            batch_size=batch_size,
            current_len=current_len,
            past_len=past_image_len,
            device=step_hidden.device,
            dtype=step_hidden.dtype,
            prefix_attention_mask=None,
        )
        position_ids = self._image_position_ids(
            prefix_len,
            rows,
            cols,
            batch_size,
            step_hidden.device,
            image_offset=image_offset,
        )
        pos_emb = self.backbone.rotary_emb(step_hidden, position_ids)
        v_hidden = step_hidden
        for layer in self.vertical_block:
            v_hidden = layer(
                v_hidden,
                position_embeddings=pos_emb,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=cache,
                use_cache=True,
            )
        return self.vertical_norm(v_hidden), cache

    def _compute_step_logits_from_prev(
        self,
        cond_horizontal_hidden: torch.Tensor,
        cond_vertical_hidden: torch.Tensor,
        prev_h_hidden: Optional[torch.Tensor],
        prev_v_hidden: Optional[torch.Tensor],
        step_positions: torch.Tensor,
        prev_positions: torch.Tensor,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = int(cond_horizontal_hidden.size(0))
        cond_h_logits = self.horizontal_head(cond_horizontal_hidden)
        cond_v_logits = self.vertical_head(cond_vertical_hidden)
        step_logits = torch.empty(
            (batch_size, step_positions.numel(), self.vocab_size),
            device=device,
            dtype=cond_h_logits.dtype,
        )

        rows = step_positions // width
        cols = step_positions % width
        left_mask = cols > 0
        up_mask = rows > 0
        both_mask = left_mask & up_mask
        corner_mask = ~left_mask & ~up_mask
        h_only = left_mask & ~up_mask
        v_only = up_mask & ~left_mask

        if prev_h_hidden is not None and prev_v_hidden is not None and prev_positions.numel() > 0:
            h_prev = self.horizontal_head(prev_h_hidden)
            v_prev = self.vertical_head(prev_v_hidden)
            total = int(height * width)
            pos_to_idx = torch.full((total,), -1, device=device, dtype=torch.long)
            pos_to_idx[prev_positions] = torch.arange(prev_positions.numel(), device=device)

            if h_only.any():
                left_idx = pos_to_idx[step_positions[h_only] - 1]
                step_logits[:, h_only, :] = h_prev[:, left_idx, :]

            if v_only.any():
                up_idx = pos_to_idx[step_positions[v_only] - width]
                step_logits[:, v_only, :] = v_prev[:, up_idx, :]

            if both_mask.any():
                left_idx = pos_to_idx[step_positions[both_mask] - 1]
                up_idx = pos_to_idx[step_positions[both_mask] - width]
                rw = self._hv_gate_from_pair(
                    prev_h_hidden[:, left_idx, :],
                    prev_v_hidden[:, up_idx, :],
                    out_dtype=h_prev.dtype,
                )
                step_logits[:, both_mask, :] = (
                    rw * h_prev[:, left_idx, :] + (1.0 - rw) * v_prev[:, up_idx, :]
                )
        elif h_only.any() or v_only.any() or both_mask.any():
            raise RuntimeError(
                "Previous diagonal hidden states are missing for non-corner prediction."
            )

        if corner_mask.any():
            rw_corner = self._hv_gate_corner(cond_horizontal_hidden, out_dtype=cond_h_logits.dtype)
            cond_logits = (
                rw_corner * cond_h_logits[:, :1, :]
                + (1.0 - rw_corner) * cond_v_logits[:, :1, :]
            )
            step_logits[:, corner_mask, :] = cond_logits.expand(
                -1, int(corner_mask.sum().item()), -1
            )
        return step_logits

    def _generate_with_kv_cache(
        self,
        *,
        height: int,
        width: int,
        device: torch.device,
        text_input_ids: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor],
        preview_input_ids: Optional[torch.Tensor],
        preview_height: int,
        preview_width: int,
        temperature: float,
        top_k: int,
        top_p: float,
        sample_logits: bool,
    ) -> Tuple[torch.Tensor, int]:
        step_id = _build_step_id(height, width, device)
        max_step = int(step_id.max().item())
        grid = torch.full(
            (1, height, width), self.mask_token_id, device=device, dtype=torch.long
        )

        text_input_ids, text_attention_mask = self._normalize_text_batch(
            text_input_ids, text_attention_mask
        )
        if text_input_ids is None or text_attention_mask is None:
            raise ValueError("KV-cache generation requires text_input_ids.")
        text_input_ids = text_input_ids.to(device=device, dtype=torch.long)
        text_attention_mask = text_attention_mask.to(device=device, dtype=torch.long)
        if text_input_ids.size(0) != 1:
            raise ValueError(
                "GlmImageFlashAR.generate currently supports batch_size=1; "
                "call it once per prompt."
            )

        (
            cond_horizontal_hidden,
            cond_vertical_hidden,
            backbone_cache,
            prefix_len,
            prefix_mask,
        ) = self._prefill_generation_prefix(
            text_input_ids,
            text_attention_mask,
            preview_input_ids=preview_input_ids,
            preview_height=preview_height,
            preview_width=preview_width,
            image_width=width,
        )

        batch_size = int(text_input_ids.size(0))
        image_offset = self._skipped_preview_offset(height, width)
        vertical_cache = (
            DynamicCache(config=getattr(self.backbone, "config", None))
            if self.vertical_block is not None
            else None
        )
        past_image_len = 0
        prev_positions = torch.empty((0,), device=device, dtype=torch.long)
        prev_h_hidden = None
        prev_v_hidden = None
        num_steps = 0

        for step in range(0, max_step + 1):
            step_positions = (step_id == step).nonzero(as_tuple=False).view(-1).to(
                device=device, dtype=torch.long
            )
            if step_positions.numel() == 0:
                continue
            num_steps += 1

            step_logits = self._compute_step_logits_from_prev(
                cond_horizontal_hidden=cond_horizontal_hidden,
                cond_vertical_hidden=cond_vertical_hidden,
                prev_h_hidden=prev_h_hidden,
                prev_v_hidden=prev_v_hidden,
                step_positions=step_positions,
                prev_positions=prev_positions,
                height=height,
                width=width,
                device=device,
            )

            # GLM deviation: restrict sampling to codebook ids [0, codebook_size).
            if self.codebook_size < self.vocab_size:
                step_logits = step_logits.clone()
                step_logits[:, :, self.codebook_size:] = float("-inf")

            step_pred = _sample_logits(
                step_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                sample_logits=sample_logits,
            )
            grid.view(1, -1)[:, step_positions] = step_pred

            rows = step_positions // width
            cols = step_positions % width
            step_token_ids = step_pred.expand(batch_size, -1).contiguous()
            current_h_hidden, current_v_input_hidden, backbone_cache = self._append_backbone_kv_step(
                step_token_ids=step_token_ids,
                rows=rows,
                cols=cols,
                prefix_len=prefix_len,
                prefix_attention_mask=prefix_mask,
                past_key_values=backbone_cache,
                past_image_len=past_image_len,
                image_offset=image_offset,
            )
            current_v_hidden, vertical_cache = self._append_vertical_kv_step(
                step_hidden=current_v_input_hidden,
                rows=rows,
                cols=cols,
                prefix_len=prefix_len,
                past_key_values=vertical_cache,
                past_image_len=past_image_len,
                image_offset=image_offset,
            )

            prev_positions = step_positions
            prev_h_hidden = current_h_hidden
            prev_v_hidden = current_v_hidden
            past_image_len += int(step_positions.numel())

        return grid[0], num_steps

    def _generate_with_recompute(
        self,
        *,
        height: int,
        width: int,
        device: torch.device,
        text_input_ids: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor],
        preview_input_ids: Optional[torch.Tensor],
        preview_height: int,
        preview_width: int,
        temperature: float,
        top_k: int,
        top_p: float,
        sample_logits: bool,
    ) -> Tuple[torch.Tensor, int]:
        """Teacher-forcing-compatible diagonal decode without KV cache.

        This recomputes prefix + all generated image tokens before each diagonal.
        It is slower than KV cache, but keeps the attention/mask path close to
        training forward and is useful as a correctness/quality check.
        """
        step_id = _build_step_id(height, width, device)
        max_step = int(step_id.max().item())
        grid = torch.full((1, height, width), self.mask_token_id, device=device, dtype=torch.long)

        text_input_ids, text_attention_mask = self._normalize_text_batch(
            text_input_ids, text_attention_mask
        )
        if text_input_ids is None or text_attention_mask is None:
            raise ValueError("Recompute generation requires text_input_ids.")
        text_input_ids = text_input_ids.to(device=device, dtype=torch.long)
        text_attention_mask = text_attention_mask.to(device=device, dtype=torch.long)
        if text_input_ids.size(0) != 1:
            raise ValueError(
                "GlmImageFlashAR.generate currently supports batch_size=1; "
                "call it once per prompt."
            )

        batch_size = int(text_input_ids.size(0))
        text_prefix_len = int(text_input_ids.size(1))
        preview_len = 0
        if preview_input_ids is not None:
            if preview_input_ids.dim() == 3:
                preview_height = int(preview_input_ids.size(1))
                preview_width = int(preview_input_ids.size(2))
                preview_input_ids = preview_input_ids.reshape(preview_input_ids.size(0), -1)
            if preview_input_ids.dim() == 1:
                preview_input_ids = preview_input_ids.unsqueeze(0)
            elif preview_input_ids.dim() != 2:
                raise ValueError(
                    f"preview_input_ids must be 1-D, 2-D or 3-D, got {tuple(preview_input_ids.shape)}."
                )
            preview_input_ids = preview_input_ids.to(device=device, dtype=torch.long)
            if preview_input_ids.size(0) != batch_size:
                raise ValueError("preview_input_ids batch size mismatch.")
            preview_len = int(preview_input_ids.size(1))
            expected_preview = int(preview_height) * int(preview_width)
            if expected_preview <= 0 or preview_len != expected_preview:
                raise ValueError(
                    f"preview_input_ids length must equal preview_height*preview_width="
                    f"{expected_preview}, got {preview_len}."
                )
        else:
            preview_height = 0
            preview_width = 0
        prefix_base = self._effective_prefix_lens(
            text_prefix_len,
            batch_size,
            device,
            text_attention_mask=text_attention_mask,
        )
        image_offset = self._skipped_preview_offset(height, width)
        mask_dtype = next(self.backbone.parameters()).dtype
        known_positions = torch.empty((0,), device=device, dtype=torch.long)
        known_ids = torch.empty((batch_size, 0), device=device, dtype=torch.long)
        prev_positions = torch.empty((0,), device=device, dtype=torch.long)
        prev_h_hidden = None
        prev_v_hidden = None
        num_steps = 0

        for step in range(max_step + 1):
            step_positions = (step_id == step).nonzero(as_tuple=False).view(-1).to(device)
            if step_positions.numel() == 0:
                continue
            num_steps += 1

            id_parts = [text_input_ids]
            if preview_input_ids is not None:
                id_parts.append(preview_input_ids)
            id_parts.append(known_ids)
            compact_ids = torch.cat(id_parts, dim=1)
            compact_mask = self._build_compact_generation_mask(
                text_attention_mask=text_attention_mask,
                known_positions=known_positions,
                width=width,
                dtype=mask_dtype,
                preview_len=preview_len,
            )
            compact_pos = self._compact_position_ids_with_preview(
                text_prefix_len=text_prefix_len,
                known_positions=known_positions,
                width=width,
                batch_size=batch_size,
                device=device,
                image_offset=image_offset,
                text_prefix_base=prefix_base,
                preview_height=preview_height,
                preview_width=preview_width,
            )
            compact_hidden, compact_v_input = self._run_backbone(
                compact_ids, compact_mask, compact_pos
            )
            total_prefix_len = text_prefix_len + preview_len
            cond_h, h_img = self._split_generation_hidden_states(
                compact_hidden, text_prefix_len, total_prefix_len, text_attention_mask
            )
            cond_v, v_in_img = self._split_generation_hidden_states(
                compact_v_input, text_prefix_len, total_prefix_len, text_attention_mask
            )
            if known_positions.numel() > 0:
                image_pos = self._image_position_ids(
                    prefix_base,
                    known_positions // width,
                    known_positions % width,
                    batch_size,
                    device,
                    image_offset=image_offset,
                )
                v_img = self._apply_vertical_block(
                    v_in_img,
                    self._compact_step_mask_2d(known_positions, width),
                    image_pos,
                )
                pos_to_idx = torch.full((height * width,), -1, device=device, dtype=torch.long)
                pos_to_idx[known_positions] = torch.arange(known_positions.numel(), device=device)
                prev_idx = pos_to_idx[prev_positions]
                prev_h_hidden = h_img[:, prev_idx, :]
                prev_v_hidden = v_img[:, prev_idx, :]

            step_logits = self._compute_step_logits_from_prev(
                cond_horizontal_hidden=cond_h,
                cond_vertical_hidden=cond_v,
                prev_h_hidden=prev_h_hidden,
                prev_v_hidden=prev_v_hidden,
                step_positions=step_positions,
                prev_positions=prev_positions,
                height=height,
                width=width,
                device=device,
            )
            if self.codebook_size < self.vocab_size:
                step_logits = step_logits.clone()
                step_logits[:, :, self.codebook_size:] = float("-inf")
            step_pred = _sample_logits(
                step_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                sample_logits=sample_logits,
            )
            grid.view(1, -1)[:, step_positions] = step_pred
            known_positions = torch.cat([known_positions, step_positions], dim=0)
            known_ids = torch.cat([known_ids, step_pred.expand(batch_size, -1)], dim=1)
            prev_positions = step_positions

        return grid[0], num_steps

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        height: int,
        width: int,
        device: torch.device,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        preview_input_ids: Optional[torch.Tensor] = None,
        preview_height: int = 0,
        preview_width: int = 0,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        sample_logits: bool = True,
        return_num_steps: bool = False,
        use_kv_cache: bool = False,
    ):
        """Anti-diagonal parallel decoding in H+W-1 diagonal steps.

        Returns an (H, W) int64 token grid (values in codebook range), or a
        (grid, num_steps) tuple when ``return_num_steps`` is True.
        """
        if text_input_ids is None:
            raise ValueError("GlmImageFlashAR.generate requires text_input_ids.")
        if int(height) <= 0 or int(width) <= 0:
            raise ValueError(f"height and width must be positive, got {height}x{width}.")
        _validate_sampling_args(temperature, top_k, top_p)
        if use_kv_cache and os.environ.get("GLM_FLASHAR_ENABLE_EXPERIMENTAL_KV_CACHE") != "1":
            raise RuntimeError(
                "KV-cache diagonal decode is disabled because it fails "
                "teacher-forced equivalence against the training forward path. "
                "Use the default recompute path. Set "
                "GLM_FLASHAR_ENABLE_EXPERIMENTAL_KV_CACHE=1 only for local "
                "timing/debug experiments."
            )
        generate_fn = self._generate_with_kv_cache if use_kv_cache else self._generate_with_recompute
        grid, num_steps = generate_fn(
            height=height,
            width=width,
            device=device,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            preview_input_ids=preview_input_ids,
            preview_height=preview_height,
            preview_width=preview_width,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            sample_logits=sample_logits,
        )
        if return_num_steps:
            return grid, num_steps
        return grid


__all__ = [
    "GlmImageFlashAR",
    "build_t2i_flashar_mask",
    "_sample_logits",
]
