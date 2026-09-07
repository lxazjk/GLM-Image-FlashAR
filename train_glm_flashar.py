# -*- coding: utf-8 -*-
"""FSDP training entry for GLM-Image FlashAR.

Trains the GLM-Image AR generator wrapped by
:class:`glmflashar.model.modeling_glm_flashar.GlmImageFlashAR`.

Pipeline:
  1. Load the GLM-Image AR generator (``GlmImageForConditionalGeneration``,
     subfolder ``vision_language_encoder``) and patch its inner text model so the
     FlashAR wrapper can drive it (``apply_glm_backbone_patch``).
  2. Wrap ``model.model.language_model`` (+ ``lm_head``) with ``GlmImageFlashAR``.
  3. FSDP-wrap over ``GlmImageTextDecoderLayer`` (multi-GPU / torchrun), or run on
     a single device with no FSDP (smoke test) when WORLD_SIZE <= 1.
  4. Two-phase schedule: phase-1 trains only the vertical branch + fusion gates
     (backbone frozen) for ``vertical_head_warmup_steps`` optimizer steps; phase-2
     restores the normal trainable set. Cosine LR, AdamW or Adafactor, gradient accumulation,
     gradient checkpointing.
  5. Periodically save the FlashAR-added params (heads, vertical block/norm,
     gates), full model weights when training the backbone, and resume state
     to ``save_dir``.

Data: pretokenized tar shards (``--pretok_glob``), each sample a (32,32) d32
codebook grid + caption. Text prefix ids are built with the GLM processor.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from contextlib import nullcontext
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from glmflashar.data.pretokenized_glm import (
    PretokGlmShardDataset,
    collate_pretok_glm,
    count_pretok_pairs,
    summarize_pretok_pairs,
)
from glmflashar.constants import FLASHAR_SEMANTICS_VERSION
from glmflashar.utils.preview import stock_preview_shape_from_pixels
try:
    from glmflashar.model.glm_backbone_patch import apply_glm_backbone_patch, set_layers_non_causal
    from glmflashar.model.modeling_glm_flashar import GlmImageFlashAR
except ModuleNotFoundError as exc:
    if "transformers.models.glm_image" in str(exc):
        raise ModuleNotFoundError(
            "GLM-Image transformers module is unavailable in this Python environment. "
            "Install the dependencies in requirements.txt."
        ) from exc
    raise
from glmflashar.utils.text_utils_glm import build_glm_processor, build_glm_text_prefix


# Env-gated diagnostics (off by default): prints backbone/FlashAR grad norms and
# peak GPU memory in the [METRIC] line. Does not affect training behaviour.
_FLASHAR_DEBUG = bool(os.environ.get("FLASHAR_DEBUG"))

# Parameter-name prefixes for the FlashAR-added (non-backbone) modules.
FLASHAR_PARAM_PREFIXES = (
    "horizontal_head.",
    "vertical_head.",
    "vertical_block.",
    "vertical_norm.",
    "hv_gate_mlp.",
    "hv_gate_corner.",
)
# Subset trained during the phase-1 vertical-only warmup (mirrors the reference
# ``set_vertical_branch_trainable``): the vertical branch + fusion gates only.
VERTICAL_ONLY_PREFIXES = (
    "vertical_block.",
    "vertical_norm.",
    "vertical_head.",
    "hv_gate_mlp.",
    "hv_gate_corner.",
)
def _import_adafactor():
    """Import ``transformers.optimization.Adafactor``.

    In this environment an outdated ``peft`` (shadowed from system site-packages)
    breaks ``transformers.trainer_utils`` at import time via
    ``from peft import PeftMixedModel``. ``transformers.optimization`` pulls in
    ``trainer_utils`` transitively, so temporarily force ``is_peft_available()``
    to ``False`` to skip that (unused-for-training) peft import.
    """
    import transformers.utils as _tu

    _orig = _tu.is_peft_available
    _tu.is_peft_available = lambda: False
    try:
        from transformers.optimization import Adafactor
    finally:
        _tu.is_peft_available = _orig
    return Adafactor


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config_json", type=str, default="")
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(parents=[pre])
    # paths / data
    p.add_argument("--model_path", type=str, default="./models/GLM-Image-Decoder")
    p.add_argument("--vle_subfolder", type=str, default="vision_language_encoder")
    p.add_argument("--processor_subfolder", type=str, default="processor")
    p.add_argument("--pretok_glob", type=str, default="")
    p.add_argument(
        "--require_preview_tokens",
        action="store_true",
        help="Require pretokenized shards to provide {stem}.preview.pt for preview-conditioned training.",
    )
    p.add_argument(
        "--pretok_preflight_deep_check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Before loading GLM weights, load pretokenized tensor payloads and validate shapes/ranges.",
    )
    p.add_argument(
        "--require_pretok_metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require {stem}.meta.json sidecars and validate tensor/source hashes before training.",
    )
    p.add_argument("--save_dir", type=str, default="./outputs/glm_flashar")
    # target grid / vocab
    p.add_argument("--target_h", type=int, default=1024)
    p.add_argument("--target_w", type=int, default=1024)
    p.add_argument("--vocab_size", type=int, default=16512)
    p.add_argument("--hidden_size", type=int, default=4096)
    p.add_argument("--codebook_size", type=int, default=16384)
    # vertical branch
    p.add_argument("--use_vertical_block", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vertical_layers", type=int, default=4)
    p.add_argument("--vertical_start_layer", type=int, default=-1)
    p.add_argument("--vertical_head_warmup_steps", type=int, default=1000)
    # optimisation
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adafactor"])
    p.add_argument("--init_ckpt", type=str, default="",
                   help="Weight-init checkpoint. Use flashar_full_step*.pt for "
                        "backbone-tuned artifacts and flashar_step*.pt only for "
                        "frozen-backbone artifacts. This is not optimizer/scheduler/"
                        "data resume.")
    p.add_argument("--allow_heads_only_init", action="store_true",
                   help="Allow --init_ckpt from flashar_heads_only_step*.pt produced by "
                        "a backbone-tuned run. Dangerous: the trained backbone is absent.")
    p.add_argument(
        "--allow_text_only_init",
        action="store_true",
        help="Allow --init_ckpt from legacy text-only checkpoints while training with preview tokens.",
    )
    p.add_argument("--resume_ckpt", type=str, default="",
                   help="flashar_resume_step*.pt training checkpoint saved by this script. "
                        "Restores model, optimizer, scheduler, RNG, global_step, epoch "
                        "and batch cursor. Mutually exclusive with --init_ckpt.")
    p.add_argument("--step_offset", type=int, default=-1,
                   help="Display/checkpoint step offset for weight-init finetunes. "
                        "Negative means auto from --init_ckpt/--resume_ckpt step when available. "
                        "Does not change optimizer schedule or data iteration.")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--backbone_lr", type=float, default=0.0)
    p.add_argument("--backbone_lr_factor", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.95)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--shuffle_buffer_size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    # lr schedule (cosine, two-phase flat + decay like the reference)
    p.add_argument("--lr_scheduler", type=str, default="cosine", choices=["none", "cosine"])
    p.add_argument("--lr_min_factor", type=float, default=0.05)
    p.add_argument("--phase1_flat_steps", type=int, default=200)
    p.add_argument("--phase2_flat_steps", type=int, default=500)
    p.add_argument("--phase2_lr_factor", type=float, default=0.1)
    # loss weighting
    p.add_argument("--aux_loss_h_weight", type=float, default=0.0)
    p.add_argument("--aux_loss_v_weight", type=float, default=0.05)
    p.add_argument("--gate_collapse_weight", type=float, default=0.0)
    p.add_argument("--chunked_loss", action=argparse.BooleanOptionalAction, default=False)
    # backbone / memory
    p.add_argument("--train_backbone", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    # FSDP
    p.add_argument("--fsdp_use_orig_params", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fsdp_cpu_offload", action="store_true")
    p.add_argument("--fsdp_no_sync", action=argparse.BooleanOptionalAction, default=False)
    # logging / saving
    p.add_argument("--log_every_steps", type=int, default=1)
    p.add_argument("--save_every_steps", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")

    if pre_args.config_json:
        with open(pre_args.config_json, "r", encoding="utf-8") as f:
            defaults = json.load(f)
        valid = {a.dest for a in p._actions}
        unknown = sorted(set(defaults) - valid)
        if unknown:
            raise ValueError(f"Unknown keys in config_json: {unknown}")
        p.set_defaults(**defaults)
    args = p.parse_args()
    if args.init_ckpt and args.resume_ckpt:
        raise ValueError("--init_ckpt and --resume_ckpt are mutually exclusive.")
    return args


def validate_training_args(args) -> None:
    def finite_float(name: str) -> float:
        value = float(getattr(args, name))
        if not math.isfinite(value):
            raise ValueError(f"--{name} must be finite, got {value}.")
        return value

    def require_file(path: str, desc: str) -> None:
        if not os.path.isfile(path):
            raise ValueError(f"missing {desc}: {path}")

    model_path = os.path.expanduser(str(args.model_path))
    vle_dir = os.path.join(model_path, str(args.vle_subfolder))
    processor_dir = os.path.join(model_path, str(args.processor_subfolder))
    require_file(os.path.join(vle_dir, "config.json"), "GLM VLE config.json")
    require_file(os.path.join(vle_dir, "generation_config.json"), "GLM VLE generation_config.json")
    for name in ("tokenizer.json", "preprocessor_config.json", "chat_template.jinja"):
        require_file(os.path.join(processor_dir, name), f"GLM processor {name}")

    if int(args.target_h) <= 0 or int(args.target_w) <= 0:
        raise ValueError("--target_h/--target_w must be positive.")
    if int(args.target_h) % 32 != 0 or int(args.target_w) % 32 != 0:
        raise ValueError(
            f"--target_h/--target_w must be multiples of 32, got {args.target_h}x{args.target_w}."
        )
    stock_preview_shape_from_pixels(args.target_h, args.target_w)
    if int(args.vocab_size) <= 0:
        raise ValueError("--vocab_size must be positive.")
    if int(args.codebook_size) <= 0:
        raise ValueError("--codebook_size must be positive.")
    if int(args.codebook_size) > int(args.vocab_size):
        raise ValueError(
            f"--codebook_size must be <= --vocab_size, got {args.codebook_size} > {args.vocab_size}."
        )
    if int(args.hidden_size) <= 0:
        raise ValueError("--hidden_size must be positive.")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch_size must be positive.")
    if int(args.grad_accum_steps) <= 0:
        raise ValueError("--grad_accum_steps must be positive.")
    if int(args.num_workers) < 0:
        raise ValueError("--num_workers must be non-negative.")
    if int(args.shuffle_buffer_size) < 0:
        raise ValueError("--shuffle_buffer_size must be non-negative.")
    if int(args.epochs) <= 0:
        raise ValueError("--epochs must be positive.")
    if int(args.max_steps) < 0:
        raise ValueError("--max_steps must be non-negative.")
    if int(args.save_every_steps) < 0:
        raise ValueError("--save_every_steps must be non-negative.")
    if int(args.log_every_steps) < 0:
        raise ValueError("--log_every_steps must be non-negative.")
    lr = finite_float("lr")
    backbone_lr = finite_float("backbone_lr")
    backbone_lr_factor = finite_float("backbone_lr_factor")
    weight_decay = finite_float("weight_decay")
    adam_beta1 = finite_float("adam_beta1")
    adam_beta2 = finite_float("adam_beta2")
    grad_clip = finite_float("grad_clip")
    lr_min_factor = finite_float("lr_min_factor")
    phase2_lr_factor = finite_float("phase2_lr_factor")
    if lr <= 0:
        raise ValueError("--lr must be positive.")
    if backbone_lr < 0:
        raise ValueError("--backbone_lr must be non-negative.")
    if backbone_lr_factor < 0:
        raise ValueError("--backbone_lr_factor must be non-negative.")
    if args.train_backbone and backbone_lr <= 0 and lr * backbone_lr_factor <= 0:
        raise ValueError(
            "--train_backbone requires an effective backbone LR > 0; set "
            "--backbone_lr or a positive --backbone_lr_factor."
        )
    if weight_decay < 0:
        raise ValueError("--weight_decay must be non-negative.")
    if not (0.0 <= adam_beta1 < 1.0):
        raise ValueError("--adam_beta1 must be in [0, 1).")
    if not (0.0 <= adam_beta2 < 1.0):
        raise ValueError("--adam_beta2 must be in [0, 1).")
    if grad_clip < 0:
        raise ValueError("--grad_clip must be non-negative.")
    if args.use_vertical_block and int(args.vertical_layers) <= 0:
        raise ValueError(
            "--vertical_layers must be positive when --use_vertical_block is enabled; "
            "use --no-use_vertical_block to disable it."
        )
    if not args.use_vertical_block and int(args.vertical_layers) < 0:
        raise ValueError("--vertical_layers must be non-negative.")
    if int(args.vertical_head_warmup_steps) < 0:
        raise ValueError("--vertical_head_warmup_steps must be non-negative.")
    if int(args.phase1_flat_steps) < 0:
        raise ValueError("--phase1_flat_steps must be non-negative.")
    if int(args.phase2_flat_steps) < 0:
        raise ValueError("--phase2_flat_steps must be non-negative.")
    if not (0.0 <= lr_min_factor <= 1.0):
        raise ValueError("--lr_min_factor must be in [0, 1].")
    if not (0.0 <= phase2_lr_factor <= 1.0):
        raise ValueError("--phase2_lr_factor must be in [0, 1].")
    for name in ("aux_loss_h_weight", "aux_loss_v_weight", "gate_collapse_weight"):
        if finite_float(name) < 0:
            raise ValueError(f"--{name} must be non-negative.")


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed(args: argparse.Namespace):
    """Return (rank, world_size, local_rank, device, use_fsdp)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if "RANK" in os.environ and world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return rank, world_size, local_rank, device, True
    # single process / single GPU: no FSDP (smoke-test path)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    return 0, 1, 0, device, False


def validate_distributed_training_args(args: argparse.Namespace, use_fsdp: bool) -> None:
    if not use_fsdp:
        return
    if not bool(args.fsdp_use_orig_params):
        raise ValueError(
            "FSDP training requires --fsdp_use_orig_params. The training code "
            "depends on original parameter names for backbone grouping/freezing."
        )
    if bool(args.fsdp_cpu_offload) and int(args.grad_accum_steps) > 1:
        raise ValueError(
            "--fsdp_cpu_offload with --grad_accum_steps > 1 is not a supported "
            "baseline training configuration."
        )


def next_batch_all_ranks(data_iter, device: torch.device, use_distributed: bool):
    """Fetch one batch and make epoch exhaustion a collective decision.

    Iterable tar shards can split unevenly across ranks/workers. If any rank runs
    out first, all ranks stop the epoch together before entering FSDP collectives.
    Ranks that already fetched an extra local batch discard it.
    """
    try:
        batch = next(data_iter)
        has_batch = 1
    except StopIteration:
        batch = None
        has_batch = 0

    if not use_distributed:
        return batch, bool(has_batch), bool(has_batch)

    flag = torch.tensor([has_batch], device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    all_have_batch = bool(int(flag.item()))
    return batch if all_have_batch else None, all_have_batch, bool(has_batch)


def skip_resume_batches(
    data_iter,
    start_batch_idx: int,
    device: torch.device,
    use_distributed: bool,
) -> int:
    skipped = 0
    while skipped < start_batch_idx:
        _, all_have_batch, _ = next_batch_all_ranks(
            data_iter,
            device,
            use_distributed=use_distributed,
        )
        if not all_have_batch:
            break
        skipped += 1
    if skipped != start_batch_idx:
        raise RuntimeError(
            f"resume cursor mismatch: skipped {skipped}/{start_batch_idx} batches; "
            "data shard order/content no longer matches the resume checkpoint."
        )
    return skipped


def reached_max_steps(args, global_step: int) -> bool:
    return int(getattr(args, "max_steps", 0)) > 0 and int(global_step) >= int(args.max_steps)


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_model(args, torch_dtype, device, is_main):
    from transformers.models.glm_image import GlmImageForConditionalGeneration

    apply_glm_backbone_patch()
    if is_main:
        print(f"[INFO] loading GLM-Image AR generator from {args.model_path} "
              f"(subfolder={args.vle_subfolder})", flush=True)
    model = GlmImageForConditionalGeneration.from_pretrained(
        args.model_path,
        subfolder=args.vle_subfolder,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    language_model = model.model.language_model
    lm_head = model.lm_head

    wrapper = GlmImageFlashAR(
        language_model,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        codebook_size=args.codebook_size,
        use_vertical_block=args.use_vertical_block,
        vertical_layers=args.vertical_layers,
        vertical_start_layer=args.vertical_start_layer,
        lm_head=lm_head,
    )
    set_layers_non_causal(wrapper.backbone, True)
    if wrapper.vertical_block is not None:
        set_layers_non_causal(wrapper.vertical_block, True)
    wrapper = wrapper.to(dtype=torch_dtype)
    if args.gradient_checkpointing:
        set_flashar_grad_checkpointing(wrapper, enabled=True)
    # free the outer generation head container; we keep language_model + lm_head refs
    del model
    return wrapper


def wrap_fsdp(args, wrapper, device):
    from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.glm_image.modeling_glm_image import (
        GlmImageTextDecoderLayer,
    )

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={GlmImageTextDecoderLayer},
    )
    cpu_offload = CPUOffload(offload_params=True) if args.fsdp_cpu_offload else None
    return FSDP(
        wrapper,
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=cpu_offload,
        device_id=device,
        use_orig_params=args.fsdp_use_orig_params,
    )


# ---------------------------------------------------------------------------
# Trainable-parameter management (two-phase schedule)
# ---------------------------------------------------------------------------

def _is_backbone_param(name: str) -> bool:
    clean = normalize_param_name(name)
    return clean.startswith("backbone.") or ".backbone." in clean


def normalize_param_name(name: str) -> str:
    clean = str(name).replace("_fsdp_wrapped_module.", "")
    while clean.startswith("module."):
        clean = clean[len("module."):]
    return clean


def normalize_state_dict_keys(state: dict, *, strip_backbone_prefix: bool = False) -> dict:
    normalized = {}
    for raw_key, value in state.items():
        key = normalize_param_name(raw_key)
        if strip_backbone_prefix and key.startswith("backbone."):
            key = key[len("backbone."):]
        if key in normalized:
            raise ValueError(f"checkpoint has duplicate tensor key after normalization: {key}")
        normalized[key] = value
    return normalized


def infer_vertical_layers_from_state(state_dict: dict) -> int:
    max_idx = -1
    for raw_key in state_dict:
        key = normalize_param_name(raw_key)
        if not key.startswith("vertical_block."):
            continue
        try:
            idx = int(key.split(".")[1])
        except (IndexError, ValueError):
            continue
        max_idx = max(max_idx, idx)
    return max_idx + 1


def validate_checkpoint_vertical_metadata(
    meta_args: dict,
    state_dict: dict,
    ckpt_path: str,
    *,
    expected_vertical_start_layer=None,
    expected_vertical_start_layer_arg=None,
    expected_default_vertical_start_layer=None,
) -> None:
    inferred_vlayers = infer_vertical_layers_from_state(state_dict)
    meta_use_vb = meta_args.get("use_vertical_block")
    if meta_use_vb is False and inferred_vlayers > 0:
        raise ValueError(
            f"{ckpt_path} metadata says use_vertical_block=False but vertical_block tensors are present."
        )
    if meta_use_vb is True and inferred_vlayers <= 0:
        raise ValueError(
            f"{ckpt_path} metadata says use_vertical_block=True but no vertical_block tensors are present."
        )
    meta_vlayers = meta_args.get("vertical_layers")
    if inferred_vlayers > 0 and meta_vlayers is not None and int(meta_vlayers) != inferred_vlayers:
        raise ValueError(
            f"{ckpt_path} vertical_layers metadata ({meta_vlayers}) does not match "
            f"vertical_block tensors ({inferred_vlayers})."
        )
    meta_vstart = meta_args.get("vertical_start_layer")
    if inferred_vlayers > 0 and meta_vstart is not None and expected_vertical_start_layer is not None:
        meta_vstart = int(meta_vstart)
        expected_values = {int(expected_vertical_start_layer)}
        if expected_vertical_start_layer_arg is not None:
            expected_values.add(int(expected_vertical_start_layer_arg))
        if (
            expected_default_vertical_start_layer is not None
            and int(expected_vertical_start_layer) == int(expected_default_vertical_start_layer)
        ):
            expected_values.add(-1)
        if meta_vstart not in expected_values:
            expected_desc = "/".join(str(v) for v in sorted(expected_values))
            raise ValueError(
                f"{ckpt_path} vertical_start_layer metadata ({meta_vstart}) does not match "
                f"current vertical_start_layer ({expected_desc})."
            )


def checkpoint_semantics_version(obj) -> str:
    if isinstance(obj, dict):
        return str(obj.get("flashar_semantics_version", ""))
    return ""


def validate_preview_checkpoint_semantics(
    args,
    obj,
    ckpt_path: str,
    *,
    context: str,
    exc_type=ValueError,
) -> None:
    if not getattr(args, "require_preview_tokens", False):
        return
    preview_conditioned = isinstance(obj, dict) and obj.get("preview_conditioned") is True
    if not preview_conditioned:
        raise exc_type(
            f"{context}={ckpt_path} is not marked preview-conditioned, "
            "but current training requires preview tokens. Use a checkpoint saved "
            "by the current preview-conditioned training script."
        )
    semver = checkpoint_semantics_version(obj)
    if semver != FLASHAR_SEMANTICS_VERSION:
        raise exc_type(
            f"{context}={ckpt_path} has flashar_semantics_version={semver!r}, "
            f"but current preview-conditioned training requires "
            f"{FLASHAR_SEMANTICS_VERSION!r}. Start clean or initialize/resume from "
            "a checkpoint saved by the current ropefix training script."
        )


def checkpoint_args_require_preview(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    meta_args = obj.get("args")
    return isinstance(meta_args, dict) and bool(meta_args.get("require_preview_tokens"))


def validate_state_dict_compatible(module, state: dict, context: str, allow_missing) -> None:
    target = module.state_dict()
    unexpected = [k for k in state if k not in target]
    missing = [k for k in target if k not in state and not allow_missing(k)]
    shape_mismatch = []
    for k, v in state.items():
        if k not in target:
            continue
        if hasattr(v, "shape") and tuple(v.shape) != tuple(target[k].shape):
            shape_mismatch.append((k, tuple(v.shape), tuple(target[k].shape)))
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys when loading {context}: {unexpected[:8]}"
            f"{' ...' if len(unexpected) > 8 else ''}"
        )
    if missing:
        raise RuntimeError(
            f"Missing required keys when loading {context}: {missing[:8]}"
            f"{' ...' if len(missing) > 8 else ''}"
        )
    if shape_mismatch:
        detail = "; ".join(
            f"{k}: checkpoint={got} model={want}"
            for k, got, want in shape_mismatch[:4]
        )
        raise RuntimeError(f"size mismatch when loading {context}: {detail}")


def snapshot_requires_grad(module) -> Dict[str, bool]:
    return {n: p.requires_grad for n, p in module.named_parameters()}


def restore_requires_grad(module, state: Dict[str, bool]) -> None:
    for n, p in module.named_parameters():
        if n in state:
            p.requires_grad = state[n]


def make_lr_lambda(args, total_decay_steps: int):
    total_decay_steps = max(1, int(total_decay_steps))

    def lr_lambda(step: int) -> float:
        if step <= 0:
            return 1.0
        progress = min(1.0, float(step) / float(total_decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(args.lr_min_factor) + (1.0 - float(args.lr_min_factor)) * cosine

    return lr_lambda


def start_lambda_scheduler(args, optimizer, total_decay_steps: int):
    for pg in optimizer.param_groups:
        pg["initial_lr"] = pg["lr"]
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=make_lr_lambda(args, total_decay_steps)
    )


def maybe_create_scheduler(args, optimizer, remaining_steps: int):
    if args.lr_scheduler != "cosine" or args.max_steps <= 0:
        return None
    return start_lambda_scheduler(args, optimizer, max(1, remaining_steps))


def scheduler_should_be_active(args, global_step: int) -> bool:
    if args.lr_scheduler != "cosine" or args.max_steps <= 0:
        return False
    warmup_steps = max(0, int(args.vertical_head_warmup_steps))
    if warmup_steps >= args.max_steps:
        warmup_steps = max(0, args.max_steps - 1)
    global_step = int(global_step)
    if warmup_steps > 0 and global_step < warmup_steps:
        return global_step >= max(0, int(args.phase1_flat_steps))
    phase2_start = warmup_steps if warmup_steps > 0 else 0
    phase2_cosine_start = phase2_start + max(0, int(args.phase2_flat_steps))
    return phase2_cosine_start <= global_step < int(args.max_steps)


def resume_data_state(args) -> dict:
    keys = (
        "pretok_glob",
        "target_h",
        "target_w",
        "codebook_size",
        "seed",
        "shuffle",
        "shuffle_buffer_size",
        "batch_size",
        "num_workers",
        "grad_accum_steps",
    )
    state = {k: getattr(args, k, None) for k in keys}
    resolved = [
        os.path.abspath(str(path))
        for path in list(getattr(args, "_resolved_pretok_shards", []))
    ]
    state["resolved_pretok_shards"] = resolved
    state["resolved_pretok_shard_fingerprints"] = shard_fingerprints(resolved)
    return state


def shard_fingerprints(paths: List[str]) -> List[dict]:
    fingerprints = []
    for raw_path in paths:
        path = os.path.abspath(str(raw_path))
        try:
            st = os.stat(path)
            fingerprints.append(
                {
                    "path": path,
                    "exists": True,
                    "size": int(st.st_size),
                    "mtime_ns": int(st.st_mtime_ns),
                }
            )
        except FileNotFoundError:
            fingerprints.append(
                {
                    "path": path,
                    "exists": False,
                    "size": None,
                    "mtime_ns": None,
                }
            )
    return fingerprints


def validate_resume_data_state(args, training_state: dict) -> None:
    saved = training_state.get("data_state")
    if saved is None:
        raise RuntimeError(
            f"--resume_ckpt={args.resume_ckpt} does not contain data_state; "
            "use --init_ckpt for weight-only initialization or resume from a "
            "flashar_resume_step*.pt saved by the current training script."
        )
    current = resume_data_state(args)
    mismatched = {
        k: (saved.get(k), current.get(k))
        for k in sorted(current)
        if saved.get(k) != current.get(k)
    }
    if mismatched:
        detail = ", ".join(
            f"{k}: saved={old!r} current={new!r}" for k, (old, new) in mismatched.items()
        )
        raise RuntimeError(
            "resume data-state mismatch; refusing to replay a different data stream: "
            f"{detail}"
        )


def resume_training_config_state(args) -> dict:
    keys = (
        "optimizer",
        "lr",
        "backbone_lr",
        "backbone_lr_factor",
        "model_path",
        "vle_subfolder",
        "processor_subfolder",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "grad_clip",
        "max_steps",
        "lr_scheduler",
        "lr_min_factor",
        "phase1_flat_steps",
        "phase2_flat_steps",
        "phase2_lr_factor",
        "vertical_head_warmup_steps",
        "train_backbone",
        "gradient_checkpointing",
        "dtype",
        "use_vertical_block",
        "vertical_layers",
        "vertical_start_layer",
        "vocab_size",
        "hidden_size",
        "codebook_size",
        "aux_loss_h_weight",
        "aux_loss_v_weight",
        "gate_collapse_weight",
        "chunked_loss",
        "require_preview_tokens",
        "require_pretok_metadata",
        "pretok_preflight_deep_check",
    )
    return {k: getattr(args, k, None) for k in keys}


def validate_resume_training_config(args, training_state: dict) -> None:
    saved = training_state.get("training_config")
    if saved is None:
        raise RuntimeError(
            f"--resume_ckpt={args.resume_ckpt} does not contain training_config; "
            "use --init_ckpt for weight-only initialization or resume from a "
            "flashar_resume_step*.pt saved by the current training script."
        )
    current = resume_training_config_state(args)
    mismatched = {
        k: (saved.get(k), current.get(k))
        for k in sorted(current)
        if saved.get(k) != current.get(k)
    }
    if mismatched:
        detail = ", ".join(
            f"{k}: saved={old!r} current={new!r}" for k, (old, new) in mismatched.items()
        )
        raise RuntimeError(
            "resume training-config mismatch; refusing to continue with changed "
            f"optimization/model settings: {detail}"
        )


def preflight_resume_checkpoint(args, is_main: bool) -> None:
    if not args.resume_ckpt:
        return
    obj = torch.load(args.resume_ckpt, map_location="meta", weights_only=False)
    training_state = obj.get("training_state") if isinstance(obj, dict) else None
    if not isinstance(training_state, dict):
        raise RuntimeError(
            f"--resume_ckpt={args.resume_ckpt} does not contain training_state; "
            "use --init_ckpt for weight-only initialization."
        )
    if not (isinstance(obj, dict) and (obj.get("state_dict") or obj.get("flashar"))):
        raise RuntimeError(
            f"--resume_ckpt={args.resume_ckpt} does not contain FlashAR model weights; "
            "resume checkpoints must include state_dict or flashar tensors."
        )
    validate_preview_checkpoint_semantics(
        args,
        obj,
        args.resume_ckpt,
        context="--resume_ckpt",
        exc_type=RuntimeError,
    )
    saved_training_config = training_state.get("training_config") or {}
    if "backbone" in obj and not obj.get("backbone"):
        raise RuntimeError(f"--resume_ckpt={args.resume_ckpt} has an empty backbone payload.")
    if obj.get("backbone") and not (obj.get("flashar") or obj.get("state_dict")):
        raise RuntimeError(f"--resume_ckpt={args.resume_ckpt} has backbone tensors but no FlashAR tensors.")
    if bool(saved_training_config.get("train_backbone")) and not obj.get("backbone"):
        raise RuntimeError(
            f"--resume_ckpt={args.resume_ckpt} was saved from train_backbone=True "
            "but has no trained backbone payload."
        )
    if isinstance(obj, dict) and "flashar" in obj and "backbone" not in obj:
        raise RuntimeError(
            f"--resume_ckpt={args.resume_ckpt} has a top-level 'flashar' payload but no backbone payload; "
            "resume checkpoints for frozen-backbone runs must use top-level state_dict."
        )
    if training_state.get("optimizer") is None:
        raise RuntimeError(
            f"--resume_ckpt={args.resume_ckpt} does not contain optimizer state; "
            "use --init_ckpt for weight-only initialization."
        )
    validate_resume_data_state(args, training_state)
    validate_resume_training_config(args, training_state)
    global_step = int(training_state.get("global_step", training_state.get("local_step", 0)))
    scheduler_meta = training_state.get("scheduler_meta")
    scheduler_state = training_state.get("scheduler")
    if scheduler_state is not None:
        if not isinstance(scheduler_meta, dict) or "total_decay_steps" not in scheduler_meta:
            raise RuntimeError("resume checkpoint has scheduler state but no scheduler_meta.total_decay_steps.")
    elif scheduler_should_be_active(args, global_step):
        raise RuntimeError(
            f"--resume_ckpt={args.resume_ckpt} is at global_step={global_step}, "
            "where cosine scheduler should be active, but it has no scheduler state. "
            "Use a flashar_resume_step*.pt saved by the current training script or "
            "restart with --init_ckpt for weight-only initialization."
        )
    if args.step_offset < 0:
        args.step_offset = int(
            training_state.get(
                "step_offset",
                int(training_state.get("step", global_step)) - global_step,
            )
        )
    if is_main:
        print(
            f"[INFO] resume preflight passed for {args.resume_ckpt}: "
            f"global_step={global_step} display_step={args.step_offset + global_step}",
            flush=True,
        )


def freeze_backbone_if_needed(args, target) -> None:
    if args.train_backbone:
        return
    for n, p in target.named_parameters():
        if _is_backbone_param(n):
            p.requires_grad = False


def set_vertical_only_trainable(target) -> None:
    for n, p in target.named_parameters():
        clean = normalize_param_name(n)
        p.requires_grad = any(clean.startswith(pfx) for pfx in VERTICAL_ONLY_PREFIXES)


def set_backbone_grad_checkpointing(target, enabled: bool) -> None:
    backbone = getattr(target, "backbone", None)
    if backbone is None:
        return
    if enabled and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
        if getattr(backbone, "config", None) is not None:
            backbone.config.use_cache = False
    elif not enabled and hasattr(backbone, "gradient_checkpointing_disable"):
        backbone.gradient_checkpointing_disable()


def set_vertical_block_grad_checkpointing(target, enabled: bool) -> None:
    block = getattr(target, "vertical_block", None)
    if block is None:
        return
    if enabled and hasattr(block, "gradient_checkpointing_enable"):
        block.gradient_checkpointing_enable()
    elif not enabled and hasattr(block, "gradient_checkpointing_disable"):
        block.gradient_checkpointing_disable()
    for module in block.modules() if hasattr(block, "modules") else []:
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = bool(enabled)


def set_flashar_grad_checkpointing(target, enabled: bool) -> None:
    # GlmImageTextDecoderLayer subclasses GradientCheckpointingLayer. The
    # backbone has model-level helpers; the deep-copied vertical_block is a plain
    # ModuleList, so set per-layer flags there explicitly.
    set_backbone_grad_checkpointing(target, enabled=enabled)
    set_vertical_block_grad_checkpointing(target, enabled=enabled)


def build_optimizer(args, wrapper) -> torch.optim.Optimizer:
    backbone_params: List[torch.nn.Parameter] = []
    other_params: List[torch.nn.Parameter] = []
    for name, p in wrapper.named_parameters():
        if not (p.requires_grad and p.numel() > 0):
            continue
        if args.train_backbone and _is_backbone_param(name):
            backbone_params.append(p)
        else:
            other_params.append(p)
    if args.train_backbone and not backbone_params:
        raise ValueError(
            "--train_backbone was set but no trainable backbone parameters were found; "
            "refusing to silently train only FlashAR heads."
        )
    if not backbone_params and not other_params:
        raise ValueError("No trainable parameters found; check freeze settings.")

    # Build the parameter groups: FlashAR group at args.lr, backbone group at
    # backbone_lr (shared by both optimizer backends).
    if args.train_backbone and backbone_params:
        backbone_lr = float(args.backbone_lr)
        if backbone_lr <= 0:
            backbone_lr = float(args.lr) * float(args.backbone_lr_factor)
        opt_input = []
        if other_params:
            opt_input.append({"params": other_params, "lr": float(args.lr), "name": "flashar"})
        opt_input.append({"params": backbone_params, "lr": backbone_lr, "name": "backbone"})
    else:
        opt_input = other_params if other_params else backbone_params

    if args.optimizer == "adafactor":
        # Factored second-moment (no per-param Adam state) so full-backbone
        # training fits on a single GPU. scale_parameter/relative_step/warmup_init
        # are disabled so the explicit per-group LRs and external scheduler apply.
        Adafactor = _import_adafactor()
        return Adafactor(
            opt_input,
            lr=float(args.lr),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=args.weight_decay,
        )

    betas = (args.adam_beta1, args.adam_beta2)
    return torch.optim.AdamW(
        opt_input, lr=float(args.lr), betas=betas, weight_decay=args.weight_decay
    )


def load_init_checkpoint(args, wrapper, is_main) -> None:
    """Load trained weights from ``args.init_ckpt`` / ``args.resume_ckpt`` into a freshly built wrapper
    (before FSDP wrap / optimizer construction).

    Two checkpoint formats are supported:

    * FULL checkpoint (has a ``"backbone"`` key): produced when training with
      ``--train_backbone``. Loads BOTH the trained backbone (into the wrapper's
      inner ``language_model``) and the FlashAR-added tensors.
    * FlashAR-only checkpoint (legacy ``{"state_dict": ...}`` or a bare state
      dict): only the FlashAR tensors are present; the backbone weights come from
      the base GLM and are expected to be missing. Mirrors
      ``glmflashar/inference/flashar_pipeline.attach_flashar``.

    Asserts every missing key is a ``backbone.*`` key and that there are no
    unexpected keys.
    """
    ckpt_path = args.resume_ckpt or args.init_ckpt
    if not ckpt_path:
        if args.step_offset < 0:
            args.step_offset = 0
        return
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if args.init_ckpt and isinstance(obj, dict) and "training_state" in obj:
        raise ValueError(
            f"--init_ckpt={args.init_ckpt} is a training-resume checkpoint. "
            "Use --resume_ckpt to restore optimizer/scheduler/data state, or use "
            "flashar_step*.pt / flashar_full_step*.pt for weight-only initialization."
        )
    if (
        args.init_ckpt
        and isinstance(obj, dict)
        and not obj.get("backbone")
        and (
            obj.get("heads_only_for_trained_backbone")
            or (obj.get("args", {}) or {}).get("train_backbone")
        )
        and not getattr(args, "allow_heads_only_init", False)
    ):
        raise ValueError(
            f"--init_ckpt={args.init_ckpt} contains only FlashAR heads from a "
            "backbone-tuned run. Use flashar_full_step*.pt, or pass "
            "--allow_heads_only_init only if this mismatch is intentional."
        )
    if isinstance(obj, dict) and "backbone" in obj:
        if not obj.get("backbone"):
            raise ValueError(f"--init_ckpt={ckpt_path} has an empty backbone payload.")
        if not (obj.get("flashar") or obj.get("state_dict")):
            raise ValueError(f"--init_ckpt={ckpt_path} has backbone tensors but no FlashAR tensors.")
    if isinstance(obj, dict) and "flashar" in obj and "backbone" not in obj:
        raise ValueError(
            f"--init_ckpt={ckpt_path} has a top-level 'flashar' payload but no backbone payload; "
            "use flashar_full_step*.pt for FULL checkpoints or flashar_step*.pt with state_dict."
        )
    if getattr(args, "require_preview_tokens", False):
        allow_legacy_text_only_init = (
            bool(args.init_ckpt)
            and getattr(args, "allow_text_only_init", False)
            and not (isinstance(obj, dict) and obj.get("preview_conditioned") is True)
            and not checkpoint_args_require_preview(obj)
        )
        if allow_legacy_text_only_init:
            pass
        else:
            validate_preview_checkpoint_semantics(
                args,
                obj,
                ckpt_path,
                context="--resume_ckpt" if args.resume_ckpt else "--init_ckpt",
                exc_type=ValueError,
            )
    if args.step_offset < 0:
        if args.resume_ckpt and isinstance(obj, dict):
            args.step_offset = int(
                obj.get(
                    "step_offset",
                    int(obj.get("step", 0)) - int(obj.get("local_step", 0)),
                )
            )
        else:
            args.step_offset = int(obj.get("step", 0)) if isinstance(obj, dict) else 0
    backbone_loaded = 0
    backbone_sd = None
    if isinstance(obj, dict) and "backbone" in obj:
        # FULL checkpoint: validate/load FlashAR tensors first. Only after that
        # succeeds do we overwrite the shared backbone, avoiding half-loaded
        # wrappers when FlashAR tensor shapes are incompatible.
        backbone_sd = normalize_state_dict_keys(obj["backbone"] or {}, strip_backbone_prefix=True)
        state = normalize_state_dict_keys(obj.get("flashar", obj.get("state_dict", {})))
        validate_checkpoint_vertical_metadata(
            obj.get("args", {}) or {},
            state,
            ckpt_path,
            expected_vertical_start_layer=getattr(wrapper, "vertical_start_layer", None),
            expected_vertical_start_layer_arg=getattr(args, "vertical_start_layer", None),
            expected_default_vertical_start_layer=(
                int(getattr(wrapper, "backbone_num_layers", 0)) - int(getattr(wrapper, "vertical_layers", 0))
                if getattr(wrapper, "backbone_num_layers", 0) and getattr(wrapper, "vertical_layers", 0)
                else None
            ),
        )
    elif isinstance(obj, dict) and "state_dict" in obj:
        state = normalize_state_dict_keys(obj["state_dict"])
        validate_checkpoint_vertical_metadata(
            obj.get("args", {}) or {},
            state,
            ckpt_path,
            expected_vertical_start_layer=getattr(wrapper, "vertical_start_layer", None),
            expected_vertical_start_layer_arg=getattr(args, "vertical_start_layer", None),
            expected_default_vertical_start_layer=(
                int(getattr(wrapper, "backbone_num_layers", 0)) - int(getattr(wrapper, "vertical_layers", 0))
                if getattr(wrapper, "backbone_num_layers", 0) and getattr(wrapper, "vertical_layers", 0)
                else None
            ),
        )
    else:
        state = normalize_state_dict_keys(obj)

    validate_state_dict_compatible(
        wrapper,
        state,
        "--init_ckpt",
        allow_missing=_is_backbone_param,
    )
    if backbone_sd is not None:
        validate_state_dict_compatible(
            wrapper.backbone,
            backbone_sd,
            "--init_ckpt full checkpoint backbone",
            allow_missing=lambda _k: False,
        )
    missing, unexpected = wrapper.load_state_dict(state, strict=False)
    non_backbone_missing = [k for k in missing if not _is_backbone_param(k)]
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys when loading --init_ckpt: {unexpected[:8]}"
            f"{' ...' if len(unexpected) > 8 else ''}"
        )
    if non_backbone_missing:
        raise RuntimeError(
            "Missing non-backbone FlashAR tensors in --init_ckpt: "
            f"{non_backbone_missing[:8]}{' ...' if len(non_backbone_missing) > 8 else ''}"
        )

    if backbone_sd is not None:
        bb_missing, bb_unexpected = wrapper.backbone.load_state_dict(
            backbone_sd, strict=False
        )
        if bb_unexpected:
            raise RuntimeError(
                f"Unexpected backbone keys in --init_ckpt: {bb_unexpected[:8]}"
                f"{' ...' if len(bb_unexpected) > 8 else ''}"
            )
        if bb_missing:
            raise RuntimeError(
                f"Missing backbone keys in --init_ckpt full checkpoint: "
                f"{bb_missing[:8]}{' ...' if len(bb_missing) > 8 else ''}"
            )
        backbone_loaded = len(backbone_sd)
    if is_main:
        if backbone_loaded:
            print(
                f"[INFO] loaded FULL checkpoint from {ckpt_path}: "
                f"{len(state)} FlashAR tensors + {backbone_loaded} trained backbone tensors.",
                flush=True,
            )
        else:
            print(f"[INFO] loaded {len(state)} FlashAR tensors from {ckpt_path} "
                  f"(backbone kept from base GLM; {len(missing)} backbone keys left untouched)",
                  flush=True)
        if args.step_offset:
            if args.resume_ckpt:
                print(
                    f"[INFO] using step_offset={args.step_offset} from resume checkpoint.",
                    flush=True,
                )
            else:
                print(
                    f"[INFO] using step_offset={args.step_offset} for checkpoint names/logged steps "
                    "(weight init only; optimizer/data state still start fresh).",
                    flush=True,
                )


# ---------------------------------------------------------------------------
# Checkpoint saving (FlashAR-added params only)
# ---------------------------------------------------------------------------

def _full_state_dict(wrapper, use_fsdp: bool):
    if not use_fsdp:
        return wrapper.state_dict()
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(wrapper, StateDictType.FULL_STATE_DICT, cfg):
        return wrapper.state_dict()


def public_args_dict(args) -> dict:
    return {k: v for k, v in vars(args).items() if not k.startswith("_")}


def atomic_torch_save(obj, path: str) -> None:
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _training_state_payload(
    args,
    optimizer,
    scheduler,
    scheduler_meta,
    *,
    global_step: int,
    epoch: int,
    next_batch_idx: int,
) -> dict:
    state = {
        "global_step": int(global_step),
        "step": int(args.step_offset) + int(global_step),
        "local_step": int(global_step),
        "step_offset": int(args.step_offset),
        "epoch": int(epoch),
        "next_batch_idx": int(next_batch_idx),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scheduler_meta": scheduler_meta,
        "data_state": resume_data_state(args),
        "training_config": resume_training_config_state(args),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    return state


def save_flashar_params(
    args,
    wrapper,
    is_main,
    step,
    use_fsdp,
    *,
    optimizer=None,
    scheduler=None,
    scheduler_meta=None,
    epoch: int = 0,
    next_batch_idx: int = 0,
) -> None:
    sd = _full_state_dict(wrapper, use_fsdp)
    if not is_main:
        return
    display_step = int(args.step_offset) + int(step)
    kept = {}
    backbone_sd = {}
    for k, v in sd.items():
        clean = normalize_param_name(k)
        if any(clean.startswith(pfx) for pfx in FLASHAR_PARAM_PREFIXES):
            kept[clean] = v
        elif clean.startswith("backbone."):
            # Strip the wrapper's ``backbone.`` prefix so keys match
            # ``language_model.state_dict()`` and load straight into the shared
            # backbone at resume / eval time.
            backbone_sd[clean[len("backbone."):]] = v
    if not kept:
        raise RuntimeError("refusing to save checkpoint with no FlashAR tensors.")
    if args.train_backbone and not backbone_sd:
        raise RuntimeError(
            "refusing to save backbone-tuned checkpoint with no backbone tensors; "
            "eval would silently fall back to base GLM or reject the artifact."
        )
    os.makedirs(args.save_dir, exist_ok=True)
    if args.train_backbone:
        out = os.path.join(args.save_dir, f"flashar_heads_only_step{display_step}.pt")
    else:
        out = os.path.join(args.save_dir, f"flashar_step{display_step}.pt")
    atomic_torch_save(
        {
            "step": display_step,
            "local_step": int(step),
            "step_offset": int(args.step_offset),
            "state_dict": kept,
            "args": public_args_dict(args),
            "preview_conditioned": bool(getattr(args, "require_preview_tokens", False)),
            "flashar_semantics_version": FLASHAR_SEMANTICS_VERSION,
            "heads_only_for_trained_backbone": bool(args.train_backbone),
            "training_state_meta": {
                "epoch": int(epoch),
                "next_batch_idx": int(next_batch_idx),
            },
        },
        out,
    )
    if args.train_backbone:
        print(
            f"[INFO] saved {len(kept)} heads-only FlashAR tensors for debugging -> {out}; "
            "use the FULL checkpoint for evaluation.",
            flush=True,
        )
    else:
        print(f"[INFO] saved {len(kept)} FlashAR tensors -> {out}", flush=True)

    # When the backbone is being trained, the FlashAR-only checkpoint above is not
    # enough to reproduce the model at eval (the trained backbone would be lost and
    # replaced by base GLM). Additionally save a FULL checkpoint bundling the
    # trained backbone so resume/eval reflect it.
    if args.train_backbone:
        full_out = os.path.join(args.save_dir, f"flashar_full_step{display_step}.pt")
        atomic_torch_save(
            {
                "step": display_step,
                "local_step": int(step),
                "step_offset": int(args.step_offset),
                "flashar": kept,
                "backbone": backbone_sd,
                "args": public_args_dict(args),
                "preview_conditioned": bool(getattr(args, "require_preview_tokens", False)),
                "flashar_semantics_version": FLASHAR_SEMANTICS_VERSION,
                "training_state_meta": {
                    "epoch": int(epoch),
                    "next_batch_idx": int(next_batch_idx),
                },
            },
            full_out,
        )
        print(
            f"[INFO] saved FULL checkpoint ({len(kept)} FlashAR + "
            f"{len(backbone_sd)} backbone tensors) -> {full_out}",
            flush=True,
        )

    if use_fsdp:
        print(
            "[INFO] skipped resume checkpoint save under FSDP; "
            "optimizer-state resume is only supported for single-process training.",
            flush=True,
        )
        return

    training_state = _training_state_payload(
        args,
        optimizer,
        scheduler,
        scheduler_meta,
        global_step=int(step),
        epoch=int(epoch),
        next_batch_idx=int(next_batch_idx),
    )
    resume_out = os.path.join(args.save_dir, f"flashar_resume_step{display_step}.pt")
    resume_payload = {
        "step": display_step,
        "local_step": int(step),
        "step_offset": int(args.step_offset),
        "training_state": training_state,
        "args": public_args_dict(args),
        "preview_conditioned": bool(getattr(args, "require_preview_tokens", False)),
        "flashar_semantics_version": FLASHAR_SEMANTICS_VERSION,
    }
    if args.train_backbone:
        resume_payload["flashar"] = kept
        resume_payload["backbone"] = backbone_sd
    else:
        resume_payload["state_dict"] = kept
    atomic_torch_save(resume_payload, resume_out)
    print(f"[INFO] saved resume checkpoint -> {resume_out}", flush=True)


def load_resume_training_state(args, optimizer, is_main):
    if not args.resume_ckpt:
        return 0, 0, 0, None, None
    obj = torch.load(args.resume_ckpt, map_location="cpu", weights_only=False)
    training_state = obj.get("training_state") if isinstance(obj, dict) else None
    if not isinstance(training_state, dict):
        raise RuntimeError(
            f"--resume_ckpt={args.resume_ckpt} does not contain training_state; "
            "use --init_ckpt for weight-only initialization."
        )
    opt_state = training_state.get("optimizer")
    if opt_state is None:
        raise RuntimeError(
            f"--resume_ckpt={args.resume_ckpt} does not contain optimizer state; "
            "use --init_ckpt for weight-only initialization."
        )
    validate_resume_data_state(args, training_state)
    validate_resume_training_config(args, training_state)
    global_step = int(training_state.get("global_step", training_state.get("local_step", 0)))
    start_epoch = int(training_state.get("epoch", 0))
    start_batch_idx = int(training_state.get("next_batch_idx", 0))

    scheduler_meta = training_state.get("scheduler_meta")
    scheduler_state = training_state.get("scheduler")
    if scheduler_state is not None:
        if not isinstance(scheduler_meta, dict) or "total_decay_steps" not in scheduler_meta:
            raise RuntimeError("resume checkpoint has scheduler state but no scheduler_meta.total_decay_steps.")
    elif scheduler_should_be_active(args, global_step):
        raise RuntimeError(
            f"--resume_ckpt={args.resume_ckpt} is at global_step={global_step}, "
            "where cosine scheduler should be active, but it has no scheduler state. "
            "Use a flashar_resume_step*.pt saved by the current training script or "
            "restart with --init_ckpt for weight-only initialization."
        )

    optimizer.load_state_dict(opt_state)
    scheduler = None
    if scheduler_state is not None:
        scheduler = start_lambda_scheduler(
            args, optimizer, int(scheduler_meta["total_decay_steps"])
        )
        scheduler.load_state_dict(scheduler_state)

    rng_state = training_state.get("torch_rng_state")
    if rng_state is not None:
        torch.set_rng_state(rng_state.cpu() if hasattr(rng_state, "cpu") else rng_state)
    cuda_rng = training_state.get("cuda_rng_state_all")
    if cuda_rng is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng)

    args.step_offset = int(
        training_state.get(
            "step_offset",
            int(training_state.get("step", global_step)) - global_step,
        )
    )
    if is_main:
        print(
            f"[INFO] resumed training state from {args.resume_ckpt}: "
            f"global_step={global_step} display_step={args.step_offset + global_step} "
            f"epoch={start_epoch} next_batch_idx={start_batch_idx}",
            flush=True,
        )
    return global_step, start_epoch, start_batch_idx, scheduler, scheduler_meta


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def resolve_shard_paths(args) -> List[str]:
    shard_paths = []
    for pattern in str(args.pretok_glob).split(","):
        pattern = pattern.strip()
        if pattern:
            matched = sorted(glob.glob(pattern))
            if not matched:
                raise FileNotFoundError(f"No pretokenized tar shards matched glob component: {pattern}")
            shard_paths.extend(os.path.abspath(path) for path in matched)
    if not shard_paths:
        raise FileNotFoundError(f"No pretokenized tar shards for glob: {args.pretok_glob}")
    return shard_paths


def validate_distributed_shard_count(
    shard_paths: List[str],
    world_size: int,
    require_preview_tokens: bool = False,
    expected_shape: Optional[Tuple[int, int]] = None,
    codebook_size: Optional[int] = None,
    deep_check: bool = True,
    require_metadata: bool = False,
) -> Tuple[int, int]:
    if int(world_size) > 1 and len(shard_paths) < int(world_size):
        raise ValueError(
            f"distributed training has world_size={world_size} but only "
            f"{len(shard_paths)} pretokenized shard entries; at least one rank "
            "would receive no data. Add more shards or repeat shard paths "
            "explicitly in --pretok_glob if intentional oversampling is desired."
        )
    pair_counts = []
    total_preview_pairs = 0
    for path in shard_paths:
        count, preview_count = summarize_pretok_pairs(path)
        count_pretok_pairs(path, require_preview=bool(require_preview_tokens), require_metadata=bool(require_metadata))
        if require_preview_tokens and preview_count != count:
            missing = count - preview_count
            raise ValueError(f"{path} has {missing}/{count} samples without .preview.pt members.")
        if deep_check or require_metadata:
            ds = PretokGlmShardDataset(
                [path],
                shuffle=False,
                expected_shape=expected_shape,
                codebook_size=codebook_size,
                require_metadata=bool(require_metadata),
            )
            loaded = 0
            for sample in ds:
                if require_preview_tokens and "preview_tokens" not in sample:
                    raise ValueError(f"{path}: loaded sample without preview_tokens despite --require_preview_tokens")
                if "preview_tokens" in sample:
                    collate_pretok_glm([sample])
                loaded += 1
            if loaded != count:
                raise ValueError(
                    f"{path}: counted {count} complete samples but loaded {loaded}; "
                    "shard contents may be inconsistent."
                )
        pair_counts.append(count)
        total_preview_pairs += preview_count
    total_pairs = sum(pair_counts)
    if total_preview_pairs not in (0, total_pairs):
        raise ValueError(
            f"mixed preview-conditioned and text-only pretokenized samples: "
            f"{total_preview_pairs}/{total_pairs} complete samples have .preview.pt. "
            "Use all preview-conditioned shards or all text-only shards for one training run."
        )
    nonempty_entries = sum(1 for count in pair_counts if count > 0)
    if int(world_size) > 1 and nonempty_entries < int(world_size):
        raise ValueError(
            f"distributed training has world_size={world_size} but only "
            f"{nonempty_entries} non-empty pretokenized shard entries; at least one rank "
            "would receive no data. Add more shards or repeat shard paths "
            "explicitly in --pretok_glob if intentional oversampling is desired."
        )
    return total_pairs, total_preview_pairs


def build_loader(args, rank, world_size):
    shard_paths = list(getattr(args, "_resolved_pretok_shards", None) or resolve_shard_paths(args))
    args._resolved_pretok_shards = shard_paths
    ds = PretokGlmShardDataset(
        shard_paths=shard_paths,
        rank=rank,
        world_size=world_size,
        seed=args.seed,
        shuffle=args.shuffle,
        shuffle_buffer_size=args.shuffle_buffer_size,
        expected_shape=(args.target_h // 32, args.target_w // 32),
        codebook_size=args.codebook_size,
        require_metadata=bool(getattr(args, "require_pretok_metadata", True)),
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pretok_glm,
    )
    return ds, loader


def train(args) -> None:
    validate_training_args(args)
    torch.manual_seed(args.seed)
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    rank, world_size, local_rank, device, use_fsdp = setup_distributed(args)
    is_main = rank == 0
    if not args.pretok_glob:
        raise ValueError("--pretok_glob with pretokenized tar shards is required.")
    validate_distributed_training_args(args, use_fsdp)
    args._resolved_pretok_shards = resolve_shard_paths(args)
    try:
        total_pairs, total_preview_pairs = validate_distributed_shard_count(
            args._resolved_pretok_shards,
            world_size,
            require_preview_tokens=getattr(args, "require_preview_tokens", False),
            expected_shape=(int(args.target_h) // 32, int(args.target_w) // 32),
            codebook_size=int(args.codebook_size),
            deep_check=bool(getattr(args, "pretok_preflight_deep_check", True)),
            require_metadata=bool(getattr(args, "require_pretok_metadata", True)),
        )
        if total_preview_pairs == total_pairs and total_pairs > 0 and not getattr(args, "require_preview_tokens", False):
            args.require_preview_tokens = True
            if is_main:
                print(
                    "[WARN] all pretokenized samples include .preview.pt; enabling "
                    "require_preview_tokens so training, checkpoints and resume config "
                    "are marked preview-conditioned.",
                    file=sys.stderr,
                    flush=True,
                )
    except Exception as exc:
        if is_main:
            print(f"PRETOK_PREFLIGHT_FAIL: {exc}", file=sys.stderr, flush=True)
            if getattr(args, "require_preview_tokens", False):
                print(
                    "Regenerate shards with python -m glmflashar.data.pretokenize_glm "
                    "and --write_preview_tokens (see README.md).",
                    file=sys.stderr,
                    flush=True,
                )
        raise SystemExit(1) from None
    if args.resume_ckpt and use_fsdp:
        raise NotImplementedError(
            "--resume_ckpt currently supports single-process training only. "
            "Use --init_ckpt for FSDP weight initialization until FSDP optimizer "
            "state save/load is implemented."
        )
    preflight_resume_checkpoint(args, is_main)

    warmup_steps = max(0, int(args.vertical_head_warmup_steps))
    if args.max_steps > 0 and warmup_steps >= args.max_steps:
        warmup_steps = max(0, args.max_steps - 1)
        if is_main:
            print(f"[WARN] clamped vertical_head_warmup_steps to {warmup_steps} "
                  f"(< max_steps={args.max_steps}).", flush=True)
    if warmup_steps == 0 and float(args.phase2_lr_factor) != 1.0 and is_main:
        print(
            "[WARN] phase2_lr_factor is ignored when vertical_head_warmup_steps=0; "
            "set phase2_lr_factor=1.0 to make this explicit.",
            flush=True,
        )

    wrapper = build_model(args, torch_dtype, device, is_main)
    if is_main:
        print(f"[INFO] vertical branch: start_layer={wrapper.vertical_start_layer} "
              f"depth={wrapper.vertical_layers} backbone_layers={wrapper.backbone_num_layers}",
              flush=True)

    # Initialize trained weights before FSDP wrap / optimizer build. This is not
    # optimizer/scheduler/data-state resume.
    load_init_checkpoint(args, wrapper, is_main)

    if use_fsdp:
        wrapper = wrap_fsdp(args, wrapper, device)
    else:
        wrapper = wrapper.to(device)
    target = wrapper.module if use_fsdp else wrapper

    freeze_backbone_if_needed(args, target)
    trainable_snapshot = snapshot_requires_grad(target)

    global_step = 0
    start_epoch = 0
    start_batch_idx = 0
    optimizer = build_optimizer(args, wrapper)
    initial_lrs = [float(pg["lr"]) for pg in optimizer.param_groups]
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    scheduler_meta: Optional[dict] = None

    if args.resume_ckpt:
        (
            global_step,
            start_epoch,
            start_batch_idx,
            scheduler,
            scheduler_meta,
        ) = load_resume_training_state(args, optimizer, is_main)
        if reached_max_steps(args, global_step):
            if is_main:
                print(
                    f"[INFO] resume checkpoint is already at global_step={global_step} "
                    f">= max_steps={args.max_steps}; nothing to train.",
                    flush=True,
                )
            if use_fsdp and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
            return

    vertical_only_active = warmup_steps > 0 and global_step < warmup_steps
    phase2_start_step = warmup_steps if warmup_steps > 0 else 0

    if vertical_only_active:
        set_vertical_only_trainable(target)
        # backbone is frozen during warmup -> its checkpointing is unnecessary
        if args.gradient_checkpointing:
            set_backbone_grad_checkpointing(target, enabled=False)
        if is_main:
            print(f"[INFO] phase-1 (vertical-only) active for steps 1..{warmup_steps}.",
                  flush=True)
        if args.phase1_flat_steps == 0 and scheduler is None:
            scheduler = maybe_create_scheduler(args, optimizer, warmup_steps - global_step)
            scheduler_meta = (
                {"kind": "cosine", "total_decay_steps": max(1, warmup_steps - global_step)}
                if scheduler is not None
                else None
            )
            if scheduler is not None and is_main:
                print(
                    f"[INFO] phase-1 cosine start @ step {global_step}, "
                    f"T_max={max(1, warmup_steps - global_step)}",
                    flush=True,
                )
    elif args.phase2_flat_steps == 0 and scheduler is None:
        scheduler = maybe_create_scheduler(args, optimizer, args.max_steps - global_step)
        scheduler_meta = (
            {"kind": "cosine", "total_decay_steps": max(1, args.max_steps - global_step)}
            if scheduler is not None
            else None
        )
        if scheduler is not None and is_main:
            print(
                f"[INFO] phase-2 cosine start @ step {global_step}, "
                f"T_max={max(1, args.max_steps - global_step)}",
                flush=True,
            )

    processor = build_glm_processor(args.model_path, args.processor_subfolder)
    pad_id = processor.tokenizer.pad_token_id
    ds, loader = build_loader(args, rank, world_size)

    accum = max(1, args.grad_accum_steps)
    wrapper.train()
    stop = False

    def maybe_start_cosine():
        nonlocal scheduler, scheduler_meta
        if args.lr_scheduler != "cosine" or args.max_steps <= 0:
            return
        if vertical_only_active:
            if (
                args.phase1_flat_steps > 0
                and global_step == args.phase1_flat_steps
                and global_step < warmup_steps
            ):
                t_max = max(1, warmup_steps - global_step)
                scheduler = maybe_create_scheduler(args, optimizer, t_max)
                scheduler_meta = (
                    {"kind": "cosine", "total_decay_steps": t_max}
                    if scheduler is not None
                    else None
                )
                if is_main:
                    print(f"[INFO] phase-1 cosine start @ step {global_step}, T_max={t_max}",
                          flush=True)
        else:
            cosine_start = phase2_start_step + max(0, args.phase2_flat_steps)
            if global_step == cosine_start:
                t_max = max(1, args.max_steps - global_step)
                scheduler = maybe_create_scheduler(args, optimizer, t_max)
                scheduler_meta = (
                    {"kind": "cosine", "total_decay_steps": t_max}
                    if scheduler is not None
                    else None
                )
                if is_main:
                    print(f"[INFO] phase-2 cosine start @ step {global_step}, T_max={t_max}",
                          flush=True)

    def maybe_finish_warmup():
        nonlocal vertical_only_active, scheduler, scheduler_meta, phase2_start_step
        if not vertical_only_active or global_step < warmup_steps:
            return
        restore_requires_grad(target, trainable_snapshot)
        if args.gradient_checkpointing:
            set_backbone_grad_checkpointing(target, enabled=True)
        vertical_only_active = False
        phase2_start_step = global_step
        for gi, pg in enumerate(optimizer.param_groups):
            pg["lr"] = initial_lrs[gi] * float(args.phase2_lr_factor)
        scheduler = None
        scheduler_meta = None
        if args.lr_scheduler == "cosine" and args.max_steps > 0 and args.phase2_flat_steps == 0:
            scheduler = maybe_create_scheduler(args, optimizer, args.max_steps - global_step)
            scheduler_meta = (
                {"kind": "cosine", "total_decay_steps": max(1, args.max_steps - global_step)}
                if scheduler is not None
                else None
            )
        if is_main:
            print(f"[INFO] phase-1 finished @ step {global_step}; restored full "
                  f"trainable set; phase2 lr factor={args.phase2_lr_factor}.", flush=True)

    ckpt_epoch = int(start_epoch)
    ckpt_next_batch_idx = int(start_batch_idx)
    last_saved_global_step: Optional[int] = None
    last_metric_payload = None

    def take_optimizer_step(loss, loss_h, loss_v, out, *, partial_micro_batches: int = 0) -> None:
        nonlocal global_step, scheduler, scheduler_meta, vertical_only_active
        if partial_micro_batches:
            grad_scale = float(accum) / float(partial_micro_batches)
            for p in wrapper.parameters():
                if p.grad is not None:
                    p.grad.mul_(grad_scale)
        if args.grad_clip > 0:
            if use_fsdp:
                wrapper.clip_grad_norm_(args.grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(wrapper.parameters(), args.grad_clip)
        debug_str = ""
        if _FLASHAR_DEBUG and is_main:
            bb_sq, fa_sq = 0.0, 0.0
            for n, p in target.named_parameters():
                if p.grad is None:
                    continue
                g2 = float(p.grad.detach().float().pow(2).sum())
                if _is_backbone_param(n):
                    bb_sq += g2
                else:
                    fa_sq += g2
            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3) \
                if torch.cuda.is_available() else 0.0
            debug_str = (f" backbone_grad_norm={bb_sq ** 0.5:.6f} "
                         f"flashar_grad_norm={fa_sq ** 0.5:.6f} "
                         f"peak_mem_gb={peak:.2f}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
        global_step += 1
        if partial_micro_batches and is_main:
            print(
                f"[WARN] committed partial gradient accumulation with "
                f"{partial_micro_batches}/{accum} micro-batches at epoch {epoch}.",
                flush=True,
            )

        if is_main and args.log_every_steps > 0 and global_step % args.log_every_steps == 0:
            display_step = int(args.step_offset) + int(global_step)
            print(
                f"[METRIC] step={display_step} local_step={global_step} "
                f"loss={loss.item():.6f} loss_h={float(loss_h):.6f} "
                f"loss_v={float(loss_v):.6f} "
                f"gate_h={float(out.get('hv_gate_h', 0.5)):.4f} "
                f"gate_entropy={float(out.get('hv_gate_entropy', 1.0)):.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.3e} "
                f"phase={'1(vert)' if vertical_only_active else '2(full)'}"
                f"{debug_str}",
                flush=True,
            )

        maybe_start_cosine()
        maybe_finish_warmup()

    for epoch in range(start_epoch, args.epochs):
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(epoch)
        data_iter = iter(loader)
        step = 0
        if epoch == start_epoch and start_batch_idx > 0:
            skipped = skip_resume_batches(
                data_iter,
                start_batch_idx,
                device,
                use_distributed=(use_fsdp and dist.is_initialized() and world_size > 1),
            )
            step = skipped
            if is_main:
                print(
                    f"[INFO] resume cursor skipped {skipped}/{start_batch_idx} "
                    f"batches in epoch {epoch}.",
                    flush=True,
                )
        while True:
            batch, all_have_batch, local_had_batch = next_batch_all_ranks(
                data_iter,
                device,
                use_distributed=(use_fsdp and dist.is_initialized() and world_size > 1),
            )
            if not all_have_batch:
                partial_micro_batches = step % accum
                if partial_micro_batches != 0 and not use_fsdp and last_metric_payload is not None:
                    take_optimizer_step(*last_metric_payload, partial_micro_batches=partial_micro_batches)
                    if args.max_steps > 0 and global_step >= args.max_steps:
                        ckpt_epoch = epoch
                        ckpt_next_batch_idx = step
                        stop = True
                elif partial_micro_batches != 0:
                    optimizer.zero_grad(set_to_none=True)
                    if is_main:
                        print(
                            f"[WARN] dropped partial gradient accumulation at epoch {epoch} "
                            f"after {partial_micro_batches}/{accum} micro-batches.",
                            flush=True,
                        )
                if local_had_batch and is_main:
                    print(
                        f"[WARN] discarded an extra local batch at epoch {epoch} "
                        "because another rank exhausted its shard split.",
                        flush=True,
                    )
                ckpt_epoch = epoch + 1
                ckpt_next_batch_idx = 0
                break

            tokens = batch["tokens"].long()               # (B, H, W)
            texts = batch["texts"]
            height = int(tokens.size(1))
            width = int(tokens.size(2))
            input_ids = tokens.view(tokens.size(0), -1).to(device)   # (B, H*W)
            text_ids, text_mask = build_glm_text_prefix(
                processor, texts, target_h=args.target_h, target_w=args.target_w,
                pad_token_id=pad_id, device=device,
            )
            preview_tokens = batch.get("preview_tokens")
            preview_h = 0
            preview_w = 0
            if preview_tokens is not None:
                if preview_tokens.dim() == 3:
                    preview_h = int(preview_tokens.size(1))
                    preview_w = int(preview_tokens.size(2))
                elif preview_tokens.dim() == 2:
                    preview_h, preview_w = target._preview_grid_shape(height, width)
                else:
                    raise ValueError(f"preview_tokens must be 2-D or 3-D batch tensor, got {tuple(preview_tokens.shape)}")
                preview_tokens = preview_tokens.to(device=device, dtype=torch.long)
            elif getattr(args, "require_preview_tokens", False):
                raise ValueError(
                    "--require_preview_tokens was set but the current batch has no preview_tokens. "
                    "Regenerate pretokenized shards with {stem}.preview.pt members."
                )

            is_sync = (step + 1) % accum == 0
            if use_fsdp and args.fsdp_no_sync and not is_sync and accum > 1:
                sync_ctx = wrapper.no_sync()
            else:
                sync_ctx = nullcontext()

            with sync_ctx:
                out = wrapper(
                    input_ids=input_ids,
                    height=height,
                    width=width,
                    text_input_ids=text_ids,
                    text_attention_mask=text_mask,
                    preview_input_ids=preview_tokens,
                    preview_height=preview_h,
                    preview_width=preview_w,
                    chunked_loss=args.chunked_loss,
                )
                loss = out["loss"]
                loss_h = out.get("loss_h", loss.detach().new_zeros(()))
                loss_v = out.get("loss_v", loss.detach().new_zeros(()))
                loss_gc = out.get("loss_gate_collapse", loss.detach().new_zeros(()))
                aux_h = 0.0 if vertical_only_active else float(args.aux_loss_h_weight)
                total = (
                    loss
                    + aux_h * loss_h
                    + float(args.aux_loss_v_weight) * loss_v
                    + float(args.gate_collapse_weight) * loss_gc
                )
                (total / accum).backward()
                last_metric_payload = (loss, loss_h, loss_v, out)

            if is_sync:
                take_optimizer_step(loss, loss_h, loss_v, out)

                if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                    ckpt_epoch = epoch
                    ckpt_next_batch_idx = step + 1
                    save_flashar_params(
                        args,
                        wrapper,
                        is_main,
                        global_step,
                        use_fsdp,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scheduler_meta=scheduler_meta,
                        epoch=ckpt_epoch,
                        next_batch_idx=ckpt_next_batch_idx,
                    )
                    last_saved_global_step = int(global_step)

                if args.max_steps > 0 and global_step >= args.max_steps:
                    ckpt_epoch = epoch
                    ckpt_next_batch_idx = step + 1
                    stop = True
                    break
            step += 1
        if stop:
            break

    if last_saved_global_step != int(global_step):
        save_flashar_params(
            args,
            wrapper,
            is_main,
            global_step,
            use_fsdp,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_meta=scheduler_meta,
            epoch=ckpt_epoch,
            next_batch_idx=ckpt_next_batch_idx,
        )
    elif is_main:
        print(
            f"[INFO] final checkpoint for global_step={global_step} was already saved; "
            "skipping duplicate final save.",
            flush=True,
        )
    if use_fsdp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if is_main:
        print("[INFO] training complete.", flush=True)


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
