# -*- coding: utf-8 -*-
"""FlashAR-driven GLM-Image pipeline.

``GlmImageFlashARPipeline`` subclasses diffusers' :class:`GlmImagePipeline` and
overrides ONLY the AR-prior generation of the LARGE (d32) token grid. Instead of
the stock ``vision_language_encoder.generate(...)`` raster-scan autoregression,
the large grid is produced by :class:`GlmImageFlashAR.generate` (anti-diagonal
parallel decoding, ``H + W - 1`` diagonal steps for an ``H x W`` grid). Everything
downstream -- 2x nearest upsampling of the token grid, DiT denoising with
classifier-free guidance, and VAE decoding -- stays 100% stock.

--------------------------------------------------------------------------------
prior_token_ids CONTRACT (the critical correctness point)
--------------------------------------------------------------------------------
We reproduce EXACTLY what the stock ``GlmImagePipeline.generate_prior_tokens``
returns, so ``__call__`` needs zero changes. Read alongside
``diffusers/pipelines/glm_image/pipeline_glm_image.py``:

  * The stock method AR-generates the d32 large grid (``token_h x token_w`` flat
    codebook ids), then calls ``self._upsample_token_ids(ids, token_h, token_w)``
    which reshapes to ``(1, 1, token_h, token_w)``, does a 2x ``nearest``
    ``F.interpolate``, and returns ``(1, 4 * token_h * token_w)`` as
    ``torch.long``. Per-sample rows are concatenated along dim 0.

  * ``__call__`` receives this as ``prior_token_ids`` (shape
    ``(batch_size, 4 * token_h * token_w)``, dtype ``torch.long``, on the
    execution device), optionally ``repeat_interleave``s it for
    ``num_images_per_prompt``, and feeds it verbatim to the DiT as
    ``prior_token_id=...``. It is NEVER reshaped/upsampled again in ``__call__``.

  * The method returns a 3-tuple
    ``(prior_token_ids, prior_token_image_ids_per_sample,
      source_image_grid_thw_per_sample)``. The latter two are the i2i condition
    tokens and are ``None`` for text-to-image.

For a 1024x1024 target: ``token_h = token_w = 1024 // 32 = 32`` (the d32 large
grid = 32x32 = 1024 tokens), upsampled to 64x64 = 4096 -> returned shape
``(batch_size, 4096)``, dtype ``torch.long``, on ``device``. Our override matches
this shape/dtype/device exactly by reusing the inherited ``_upsample_token_ids``.

--------------------------------------------------------------------------------
LIMITATIONS (documented)
--------------------------------------------------------------------------------
1. PREVIEW CONDITIONING. Stock GLM AR first emits a small preview grid (16x16
   for a square target) and then the large d32 grid; only the LARGE grid is fed
   to the DiT. FlashAR keeps that conditioning path by generating the preview
   grid with stock AR, then decoding only the large grid anti-diagonally.

2. TEXT-TO-IMAGE ONLY. Image-conditioned (i2i) generation is not supported by the
   FlashAR wrapper; ``generate_prior_tokens`` raises if ``image`` is provided.

3. EAGER ATTENTION REQUIRED. FlashAR drives the backbone with explicit 4-D
   additive proximity/causal masks (as in training, which loads the VLE with
   ``attn_implementation="eager"``). ``attach_flashar`` refuses non-eager
   backbones so evaluation cannot silently use a different attention path.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import PIL
import torch

try:
    from diffusers.pipelines.glm_image import GlmImagePipeline
except Exception as exc:
    raise RuntimeError(
        "GLM-Image diffusers pipeline is unavailable or incompatible in this Python "
        "environment. Install the dependencies in requirements.txt."
    ) from exc

from glmflashar.model.glm_backbone_patch import apply_glm_backbone_patch, set_layers_non_causal
from glmflashar.model.modeling_glm_flashar import GlmImageFlashAR, _validate_sampling_args
from glmflashar.constants import FLASHAR_SEMANTICS_VERSION
from glmflashar.utils.preview import stock_preview_shape_from_pixels
from glmflashar.utils.text_utils_glm import build_glm_text_prefix


# GLM-Image downsamples by 32x for the d32 token grid.
GLM_D32_FACTOR = 32


def _attention_implementation(language_model) -> Optional[str]:
    config = getattr(language_model, "config", None)
    if config is None:
        return None
    return getattr(config, "_attn_implementation", None) or getattr(config, "attn_implementation", None)


def _require_eager_attention(language_model) -> None:
    attn_impl = _attention_implementation(language_model)
    if str(attn_impl) != "eager":
        raise RuntimeError(
            "FlashAR evaluation requires the GLM language model to use "
            'attn_implementation="eager" so its 4-D additive masks match training; '
            f"got {attn_impl!r}."
        )


def _infer_vertical_layers_from_state(state_dict: dict) -> int:
    """Infer the number of ``vertical_block`` decoder layers from checkpoint keys."""
    max_idx = -1
    for raw_key in state_dict:
        key = _normalize_param_name(raw_key)
        if key.startswith("vertical_block."):
            try:
                idx = int(key.split(".")[1])
            except (IndexError, ValueError):
                continue
            max_idx = max(max_idx, idx)
    return max_idx + 1


def _normalize_param_name(key: str) -> str:
    clean = str(key).replace("_fsdp_wrapped_module.", "")
    while clean.startswith("module."):
        clean = clean[len("module."):]
    return clean


def _normalize_state_dict_keys(state: dict, *, strip_backbone_prefix: bool = False) -> dict:
    normalized = {}
    for raw_key, value in state.items():
        key = _normalize_param_name(raw_key)
        if strip_backbone_prefix and key.startswith("backbone."):
            key = key[len("backbone."):]
        if key in normalized:
            raise ValueError(f"checkpoint has duplicate tensor key after normalization: {key}")
        normalized[key] = value
    return normalized


def _is_backbone_key(key: str) -> bool:
    clean = _normalize_param_name(key)
    return clean.startswith("backbone.") or ".backbone." in clean


def _validate_state_dict_compatible(module, state: dict, context: str, allow_missing) -> None:
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


def _load_flashar_checkpoint(ckpt_path: str) -> Tuple[dict, dict, dict]:
    """Load a ``flashar_step*.pt`` or ``flashar_full_step*.pt`` checkpoint.

    Returns ``(state_dict, meta_args, backbone_sd)`` where:
      * ``state_dict`` holds only the FlashAR-added tensors (heads, vertical
        block/norm, gates);
      * ``meta_args`` is the training-time ``vars(args)`` dict (may be empty for
        bare state dicts);
      * ``backbone_sd`` holds the TRAINED backbone tensors (keys matching
        ``language_model.state_dict()``) for a FULL checkpoint, else an empty
        dict. When non-empty these must be loaded into the shared backbone so eval
        reflects the trained backbone.
    """
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "training_state" in obj:
        raise ValueError(
            "flashar_resume_step*.pt is a training-resume checkpoint and should "
            "not be used for evaluation/generation. Use flashar_full_step*.pt "
            "for backbone-tuned runs or flashar_step*.pt for frozen-backbone runs."
        )
    if (
        isinstance(obj, dict)
        and not obj.get("backbone")
        and (obj.get("heads_only_for_trained_backbone") or (obj.get("args", {}) or {}).get("train_backbone"))
    ):
        raise ValueError(
            "This checkpoint contains only FlashAR heads from a backbone-tuned run. "
            "Use flashar_full_step*.pt so the trained backbone is loaded."
        )
    if isinstance(obj, dict) and "backbone" in obj and not obj.get("backbone"):
        raise ValueError(
            "FULL checkpoint has an empty backbone payload. Use a valid "
            "flashar_full_step*.pt with trained backbone tensors."
        )
    if isinstance(obj, dict) and "flashar" in obj and "backbone" not in obj:
        raise ValueError(
            "checkpoint has a top-level 'flashar' payload but no backbone payload. "
            "Use flashar_full_step*.pt for FULL checkpoints or flashar_step*.pt "
            "with a top-level state_dict for frozen-backbone checkpoints."
        )
    backbone_sd: dict = {}
    if isinstance(obj, dict) and "backbone" in obj:
        # FULL checkpoint bundling the trained backbone.
        backbone_sd = _normalize_state_dict_keys(
            obj.get("backbone", {}) or {}, strip_backbone_prefix=True
        )
        state_dict = _normalize_state_dict_keys(obj.get("flashar", obj.get("state_dict", {})))
        meta_args = obj.get("args", {}) or {}
        meta_args["_preview_conditioned_checkpoint"] = obj.get("preview_conditioned") is True
        meta_args["_flashar_semantics_version"] = str(obj.get("flashar_semantics_version", ""))
    elif isinstance(obj, dict) and "state_dict" in obj:
        state_dict = _normalize_state_dict_keys(obj["state_dict"])
        meta_args = obj.get("args", {}) or {}
        meta_args["_preview_conditioned_checkpoint"] = obj.get("preview_conditioned") is True
        meta_args["_flashar_semantics_version"] = str(obj.get("flashar_semantics_version", ""))
    else:
        # Bare state dict fallback.
        state_dict = _normalize_state_dict_keys(obj)
        meta_args = {"_preview_conditioned_checkpoint": False, "_flashar_semantics_version": ""}
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(
            "FlashAR checkpoint has no FlashAR tensors. Use a valid "
            "flashar_full_step*.pt or flashar_step*.pt checkpoint."
        )
    inferred_vlayers = _infer_vertical_layers_from_state(state_dict)
    meta_use_vb = meta_args.get("use_vertical_block")
    if meta_use_vb is False and inferred_vlayers > 0:
        raise ValueError(
            "checkpoint metadata says use_vertical_block=False but vertical_block "
            "tensors are present."
        )
    if meta_use_vb is True and inferred_vlayers <= 0:
        raise ValueError(
            "checkpoint metadata says use_vertical_block=True but no vertical_block "
            "tensors are present."
        )
    meta_vlayers = meta_args.get("vertical_layers")
    if inferred_vlayers > 0 and meta_vlayers is not None and int(meta_vlayers) != inferred_vlayers:
        raise ValueError(
            f"checkpoint vertical_layers metadata ({meta_vlayers}) does not match "
            f"vertical_block tensors ({inferred_vlayers})."
        )
    return state_dict, meta_args, backbone_sd


class GlmImageFlashARPipeline(GlmImagePipeline):
    """GLM-Image pipeline whose large-grid AR prior comes from FlashAR decoding.

    Usage::

        from glmflashar.inference.flashar_pipeline import GlmImageFlashARPipeline
        pipe = GlmImageFlashARPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation="eager",
        ).to("cuda")
        pipe.attach_flashar(ckpt_path="outputs/glm_flashar_run10_ropefix/flashar_full_stepXXXX.pt")
        image = pipe(prompt="a photo of a red car", height=1024, width=1024,
                     guidance_scale=5.0, num_inference_steps=50).images[0]
        print("AR diagonal steps:", pipe.last_ar_num_steps)  # == H + W - 1
    """

    # Populated by attach_flashar / generate_prior_tokens.
    _flashar: Optional[GlmImageFlashAR] = None
    last_ar_num_steps: Optional[int] = None
    last_preview_num_tokens: Optional[int] = None

    # Default FlashAR sampling controls (override via attach_flashar or setters).
    flashar_temperature: float = 0.9
    flashar_top_k: int = 16512
    flashar_top_p: float = 0.75
    flashar_sample_logits: bool = True
    flashar_use_stock_preview: bool = True
    # KV-cache diagonal decode is faster but currently fails teacher-forced
    # equivalence against the training full-forward path. Prefer the slower
    # recompute path by default until KV parity is fixed.
    flashar_use_kv_cache: bool = False

    # ------------------------------------------------------------------
    # Attaching the FlashAR wrapper
    # ------------------------------------------------------------------

    def attach_flashar(
        self,
        ckpt_path: Optional[str] = None,
        state_dict: Optional[dict] = None,
        *,
        backbone_state_dict: Optional[dict] = None,
        vocab_size: Optional[int] = None,
        hidden_size: Optional[int] = None,
        codebook_size: Optional[int] = None,
        use_vertical_block: Optional[bool] = None,
        vertical_layers: Optional[int] = None,
        vertical_start_layer: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        sample_logits: Optional[bool] = None,
        use_kv_cache: Optional[bool] = None,
        use_stock_preview: Optional[bool] = None,
        preview_conditioned: Optional[bool] = None,
        flashar_semantics_version: Optional[str] = None,
    ) -> "GlmImageFlashARPipeline":
        """Build a ``GlmImageFlashAR`` on top of ``self.vision_language_encoder``
        and load the trained FlashAR tensors into it.

        The backbone (``vision_language_encoder.model.language_model``) and
        ``lm_head`` are SHARED with the pipeline (not copied), so only the
        FlashAR-added tensors are loaded from the checkpoint; the shared backbone
        weights come from the pipeline. Constructor hyper-parameters default to the
        values recorded in the checkpoint's ``args`` (falling back to sensible
        GLM-Image defaults / inference from the state dict).
        """
        if ckpt_path is None and state_dict is None:
            raise ValueError("attach_flashar requires either ckpt_path or state_dict.")

        meta_args: dict = {}
        backbone_sd: dict = _normalize_state_dict_keys(
            backbone_state_dict or {}, strip_backbone_prefix=True
        )
        if state_dict is None:
            state_dict, meta_args, loaded_backbone_sd = _load_flashar_checkpoint(ckpt_path)
            if not backbone_sd:
                backbone_sd = loaded_backbone_sd
        else:
            state_dict = _normalize_state_dict_keys(state_dict)
            meta_args["_preview_conditioned_checkpoint"] = bool(preview_conditioned)
            if bool(preview_conditioned):
                if flashar_semantics_version is None:
                    raise ValueError(
                        "attach_flashar(state_dict=..., preview_conditioned=True) requires "
                        "flashar_semantics_version so stale preview-conditioned weights cannot "
                        "be treated as current."
                    )
                meta_args["_flashar_semantics_version"] = str(flashar_semantics_version)
            else:
                meta_args["_flashar_semantics_version"] = ""
        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError("attach_flashar received no FlashAR tensors to load.")

        # Resolve constructor hyper-parameters: explicit arg > checkpoint meta >
        # default / inferred-from-state.
        def _pick(explicit, key, default):
            if explicit is not None:
                return explicit
            if key in meta_args and meta_args[key] is not None:
                return meta_args[key]
            return default

        inferred_vlayers = _infer_vertical_layers_from_state(state_dict)
        resolved_vocab = int(_pick(vocab_size, "vocab_size", 16512))
        resolved_hidden = int(_pick(hidden_size, "hidden_size", 4096))
        resolved_codebook = int(_pick(codebook_size, "codebook_size", 16384))
        resolved_use_vb = bool(
            _pick(use_vertical_block, "use_vertical_block", inferred_vlayers > 0)
        )
        resolved_vlayers = int(
            _pick(vertical_layers, "vertical_layers", inferred_vlayers if inferred_vlayers > 0 else 4)
        )
        resolved_vstart = int(_pick(vertical_start_layer, "vertical_start_layer", -1))
        meta_vstart = meta_args.get("vertical_start_layer")
        if vertical_start_layer is not None and meta_vstart is not None and int(meta_vstart) != resolved_vstart:
            raise RuntimeError(
                f"FlashAR checkpoint vertical_start_layer metadata ({meta_vstart}) does not match "
                f"requested vertical_start_layer ({resolved_vstart})."
            )

        vle = self.vision_language_encoder
        language_model = vle.model.language_model
        _require_eager_attention(language_model)
        lm_head = vle.lm_head

        wrapper = GlmImageFlashAR(
            language_model,
            vocab_size=resolved_vocab,
            hidden_size=resolved_hidden,
            codebook_size=resolved_codebook,
            use_vertical_block=resolved_use_vb,
            vertical_layers=resolved_vlayers,
            vertical_start_layer=resolved_vstart,
            lm_head=lm_head,
        )

        # Move the FlashAR-added modules onto the backbone's device/dtype. The
        # backbone params are shared (already placed); .to() is a no-op for them.
        backbone_param = next(language_model.parameters())
        wrapper = wrapper.to(device=backbone_param.device, dtype=backbone_param.dtype)

        # Load ONLY the FlashAR tensors; backbone keys are expected to be missing
        # (they are shared with the pipeline and already correct).
        _validate_state_dict_compatible(
            wrapper,
            state_dict,
            "FlashAR checkpoint",
            allow_missing=_is_backbone_key,
        )
        if backbone_sd:
            _validate_state_dict_compatible(
                language_model,
                backbone_sd,
                "FlashAR full checkpoint backbone",
                allow_missing=lambda _k: False,
            )

        # Sampling controls: validate before mutating the shared backbone or
        # attaching the wrapper, so bad user args leave the pipeline unchanged.
        gen_cfg = getattr(vle, "generation_config", None)
        default_temperature = getattr(gen_cfg, "temperature", None)
        default_top_k = getattr(gen_cfg, "top_k", None)
        default_top_p = getattr(gen_cfg, "top_p", None)
        if default_temperature is None:
            default_temperature = type(self).flashar_temperature
        if default_top_k is None:
            default_top_k = type(self).flashar_top_k
        if default_top_p is None:
            default_top_p = type(self).flashar_top_p
        next_temperature = float(default_temperature) if temperature is None else float(temperature)
        next_top_k = int(default_top_k) if top_k is None else int(top_k)
        next_top_p = float(default_top_p) if top_p is None else float(top_p)
        next_sample_logits = True if sample_logits is None else bool(sample_logits)
        next_use_kv_cache = False if use_kv_cache is None else bool(use_kv_cache)
        _validate_sampling_args(next_temperature, next_top_k, next_top_p)
        if next_use_kv_cache and os.environ.get("GLM_FLASHAR_ENABLE_EXPERIMENTAL_KV_CACHE") != "1":
            raise RuntimeError(
                "FlashAR KV-cache diagonal decode is disabled because it fails "
                "teacher-forced equivalence; use the default recompute path. Set "
                "GLM_FLASHAR_ENABLE_EXPERIMENTAL_KV_CACHE=1 only for local "
                "timing/debug experiments."
            )
        next_use_stock_preview = (
            True
            if use_stock_preview is None
            else bool(use_stock_preview)
        )
        if (not next_use_stock_preview) and bool(meta_args.get("_preview_conditioned_checkpoint")):
            raise RuntimeError(
                "use_stock_preview=False is only for legacy text-only checkpoints; "
                "this checkpoint is marked preview-conditioned."
            )
        if next_use_stock_preview and not bool(meta_args.get("_preview_conditioned_checkpoint")):
            raise RuntimeError(
                "FlashAR stock-preview evaluation requires a preview-conditioned checkpoint. "
                "Regenerate/train with --require_preview_tokens, or attach with "
                "use_stock_preview=False only for legacy text-only evaluation."
            )
        if (
            bool(meta_args.get("_preview_conditioned_checkpoint"))
            and str(meta_args.get("_flashar_semantics_version", "")) != FLASHAR_SEMANTICS_VERSION
        ):
            raise RuntimeError(
                "FlashAR stock-preview evaluation requires a checkpoint saved with "
                f"flashar_semantics_version={FLASHAR_SEMANTICS_VERSION!r}; got "
                f"{meta_args.get('_flashar_semantics_version', '')!r}."
            )

        # Patch the inner text model only after all preflight checks pass. A failed
        # attach should not leave a stock GLM pipeline globally monkeypatched.
        apply_glm_backbone_patch()
        set_layers_non_causal(language_model, True)
        if wrapper.vertical_block is not None:
            set_layers_non_causal(wrapper.vertical_block, True)

        missing, unexpected = wrapper.load_state_dict(state_dict, strict=False)
        non_backbone_missing = [
            k for k in missing if not (k.startswith("backbone.") or ".backbone." in k)
        ]
        if unexpected:
            raise RuntimeError(
                f"Unexpected keys when loading FlashAR checkpoint: {unexpected[:8]}"
                f"{' ...' if len(unexpected) > 8 else ''}"
            )
        if non_backbone_missing:
            raise RuntimeError(
                "Missing non-backbone FlashAR tensors in checkpoint: "
                f"{non_backbone_missing[:8]}{' ...' if len(non_backbone_missing) > 8 else ''}"
            )

        # FULL checkpoint: overwrite the shared base-GLM backbone with the TRAINED
        # backbone only after FlashAR tensors are known to load cleanly. This avoids
        # leaving a pipeline half-mutated if FlashAR tensor shapes are incompatible.
        if backbone_sd:
            bb_missing, bb_unexpected = language_model.load_state_dict(
                backbone_sd, strict=False
            )
            if bb_unexpected:
                raise RuntimeError(
                    f"Unexpected backbone keys in FlashAR full checkpoint: "
                    f"{bb_unexpected[:8]}{' ...' if len(bb_unexpected) > 8 else ''}"
                )
            if bb_missing:
                raise RuntimeError(
                    f"Missing backbone keys in FlashAR full checkpoint: "
                    f"{bb_missing[:8]}{' ...' if len(bb_missing) > 8 else ''}"
                )
            print(
                f"[FlashAR] loaded {len(backbone_sd)} TRAINED backbone tensors into "
                f"the shared vision_language_encoder backbone.",
                flush=True,
            )

        wrapper.eval()
        self._flashar = wrapper

        self.flashar_temperature = next_temperature
        self.flashar_top_k = next_top_k
        self.flashar_top_p = next_top_p
        self.flashar_sample_logits = next_sample_logits
        self.flashar_use_kv_cache = next_use_kv_cache
        self.flashar_use_stock_preview = next_use_stock_preview
        return self

    # ------------------------------------------------------------------
    # Overridden prior-token generation (LARGE grid via FlashAR)
    # ------------------------------------------------------------------

    def _generate_stock_preview_tokens(
        self,
        prompt: str,
        height: int,
        width: int,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], int, int]:
        """Generate stock GLM's preview grid to condition FlashAR large-grid decode."""
        messages = [[{"role": "user", "content": [{"type": "text", "text": prompt}]}]]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=False,
            target_h=height,
            target_w=width,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is None or int(image_grid_thw.shape[0]) < 2:
            raise RuntimeError("stock preview generation requires a target preview grid, but processor returned none")
        _max_new, large_offset, _token_h, _token_w = self._compute_generation_params(
            image_grid_thw=image_grid_thw,
            is_text_to_image=True,
        )
        preview_tokens = int(large_offset)
        if preview_tokens <= 0:
            raise RuntimeError(f"stock preview generation produced invalid preview token count: {preview_tokens}")
        preview_grid = image_grid_thw[1]
        preview_h = int(preview_grid[1].item())
        preview_w = int(preview_grid[2].item())
        if preview_tokens != preview_h * preview_w:
            raise RuntimeError(
                f"stock preview token count mismatch: offset={preview_tokens} "
                f"grid={preview_h}x{preview_w}"
            )
        # Preview generation should retain stock AR causal semantics. attach_flashar
        # puts the shared backbone in non-causal mode for FlashAR's explicit masks,
        # so temporarily restore causal layer flags around the stock generate call.
        language_model = self.vision_language_encoder.model.language_model
        set_layers_non_causal(language_model, False)
        try:
            outputs = self.vision_language_encoder.generate(
                **inputs,
                max_new_tokens=preview_tokens,
                do_sample=True,
            )
        finally:
            set_layers_non_causal(language_model, True)
        input_len = int(inputs["input_ids"].shape[-1])
        preview_flat = outputs[0, input_len: input_len + preview_tokens]
        if int(preview_flat.numel()) != preview_tokens:
            raise RuntimeError(
                f"stock preview generation too short: got {int(preview_flat.numel())}, "
                f"expected {preview_tokens}"
            )
        preview_ids = preview_flat.view(1, -1).to(
            device=device,
            dtype=torch.long,
        )
        return preview_ids, preview_h, preview_w

    def generate_prior_tokens(
        self,
        prompt,
        height: int,
        width: int,
        image: Optional[List[List["PIL.Image.Image"]]] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ):
        """Produce the large-grid prior tokens with FlashAR anti-diagonal decoding.

        Returns the SAME 3-tuple as the stock method:
            ``(prior_token_ids, prior_token_image_ids_per_sample,
               source_image_grid_thw_per_sample)``
        with the latter two ``None`` (text-to-image only). ``prior_token_ids`` has
        shape ``(batch_size, 4 * token_h * token_w)`` (e.g. ``(B, 4096)`` for
        1024x1024), dtype ``torch.long``, on ``device`` -- exactly what
        ``__call__`` feeds to the DiT.
        """
        if self._flashar is None:
            raise RuntimeError(
                "FlashAR wrapper is not attached. Call attach_flashar(...) first."
            )
        if image is not None:
            raise NotImplementedError(
                "GlmImageFlashARPipeline supports text-to-image only; got condition images."
            )

        device = device or self._execution_device
        prompt_list = [prompt] if isinstance(prompt, str) else list(prompt)
        if not prompt_list:
            raise ValueError("prompt must contain at least one item.")
        for idx, item in enumerate(prompt_list):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"prompt item {idx} must be a non-empty string.")

        if int(height) <= 0 or int(width) <= 0:
            raise ValueError(f"height/width must be positive; got {height}x{width}.")
        if height % GLM_D32_FACTOR != 0 or width % GLM_D32_FACTOR != 0:
            raise ValueError(
                f"height/width must be multiples of {GLM_D32_FACTOR}; got {height}x{width}."
            )
        stock_preview_shape_from_pixels(height, width, downsample=GLM_D32_FACTOR)
        token_h = height // GLM_D32_FACTOR
        token_w = width // GLM_D32_FACTOR

        # Mirror the stock method's generator->seed handling for reproducibility.
        if generator is not None:
            seed = generator.initial_seed()
            torch.manual_seed(seed)
            if device is not None and torch.device(device).type == "cuda":
                torch.cuda.manual_seed(seed)

        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)

        all_prior_token_ids = []
        num_steps = None
        preview_tokens_used = 0
        for p in prompt_list:
            # Build the text+shape prefix identically to training
            # (build_glm_text_prefix -> processor.apply_chat_template with
            # target_h/target_w). batch_size=1 per call: the FlashAR wrapper's
            # diagonal decoder operates on a single grid.
            text_ids, text_mask = build_glm_text_prefix(
                self.processor,
                [p],
                target_h=height,
                target_w=width,
                pad_token_id=pad_id,
                device=device,
            )
            preview_ids = None
            preview_h = 0
            preview_w = 0
            if self.flashar_use_stock_preview:
                preview_ids, preview_h, preview_w = self._generate_stock_preview_tokens(
                    p,
                    height=height,
                    width=width,
                    device=device,
                )
                preview_tokens_used = int(preview_h) * int(preview_w)
            grid, num_steps = self._flashar.generate(
                height=token_h,
                width=token_w,
                device=device,
                text_input_ids=text_ids,
                text_attention_mask=text_mask,
                preview_input_ids=preview_ids,
                preview_height=preview_h,
                preview_width=preview_w,
                temperature=self.flashar_temperature,
                top_k=self.flashar_top_k,
                top_p=self.flashar_top_p,
                sample_logits=self.flashar_sample_logits,
                use_kv_cache=self.flashar_use_kv_cache,
                return_num_steps=True,
            )
            # (token_h, token_w) codebook grid -> flat d32 ids -> 2x nearest upsample.
            flat_d32 = grid.reshape(1, token_h * token_w).to(device=device, dtype=torch.long)
            upsampled = self._upsample_token_ids(flat_d32, token_h, token_w)  # (1, 4*h*w)
            all_prior_token_ids.append(upsampled)

        prior_token_ids = torch.cat(all_prior_token_ids, dim=0)
        self.last_ar_num_steps = num_steps
        self.last_preview_num_tokens = preview_tokens_used
        return prior_token_ids, None, None


__all__ = ["GlmImageFlashARPipeline"]
