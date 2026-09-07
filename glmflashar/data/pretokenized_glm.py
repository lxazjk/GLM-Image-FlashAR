# -*- coding: utf-8 -*-
"""Pretokenized tar dataset for GLM-Image FlashAR training.

Each tar shard produced by
``glmflashar/data/pretokenize_glm.py`` contains, per sample:
  * ``{stem}.pt``  -- a torch-saved LongTensor of shape (32, 32) holding the
    d32 GLM-Image codebook ids (values in ``[0, 16384)``).
  * ``{stem}.txt`` -- the UTF-8 caption for that image.

The dataset yields ``{"tokens": (32, 32) LongTensor, "text": str}`` and
``collate_pretok_glm`` stacks the token grids to ``(B, 32, 32)`` and returns the
captions as a list under ``"texts"``.
"""

from __future__ import annotations

import hashlib
import io
import json
import os.path as osp
import random
import tarfile
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from glmflashar.utils.preview import stock_preview_shape_from_large_grid


PRETOK_METADATA_SCHEMA = "glmflashar_pretok_v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_sha256_hex(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def tensor_payload_sha256(tensor: torch.Tensor) -> str:
    buf = io.BytesIO()
    torch.save(tensor.detach().cpu().contiguous(), buf)
    return sha256_bytes(buf.getvalue())


def _coerce_shape2(name: str, value) -> tuple[int, int]:
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a 2-D shape, got {value!r}.") from exc
    if len(items) != 2:
        raise ValueError(f"{name} must be a 2-D shape, got {value!r}.")
    return int(items[0]), int(items[1])


def build_sample_metadata(
    *,
    stem: str,
    pt_data: bytes,
    text_data: bytes,
    preview_data: bytes | None,
    large_shape: Tuple[int, int],
    preview_shape: Tuple[int, int] | None,
    target_h: int,
    target_w: int,
    source_type: str,
    source_id: str,
    source: dict,
) -> bytes:
    large_shape = _coerce_shape2("large_shape", large_shape)
    if large_shape[0] <= 0 or large_shape[1] <= 0:
        raise ValueError(f"large_shape must be positive, got {large_shape}.")
    if int(target_h) != large_shape[0] * 32 or int(target_w) != large_shape[1] * 32:
        raise ValueError(
            f"target_h/target_w must match large_shape*32, got target={target_h}x{target_w} "
            f"large_shape={large_shape}."
        )
    if preview_data is None and preview_shape is not None:
        raise ValueError("preview_shape must be None when preview_data is absent.")
    if preview_data is not None:
        if preview_shape is None:
            raise ValueError("preview_shape is required when preview_data is present.")
        preview_shape = _coerce_shape2("preview_shape", preview_shape)
        expected_preview = stock_preview_shape_from_large_grid(*large_shape)
        if expected_preview[0] <= 0 or expected_preview[1] <= 0:
            raise ValueError(f"stock preview grid must be non-empty, got {expected_preview}.")
        if preview_shape != expected_preview:
            raise ValueError(
                f"preview_shape must match stock preview grid {expected_preview} "
                f"for large_shape={large_shape}, got {preview_shape}."
            )
    source_type = str(source_type)
    if source_type not in {"image", "glm_prior"}:
        raise ValueError(f"source_type must be 'image' or 'glm_prior', got {source_type!r}.")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("source_id must be a non-empty string.")
    if not isinstance(source, dict):
        raise ValueError("source must be a dict.")
    if source_type == "image":
        image_sha = source.get("image_sha256")
        if not is_sha256_hex(image_sha):
            raise ValueError("image source metadata requires a SHA-256 hex image_sha256.")
        if source_id != image_sha:
            raise ValueError("image source_id must equal image_sha256.")
    if source_type == "glm_prior":
        prompt_sha = source.get("prompt_sha256")
        seed = source.get("seed")
        trajectory_sha = source.get("trajectory_sha256")
        source_large_tokens = source.get("large_tokens")
        source_preview_tokens = source.get("preview_tokens")
        source_preview_offset = source.get("preview_offset")
        text_sha = sha256_bytes(text_data)
        if prompt_sha != text_sha:
            raise ValueError("glm_prior prompt_sha256 must match text_sha256.")
        if not isinstance(seed, int):
            raise ValueError("glm_prior seed must be an integer.")
        if source_id != f"{prompt_sha}:{seed}":
            raise ValueError("glm_prior source_id must equal prompt_sha256:seed.")
        if not is_sha256_hex(trajectory_sha):
            raise ValueError("glm_prior trajectory_sha256 must be a SHA-256 hex string.")
        if source_large_tokens != large_shape[0] * large_shape[1]:
            raise ValueError("glm_prior large_tokens must match large_shape area.")
        if not isinstance(source_preview_tokens, int) or source_preview_tokens < 0:
            raise ValueError("glm_prior preview_tokens must be a non-negative integer.")
        if not isinstance(source_preview_offset, int) or source_preview_offset < 0:
            raise ValueError("glm_prior preview_offset must be a non-negative integer.")
        if source_preview_offset != source_preview_tokens:
            raise ValueError("glm_prior preview_offset must equal preview_tokens.")
        if preview_shape is not None and source_preview_tokens != preview_shape[0] * preview_shape[1]:
            raise ValueError("glm_prior preview_tokens must match preview_shape area when preview payload is written.")
    payload = {
        "schema": PRETOK_METADATA_SCHEMA,
        "stem": str(stem),
        "source_type": source_type,
        "source_id": str(source_id),
        "source": dict(source),
        "target_h": int(target_h),
        "target_w": int(target_w),
        "large_shape": [large_shape[0], large_shape[1]],
        "preview_shape": (
            [preview_shape[0], preview_shape[1]]
            if preview_shape is not None
            else None
        ),
        "preview_conditioned": preview_data is not None,
        "large_pt_sha256": sha256_bytes(pt_data),
        "preview_pt_sha256": sha256_bytes(preview_data) if preview_data is not None else None,
        "text_sha256": sha256_bytes(text_data),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _member_kind(name: str) -> Tuple[str, str] | None:
    stem, ext = osp.splitext(name)
    ext = ext.lower()
    if name.endswith(".preview.pt"):
        return name[: -len(".preview.pt")], "preview"
    if name.endswith(".meta.json"):
        return name[: -len(".meta.json")], "meta"
    if ext == ".pt":
        return stem, "pt"
    if ext == ".txt":
        return stem, "text"
    return None


def summarize_pretok_pairs(shard_path: str) -> Tuple[int, int]:
    """Count complete samples and complete samples with ``.preview.pt`` members."""
    complete, with_preview, _with_meta = summarize_pretok_members(shard_path)
    return complete, with_preview


def summarize_pretok_members(shard_path: str) -> Tuple[int, int, int]:
    """Count complete samples plus preview/meta coverage without loading tensors."""
    with tarfile.open(shard_path, "r:*") as tf:
        stems: Dict[str, set[str]] = {}
        for member in tf:
            if not member.isfile():
                continue
            name = osp.basename(member.name)
            parsed = _member_kind(name)
            if parsed is None:
                continue
            stem, key = parsed
            parts = stems.setdefault(stem, set())
            if key in parts:
                raise ValueError(f"{shard_path}:{stem} has duplicate {key} members")
            parts.add(key)
    complete = sum(1 for parts in stems.values() if {"pt", "text"}.issubset(parts))
    if complete == 0:
        raise ValueError(f"{shard_path} contains zero complete .pt/.txt sample pairs.")
    incomplete = {
        stem: parts for stem, parts in stems.items()
        if parts and not {"pt", "text"}.issubset(parts)
    }
    if incomplete:
        details = ", ".join(
            f"{stem}:{'/'.join(sorted(parts))}"
            for stem, parts in sorted(incomplete.items())[:8]
        )
        raise ValueError(f"{shard_path} has unpaired .pt/.txt members: {details}")
    with_preview = sum(
        1 for parts in stems.values()
        if {"pt", "text"}.issubset(parts) and "preview" in parts
    )
    with_meta = sum(
        1 for parts in stems.values()
        if {"pt", "text"}.issubset(parts) and "meta" in parts
    )
    return complete, with_preview, with_meta


def count_pretok_pairs(
    shard_path: str,
    require_preview: bool = False,
    require_metadata: bool = False,
) -> int:
    """Count complete ``.pt``/``.txt`` pairs in one tar shard without loading tensors."""
    complete, with_preview, with_meta = summarize_pretok_members(shard_path)
    if require_preview:
        missing = complete - with_preview
        if missing:
            raise ValueError(
                f"{shard_path} has {missing}/{complete} samples without .preview.pt members."
            )
    if require_metadata:
        missing_meta = complete - with_meta
        if missing_meta:
            raise ValueError(
                f"{shard_path} has {missing_meta}/{complete} samples without .meta.json members."
            )
    return complete


def validate_sample_metadata(
    *,
    shard: str,
    stem: str,
    entry: Dict[str, bytes],
    tokens: torch.Tensor,
    preview_tokens: torch.Tensor | None,
) -> dict:
    if "meta" not in entry:
        raise ValueError(f"{shard}:{stem} missing .meta.json metadata member.")
    try:
        meta = json.loads(entry["meta"].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{shard}:{stem} metadata is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(meta, dict):
        raise ValueError(f"{shard}:{stem} metadata must be a JSON object.")
    if meta.get("schema") != PRETOK_METADATA_SCHEMA:
        raise ValueError(
            f"{shard}:{stem} metadata schema mismatch: got {meta.get('schema')!r}, "
            f"expected {PRETOK_METADATA_SCHEMA!r}."
        )
    if meta.get("stem") != stem:
        raise ValueError(f"{shard}:{stem} metadata stem mismatch: {meta.get('stem')!r}.")
    for key, member_key in (
        ("large_pt_sha256", "pt"),
        ("text_sha256", "text"),
    ):
        if meta.get(key) != sha256_bytes(entry[member_key]):
            raise ValueError(f"{shard}:{stem} metadata {key} mismatch.")
    has_preview = preview_tokens is not None
    if bool(meta.get("preview_conditioned")) != has_preview:
        raise ValueError(f"{shard}:{stem} metadata preview_conditioned mismatch.")
    if has_preview:
        if meta.get("preview_pt_sha256") != sha256_bytes(entry["preview"]):
            raise ValueError(f"{shard}:{stem} metadata preview_pt_sha256 mismatch.")
    elif meta.get("preview_pt_sha256") is not None:
        raise ValueError(f"{shard}:{stem} metadata preview_pt_sha256 set without preview member.")
    if list(meta.get("large_shape", [])) != [int(tokens.size(0)), int(tokens.size(1))]:
        raise ValueError(f"{shard}:{stem} metadata large_shape mismatch.")
    if meta.get("target_h") != int(tokens.size(0)) * 32 or meta.get("target_w") != int(tokens.size(1)) * 32:
        raise ValueError(f"{shard}:{stem} metadata target_h/target_w mismatch.")
    if has_preview:
        if preview_tokens.ndim != 2:
            raise ValueError(f"{shard}:{stem} metadata requires a 2-D preview token grid.")
        preview_shape = [int(preview_tokens.size(0)), int(preview_tokens.size(1))]
        if list(meta.get("preview_shape", [])) != preview_shape:
            raise ValueError(f"{shard}:{stem} metadata preview_shape mismatch.")
        expected_preview = list(stock_preview_shape_from_large_grid(int(tokens.size(0)), int(tokens.size(1))))
        if expected_preview[0] <= 0 or expected_preview[1] <= 0:
            raise ValueError(f"{shard}:{stem} metadata stock preview grid must be non-empty.")
        if preview_shape != expected_preview:
            raise ValueError(
                f"{shard}:{stem} metadata preview_shape does not match stock preview grid "
                f"{tuple(expected_preview)} for large grid {tuple(tokens.shape)}."
            )
    elif meta.get("preview_shape") is not None:
        raise ValueError(f"{shard}:{stem} metadata preview_shape set without preview member.")
    source_type = meta.get("source_type")
    source = meta.get("source")
    if source_type not in {"image", "glm_prior"}:
        raise ValueError(f"{shard}:{stem} metadata source_type is invalid: {source_type!r}.")
    if not isinstance(meta.get("source_id"), str) or not meta["source_id"]:
        raise ValueError(f"{shard}:{stem} metadata source_id must be a non-empty string.")
    if not isinstance(source, dict):
        raise ValueError(f"{shard}:{stem} metadata source must be a JSON object.")
    if source_type == "image":
        image_sha = source.get("image_sha256")
        if not is_sha256_hex(image_sha):
            raise ValueError(f"{shard}:{stem} image metadata requires a SHA-256 hex image_sha256.")
        if meta["source_id"] != image_sha:
            raise ValueError(f"{shard}:{stem} image metadata source_id must equal image_sha256.")
    if source_type == "glm_prior":
        for key in ("prompt_sha256", "seed", "trajectory_sha256"):
            if key not in source:
                raise ValueError(f"{shard}:{stem} glm_prior metadata missing {key}.")
        prompt_sha = source.get("prompt_sha256")
        seed = source.get("seed")
        trajectory_sha = source.get("trajectory_sha256")
        source_large_tokens = source.get("large_tokens")
        source_preview_tokens = source.get("preview_tokens")
        source_preview_offset = source.get("preview_offset")
        if prompt_sha != meta.get("text_sha256"):
            raise ValueError(f"{shard}:{stem} glm_prior metadata prompt_sha256 must match text_sha256.")
        if not isinstance(seed, int):
            raise ValueError(f"{shard}:{stem} glm_prior metadata seed must be an integer.")
        if meta["source_id"] != f"{prompt_sha}:{seed}":
            raise ValueError(f"{shard}:{stem} glm_prior metadata source_id must equal prompt_sha256:seed.")
        if not is_sha256_hex(trajectory_sha):
            raise ValueError(f"{shard}:{stem} glm_prior metadata trajectory_sha256 must be a SHA-256 hex string.")
        expected_large_tokens = int(tokens.numel())
        expected_preview_tokens = int(preview_tokens.numel()) if preview_tokens is not None else 0
        if source_large_tokens != expected_large_tokens:
            raise ValueError(f"{shard}:{stem} glm_prior metadata large_tokens must match token count.")
        if not isinstance(source_preview_tokens, int) or source_preview_tokens < 0:
            raise ValueError(f"{shard}:{stem} glm_prior metadata preview_tokens must be a non-negative integer.")
        if not isinstance(source_preview_offset, int) or source_preview_offset < 0:
            raise ValueError(f"{shard}:{stem} glm_prior metadata preview_offset must be a non-negative integer.")
        if source_preview_offset != source_preview_tokens:
            raise ValueError(f"{shard}:{stem} glm_prior metadata preview_offset must equal preview_tokens.")
        if preview_tokens is not None and source_preview_tokens != expected_preview_tokens:
            raise ValueError(f"{shard}:{stem} glm_prior metadata preview_tokens must match preview token count.")
    return meta


class PretokGlmShardDataset(torch.utils.data.IterableDataset):
    """Iterable dataset over pretokenized GLM-Image tar shards.

    Shards are split deterministically across distributed ranks and dataloader
    workers so every ``(rank, worker)`` sees a disjoint subset. Shuffling is
    per-epoch (seeded) at the shard granularity.
    """

    def __init__(
        self,
        shard_paths: Iterable[str],
        rank: int = 0,
        world_size: int = 1,
        seed: int = 1234,
        shuffle: bool = True,
        shuffle_buffer_size: int = 0,
        expected_shape: Optional[Tuple[int, int]] = None,
        codebook_size: Optional[int] = None,
        require_metadata: bool = False,
    ) -> None:
        super().__init__()
        self.shard_paths = list(shard_paths)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.expected_shape = tuple(expected_shape) if expected_shape is not None else None
        self.codebook_size = int(codebook_size) if codebook_size is not None else None
        self.require_metadata = bool(require_metadata)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _iter_shards(self) -> List[str]:
        shards = list(self.shard_paths)
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(shards)
        if self.world_size > 1:
            shards = shards[self.rank :: self.world_size]
        worker = torch.utils.data.get_worker_info()
        if worker is not None and worker.num_workers > 1:
            shards = shards[worker.id :: worker.num_workers]
        return shards

    def __iter__(self):
        base_iter = self._iter_samples()
        if self.shuffle_buffer_size <= 1:
            yield from base_iter
            return
        rng = random.Random(self.seed + 1000003 * self.epoch + 17 * self.rank)
        buf = []
        for sample in base_iter:
            buf.append(sample)
            if len(buf) >= self.shuffle_buffer_size:
                idx = rng.randrange(len(buf))
                yield buf.pop(idx)
        while buf:
            idx = rng.randrange(len(buf))
            yield buf.pop(idx)

    def _iter_samples(self):
        """Underlying deterministic shard/member iterator."""
        for shard in self._iter_shards():
            with tarfile.open(shard, "r:*") as tf:
                bucket: Dict[str, Dict[str, bytes]] = {}
                for member in tf:
                    if not member.isfile():
                        continue
                    name = osp.basename(member.name)
                    parsed = _member_kind(name)
                    if parsed is None:
                        continue
                    stem, key = parsed
                    fobj = tf.extractfile(member)
                    if fobj is None:
                        continue
                    entry = bucket.setdefault(stem, {})
                    if key in entry:
                        label = ".pt" if key == "pt" else ".txt" if key == "text" else key
                        raise ValueError(f"{shard}:{stem} has duplicate {label} members")
                    entry[key] = fobj.read()

                complete_stems = [
                    stem for stem, entry in sorted(bucket.items())
                    if "pt" in entry and "text" in entry
                ]
                incomplete = {
                    stem: entry for stem, entry in sorted(bucket.items())
                    if "pt" not in entry or "text" not in entry
                }
                if incomplete:
                    details = ", ".join(
                        f"{stem}:{'/'.join(sorted(parts))}"
                        for stem, parts in list(incomplete.items())[:8]
                    )
                    raise ValueError(
                        f"{shard} has unpaired .pt/.txt members left after reading: {details}"
                    )
                if not complete_stems:
                    raise ValueError(f"{shard} contains zero complete .pt/.txt sample pairs.")

                for stem in complete_stems:
                    entry = bucket[stem]
                    tokens = torch.load(io.BytesIO(entry["pt"]), map_location="cpu", weights_only=True)
                    if not torch.is_tensor(tokens):
                        tokens = torch.as_tensor(tokens)
                    tokens = tokens.long()
                    if tokens.ndim != 2:
                        raise ValueError(f"{shard}:{stem} expected a 2-D token grid, got {tuple(tokens.shape)}")
                    if self.expected_shape is not None and tuple(tokens.shape) != self.expected_shape:
                        raise ValueError(
                            f"{shard}:{stem} expected token grid {self.expected_shape}, "
                            f"got {tuple(tokens.shape)}"
                        )
                    if self.codebook_size is not None and tokens.numel() > 0:
                        min_id = int(tokens.min().item())
                        max_id = int(tokens.max().item())
                        if min_id < 0 or max_id >= self.codebook_size:
                            raise ValueError(
                                f"{shard}:{stem} token ids out of range [0, {self.codebook_size}): "
                                f"min={min_id} max={max_id}"
                            )
                    try:
                        text = entry["text"].decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise ValueError(f"{shard}:{stem} caption is not valid UTF-8: {exc}") from exc
                    if not text.strip():
                        raise ValueError(f"{shard}:{stem} caption is empty.")
                    preview_tokens = None
                    if "preview" in entry:
                        preview_tokens = torch.load(
                            io.BytesIO(entry["preview"]), map_location="cpu", weights_only=True
                        )
                        if not torch.is_tensor(preview_tokens):
                            preview_tokens = torch.as_tensor(preview_tokens)
                        preview_tokens = preview_tokens.long()
                        if preview_tokens.ndim not in (1, 2):
                            raise ValueError(
                                f"{shard}:{stem} expected preview ids to be 1-D or 2-D, "
                                f"got {tuple(preview_tokens.shape)}"
                            )
                        if self.codebook_size is not None and preview_tokens.numel() > 0:
                            min_id = int(preview_tokens.min().item())
                            max_id = int(preview_tokens.max().item())
                            if min_id < 0 or max_id >= self.codebook_size:
                                raise ValueError(
                                    f"{shard}:{stem} preview token ids out of range [0, {self.codebook_size}): "
                                    f"min={min_id} max={max_id}"
	                                )
                    sample = {"tokens": tokens, "text": text}
                    if preview_tokens is not None:
                        sample["preview_tokens"] = preview_tokens
                    if self.require_metadata or "meta" in entry:
                        sample["metadata"] = validate_sample_metadata(
                            shard=shard,
                            stem=stem,
                            entry=entry,
                            tokens=tokens,
                            preview_tokens=preview_tokens,
                        )
                    yield sample


def collate_pretok_glm(batch: List[dict]) -> dict:
    if not batch:
        raise ValueError("collate_pretok_glm received an empty batch.")
    tokens = [b["tokens"] for b in batch]
    texts = [b["text"] for b in batch]
    shapes = {tuple(t.shape) for t in tokens}
    if len(shapes) != 1:
        raise ValueError(
            f"All token grids in a batch must share the same shape; got {shapes}."
        )
    out = {"tokens": torch.stack(tokens, dim=0), "texts": texts}
    has_preview = ["preview_tokens" in b for b in batch]
    if any(has_preview):
        if not all(has_preview):
            raise ValueError("All samples in a batch must either include preview_tokens or omit them.")
        previews = [b["preview_tokens"] for b in batch]
        preview_shapes = {tuple(t.shape) for t in previews}
        if len(preview_shapes) != 1:
            raise ValueError(
                f"All preview token grids in a batch must share the same shape; got {preview_shapes}."
            )
        height, width = next(iter(shapes))
        preview_h, preview_w = stock_preview_shape_from_large_grid(height, width)
        expected_preview = (preview_h, preview_w)
        for preview in previews:
            if preview.ndim == 1:
                if int(preview.numel()) != preview_h * preview_w:
                    raise ValueError(
                        f"Flattened preview token length must equal {preview_h * preview_w} "
                        f"for large grid {(height, width)}, got {int(preview.numel())}."
                    )
            elif preview.ndim == 2:
                if tuple(preview.shape) != expected_preview:
                    raise ValueError(
                        f"Preview token grid must have shape {expected_preview} for "
                        f"large grid {(height, width)}, got {tuple(preview.shape)}."
                    )
            else:
                raise ValueError(f"preview_tokens must be 1-D or 2-D per sample, got {tuple(preview.shape)}.")
        out["preview_tokens"] = torch.stack(previews, dim=0)
    return out


__all__ = [
    "PretokGlmShardDataset",
    "collate_pretok_glm",
    "count_pretok_pairs",
    "build_sample_metadata",
    "is_sha256_hex",
    "sha256_bytes",
    "summarize_pretok_pairs",
    "summarize_pretok_members",
    "tensor_payload_sha256",
    "validate_sample_metadata",
]
