#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pretokenize images -> GLM-Image VQ token grids for FlashAR training.

GLM-Image's AR prior operates on discrete VQ tokens. The encode path (mirrors
the i2i branch of GlmImagePipeline) is:

    pixel_values, image_grid_thw = processor(images=[img], target_h=H, target_w=W)
    feats = model.get_image_features(pixel_values, image_grid_thw)   # vision encoder
    image_embeds = torch.cat(feats.pooler_output, dim=0)
    tokens = model.get_image_tokens(image_embeds, image_grid_thw)  # vqmodel.encode -> indices

The returned `tokens` are flat discrete codebook indices; reshape to the d32
large grid from image_grid_thw. For target 1024px this is (1,32,32), matching
the FlashAR AR-prior training target directly.

Writes tar shards of {stem}.pt (LongTensor grid) + {stem}.txt (caption).
With --write_preview_tokens
it also writes {stem}.preview.pt for preview-conditioned FlashAR training.
"""
import argparse
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

import torch
from PIL import Image
from glmflashar.data.pretokenized_glm import build_sample_metadata, sha256_bytes
from glmflashar.utils.preview import stock_preview_shape_from_pixels


class LazyTarShardWriter:
    """Write tar shards only after the first successful sample.

    This avoids creating empty shards when every sample is skipped, or when the
    number of successful samples is exactly divisible by ``shard_size``.
    """

    def __init__(self, output_dir: Path, split: str, tag: str, shard_size: int) -> None:
        self.output_dir = Path(output_dir)
        self.split = str(split)
        self.tag = str(tag)
        self.shard_size = max(1, int(shard_size))
        self.shard_idx = 0
        self.in_shard = 0
        self.tar = None
        self.path = None
        self.tmp_path = None
        self.n_shards = 0
        self.written_paths = []

    def _open_shard(self):
        path = self.output_dir / f"{self.split}-{self.tag}-{self.shard_idx:05d}.tar"
        tmp_path = path.with_name(f"{path.name}.tmp")
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing shard: {path}")
        if tmp_path.exists():
            tmp_path.unlink()
        self.path = path
        self.tmp_path = tmp_path
        self.tar = tarfile.open(tmp_path, "w")
        self.n_shards += 1

    def _close_shard(self) -> None:
        if self.tar is None:
            return
        self.tar.close()
        self.tar = None
        os.replace(self.tmp_path, self.path)
        self.written_paths.append(self.path)
        self.path = None
        self.tmp_path = None

    @staticmethod
    def _add(tar, name: str, data: bytes) -> None:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    def write_pair(
        self,
        stem: str,
        pt_data: bytes,
        text_data: bytes,
        preview_data: bytes | None = None,
        metadata_data: bytes | None = None,
    ) -> None:
        if self.tar is None:
            self._open_shard()
        self._add(self.tar, f"{stem}.pt", pt_data)
        if preview_data is not None:
            self._add(self.tar, f"{stem}.preview.pt", preview_data)
        if metadata_data is not None:
            self._add(self.tar, f"{stem}.meta.json", metadata_data)
        self._add(self.tar, f"{stem}.txt", text_data)
        self.in_shard += 1
        if self.in_shard >= self.shard_size:
            self._close_shard()
            self.shard_idx += 1
            self.in_shard = 0

    def close(self) -> None:
        self._close_shard()

    def cleanup(self) -> None:
        if self.tar is not None:
            self.tar.close()
            self.tar = None
        for path in [self.tmp_path, *self.written_paths]:
            if path is not None and Path(path).exists():
                Path(path).unlink()
        self.path = None
        self.tmp_path = None
        self.written_paths = []


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="./models/GLM-Image-Decoder")
    p.add_argument("--json_path", required=True, help="manifest: [{image, text}, ...] or jsonl")
    p.add_argument("--image_root", default="")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--split", default="text_to_image")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--codebook_size", type=int, default=16384)
    p.add_argument("--shard_size", type=int, default=2000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--shard_tag", default="")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--allow_existing_other_shards",
        action="store_true",
        help="Permit this shard_tag in a split directory that already contains other shard tags.",
    )
    p.add_argument("--allow_skip", action="store_true")
    p.add_argument("--max_skip_rate", type=float, default=0.0)
    p.add_argument("--allow_empty_text", action="store_true")
    p.add_argument(
        "--write_preview_tokens",
        action="store_true",
        help="Also VQ-encode same-image stock preview-grid tokens and write {stem}.preview.pt.",
    )
    return p.parse_args()


def load_manifest(path):
    items = []
    with open(path) as f:
        if path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        else:
            items = json.load(f)
    if not isinstance(items, list):
        raise ValueError(f"manifest must be a list or JSONL records, got {type(items).__name__}")
    return items


def select_manifest_items(items, start: int, end: int, limit: int):
    if limit > 0:
        items = items[:limit]
    if start < 0:
        raise ValueError(f"--start must be >= 0, got {start}")
    resolved_end = len(items) if end < 0 else min(int(end), len(items))
    selected = items[int(start):resolved_end]
    if not selected:
        raise ValueError(
            f"empty pretokenize range: start={start} end={end} "
            f"resolved_end={resolved_end} manifest_items={len(items)}"
        )
    return selected


def validate_manifest_items(items, allow_empty_text: bool = False) -> None:
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"manifest item {idx} must be a JSON object, got {type(item).__name__}")
        img_path = item.get("image") or item.get("image_path")
        if not isinstance(img_path, str) or not img_path.strip():
            raise ValueError(f"manifest item {idx} missing non-empty image/image_path field")
        text = item.get("text") or item.get("caption") or item.get("input_prompt")
        if not allow_empty_text and (not isinstance(text, str) or not text.strip()):
            raise ValueError(
                f"manifest item {idx} missing non-empty text/caption/input_prompt field; "
                "pass --allow_empty_text only for intentional unconditional data."
            )


def resolve_image_path(image_root: str, image_path: str) -> Path:
    path = Path(str(image_path).strip()).expanduser()
    if image_root and not path.is_absolute():
        path = Path(image_root).expanduser() / path
    return path


def validate_image_paths(
    items,
    image_root: str,
    allow_skip: bool,
    max_skip_rate: float,
) -> None:
    missing = []
    for idx, item in enumerate(items):
        img_path = item.get("image") or item.get("image_path")
        path = resolve_image_path(image_root, img_path)
        if not path.is_file():
            missing.append((idx, str(path)))
    if not missing:
        return
    missing_rate = len(missing) / max(1, len(items))
    if not allow_skip or missing_rate > float(max_skip_rate):
        preview = ", ".join(f"{idx}:{path}" for idx, path in missing[:4])
        raise ValueError(
            f"{len(missing)}/{len(items)} image paths are missing before pretokenization "
            f"({missing_rate:.2%}); refusing to load GLM weights. Missing: {preview}"
            f"{' ...' if len(missing) > 4 else ''}"
        )


def validate_pretokenize_args(args) -> None:
    if int(args.height) <= 0 or int(args.width) <= 0:
        raise ValueError("--height/--width must be positive.")
    if int(args.height) % 32 != 0 or int(args.width) % 32 != 0:
        raise ValueError(f"--height/--width must be multiples of 32, got {args.height}x{args.width}.")
    stock_preview_shape(args.height, args.width)
    if int(args.shard_size) <= 0:
        raise ValueError(f"--shard_size must be positive, got {args.shard_size}.")
    if int(args.codebook_size) <= 0:
        raise ValueError(f"--codebook_size must be positive, got {args.codebook_size}.")
    if int(args.limit) == 0:
        raise ValueError("--limit=0 selects no samples.")
    max_skip_rate = float(getattr(args, "max_skip_rate", 0.0))
    if max_skip_rate < 0 or max_skip_rate > 1:
        raise ValueError("--max_skip_rate must be in [0, 1].")


def validate_model_path(model_path: str) -> None:
    root = Path(model_path).expanduser()
    required = [
        ("GLM VLE config.json", root / "vision_language_encoder" / "config.json"),
        ("GLM VLE generation_config.json", root / "vision_language_encoder" / "generation_config.json"),
        ("GLM processor tokenizer.json", root / "processor" / "tokenizer.json"),
        ("GLM processor preprocessor_config.json", root / "processor" / "preprocessor_config.json"),
        ("GLM processor chat_template.jinja", root / "processor" / "chat_template.jinja"),
    ]
    for desc, path in required:
        if not path.is_file():
            raise ValueError(f"missing {desc} for --model_path={model_path}: {path}")


def _all_shard_files(outdir: Path) -> list[Path]:
    return (
        sorted(outdir.glob("*.tar"))
        + sorted(outdir.glob("*.tar.tmp"))
        + sorted(outdir.glob("*.tmp"))
    )


def prepare_output_shards(
    outdir: Path,
    split: str,
    tag: str,
    overwrite: bool,
    allow_existing_other_shards: bool = False,
) -> None:
    matching = sorted(outdir.glob(f"{split}-{tag}-*.tar")) + sorted(
        outdir.glob(f"{split}-{tag}-*.tar.tmp")
    )
    existing = matching if allow_existing_other_shards else _all_shard_files(outdir)
    if not existing:
        return
    if not overwrite:
        preview = ", ".join(str(path) for path in existing[:4])
        raise FileExistsError(
            f"existing pretokenize shards/tmp files in {outdir}; refusing to mix new data "
            f"with old shards. Pass --overwrite to remove them first, or "
            f"--allow_existing_other_shards for explicit multi-tag appends. Existing: {preview}"
        )


def remove_output_shards(
    outdir: Path,
    split: str,
    tag: str,
    allow_existing_other_shards: bool = False,
) -> None:
    existing = (
        sorted(outdir.glob(f"{split}-{tag}-*.tar"))
        + sorted(outdir.glob(f"{split}-{tag}-*.tar.tmp"))
        if allow_existing_other_shards
        else _all_shard_files(outdir)
    )
    for path in existing:
        path.unlink()


def validate_skip_counts(n_ok: int, n_total: int, allow_skip: bool, max_skip_rate: float) -> None:
    if n_ok <= 0:
        raise SystemExit("No samples were pretokenized successfully; check manifest/image paths.")
    skipped = int(n_total) - int(n_ok)
    if skipped <= 0:
        return
    skip_rate = skipped / max(1, int(n_total))
    if not allow_skip or skip_rate > float(max_skip_rate):
        raise SystemExit(
            f"pretokenization skipped {skipped}/{n_total} samples ({skip_rate:.2%}); "
            "refusing to write a silently truncated dataset. Fix the inputs or pass "
            "--allow_skip with an explicit --max_skip_rate."
        )


def stock_preview_shape(height: int, width: int) -> tuple[int, int]:
    return stock_preview_shape_from_pixels(height, width)


def encode_image_token_grid(model, processor, image: Image.Image, height: int, width: int, device: torch.device) -> torch.Tensor:
    image = image.resize((int(width), int(height)), Image.BICUBIC)
    inputs = processor(
        images=[image],
        target_h=height,
        target_w=width,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        feats = model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
        image_embeds = torch.cat(feats.pooler_output, dim=0)
        tokens = model.get_image_tokens(image_embeds, inputs["image_grid_thw"])
    grid_t, grid_h, grid_w = inputs["image_grid_thw"][0].tolist()
    expected_tokens = int(grid_t) * int(grid_h) * int(grid_w)
    if int(tokens.numel()) != expected_tokens:
        raise RuntimeError(
            f"token count mismatch: got {int(tokens.numel())}, "
            f"expected {expected_tokens} from image_grid_thw={inputs['image_grid_thw'][0].tolist()}"
        )
    return tokens.view(int(grid_h), int(grid_w)).to(torch.long).cpu()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    args = parse_args()
    validate_pretokenize_args(args)
    items = select_manifest_items(load_manifest(args.json_path), args.start, args.end, args.limit)
    validate_manifest_items(items, allow_empty_text=args.allow_empty_text)
    validate_image_paths(
        items,
        image_root=args.image_root,
        allow_skip=args.allow_skip,
        max_skip_rate=args.max_skip_rate,
    )
    validate_model_path(args.model_path)
    tag = args.shard_tag or f"{args.start}"
    outdir = Path(args.output_dir) / f"{args.split}_partfirst"
    outdir.mkdir(parents=True, exist_ok=True)
    prepare_output_shards(
        outdir,
        args.split,
        tag,
        overwrite=args.overwrite,
        allow_existing_other_shards=args.allow_existing_other_shards,
    )

    from transformers import AutoProcessor
    from transformers.models.glm_image import GlmImageForConditionalGeneration

    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.model_path, subfolder="processor", trust_remote_code=True)
    model = GlmImageForConditionalGeneration.from_pretrained(
        args.model_path, subfolder="vision_language_encoder",
        torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).to(device).eval()

    if args.overwrite:
        remove_output_shards(
            outdir,
            args.split,
            tag,
            allow_existing_other_shards=args.allow_existing_other_shards,
        )
    writer = LazyTarShardWriter(outdir, args.split, tag, args.shard_size)
    n_ok = 0
    try:
        for idx, it in enumerate(items):
            try:
                img_path = it.get("image") or it.get("image_path")
                text = it.get("text") or it.get("caption") or it.get("input_prompt") or ""
                img_path = resolve_image_path(args.image_root, img_path)
                image = Image.open(img_path).convert("RGB")
                grid = encode_image_token_grid(
                    model,
                    processor,
                    image,
                    height=args.height,
                    width=args.width,
                    device=device,
                )
                expected_grid = (int(args.height) // 32, int(args.width) // 32)
                if tuple(grid.shape) != expected_grid:
                    raise RuntimeError(
                        f"bad token grid shape for image {idx}: got {tuple(grid.shape)}, "
                        f"expected {expected_grid}"
                    )
                if grid.numel() > 0:
                    min_id = int(grid.min().item())
                    max_id = int(grid.max().item())
                    if min_id < 0 or max_id >= int(args.codebook_size):
                        raise RuntimeError(
                            f"token ids out of range [0, {args.codebook_size}) for "
                            f"image {idx}: min={min_id} max={max_id}"
                        )
                # NOTE: for target 1024px the processor yields image_grid_thw=(1,32,32),
                # i.e. get_image_tokens already returns the d32 AR large grid directly.
                # No downsample.

                global_idx = args.start + idx
                stem = f"{global_idx:08d}"
                buf = io.BytesIO()
                torch.save(grid, buf)
                pt_data = buf.getvalue()
                preview_data = None
                preview_shape = None
                if args.write_preview_tokens:
                    preview_h, preview_w = stock_preview_shape(args.height, args.width)
                    preview_grid = encode_image_token_grid(
                        model,
                        processor,
                        image,
                        height=preview_h * 32,
                        width=preview_w * 32,
                        device=device,
                    )
                    if tuple(preview_grid.shape) != (preview_h, preview_w):
                        raise RuntimeError(
                            f"bad preview grid shape for image {idx}: got {tuple(preview_grid.shape)}, "
                            f"expected {(preview_h, preview_w)}"
                        )
                    if preview_grid.numel() > 0:
                        min_id = int(preview_grid.min().item())
                        max_id = int(preview_grid.max().item())
                        if min_id < 0 or max_id >= int(args.codebook_size):
                            raise RuntimeError(
                                f"preview token ids out of range [0, {args.codebook_size}) for "
                                f"image {idx}: min={min_id} max={max_id}"
                            )
                    preview_buf = io.BytesIO()
                    torch.save(preview_grid, preview_buf)
                    preview_data = preview_buf.getvalue()
                    preview_shape = tuple(preview_grid.shape)
                text_data = text.encode("utf-8")
                image_sha = file_sha256(Path(img_path))
                metadata_data = build_sample_metadata(
                    stem=stem,
                    pt_data=pt_data,
                    text_data=text_data,
                    preview_data=preview_data,
                    large_shape=tuple(grid.shape),
                    preview_shape=preview_shape,
                    target_h=int(args.height),
                    target_w=int(args.width),
                    source_type="image",
                    source_id=image_sha,
                    source={
                        "image_path": str(img_path),
                        "image_sha256": image_sha,
                        "manifest_index": int(global_idx),
                    },
                )
            except Exception as e:
                print(f"[skip {idx}] {e}", flush=True)
                continue
            writer.write_pair(
                stem,
                pt_data,
                text_data,
                preview_data=preview_data,
                metadata_data=metadata_data,
            )
            n_ok += 1
            if idx % 100 == 0:
                print(f"[{global_idx}/{args.start + len(items)}] ok={n_ok} grid={tuple(grid.shape)}", flush=True)
        writer.close()
        validate_skip_counts(n_ok, len(items), args.allow_skip, args.max_skip_rate)
    except BaseException:
        writer.cleanup()
        raise
    print(f"DONE ok={n_ok} shards={writer.n_shards} -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
