# -*- coding: utf-8 -*-
"""Shared GLM-Image preview-grid helpers."""

from __future__ import annotations

import math


def stock_preview_shape_from_large_grid(height: int, width: int) -> tuple[int, int]:
    """Return stock GLM preview grid shape for a d32 large-token grid."""
    height = int(height)
    width = int(width)
    if height <= 0 or width <= 0:
        raise ValueError(f"large grid height/width must be positive, got {height}x{width}.")
    ratio = float(height) / float(width)
    preview_h = int(math.sqrt(ratio) * 16)
    preview_w = int(math.sqrt(1.0 / ratio) * 16)
    return preview_h, preview_w


def stock_preview_shape_from_pixels(height: int, width: int, downsample: int = 32) -> tuple[int, int]:
    """Return stock GLM preview grid shape for a target image size in pixels."""
    height = int(height)
    width = int(width)
    downsample = int(downsample)
    if downsample <= 0:
        raise ValueError(f"downsample must be positive, got {downsample}.")
    if height <= 0 or width <= 0:
        raise ValueError(f"target height/width must be positive, got {height}x{width}.")
    if height % downsample != 0 or width % downsample != 0:
        raise ValueError(
            f"target height/width must be multiples of {downsample}, got {height}x{width}."
        )
    token_h = height // downsample
    token_w = width // downsample
    if token_h <= 0 or token_w <= 0:
        raise RuntimeError(f"invalid large grid shape from target size: {token_h}x{token_w}")
    preview_h, preview_w = stock_preview_shape_from_large_grid(token_h, token_w)
    if preview_h <= 0 or preview_w <= 0:
        raise RuntimeError(f"invalid preview grid shape: {preview_h}x{preview_w}")
    return preview_h, preview_w


__all__ = ["stock_preview_shape_from_large_grid", "stock_preview_shape_from_pixels"]
