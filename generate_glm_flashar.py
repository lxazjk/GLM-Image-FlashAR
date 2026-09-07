"""Generate a single image with a GLM-Image FlashAR checkpoint."""

import argparse
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default="./models/GLM-Image-Decoder")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out_path", default="./outputs/sample.png")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=6666)
    args = parser.parse_args()
    from glmflashar.utils.preview import stock_preview_shape_from_pixels

    stock_preview_shape_from_pixels(args.height, args.width)
    if not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if args.num_inference_steps <= 0:
        parser.error("--num_inference_steps must be positive")
    if not math.isfinite(args.guidance_scale):
        parser.error("--guidance_scale must be finite")
    if not Path(args.ckpt_path).is_file():
        parser.error(f"checkpoint not found: {args.ckpt_path}")
    return args


def main():
    args = parse_args()
    import torch
    from glmflashar.inference.flashar_pipeline import GlmImageFlashARPipeline

    pipe = GlmImageFlashARPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda")
    pipe.attach_flashar(ckpt_path=args.ckpt_path)
    with torch.inference_mode():
        image = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=1,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        ).images[0]
    output = Path(args.out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"Saved {output} (large-grid decoding steps: {pipe.last_ar_num_steps})")


if __name__ == "__main__":
    main()
