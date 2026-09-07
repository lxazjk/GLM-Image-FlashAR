<div align="center">
<h1>⚡ FlashAR</h1>

**Efficient Post-Training Acceleration for Autoregressive Image Generation**

[arXiv](https://arxiv.org/abs/2605.09430) | [Project Page](https://lxazjk.github.io/FlashAR/)

**GLM-Image implementation**
</div>

FlashAR adds a vertical prediction branch and a learnable fusion gate to a pretrained autoregressive image generator, then decodes visual tokens along anti-diagonals. An `H × W` grid takes `H + W − 1` decoding steps instead of `H × W`.

This repository applies FlashAR to GLM-Image's large-grid AR prior. At 1024 × 1024 resolution, the 32 × 32 token grid takes **63 steps instead of 1024**. The original preview-grid generation, token upsampling, DiT and VAE decoding are retained. This step count covers the large-grid prior only; it is not an end-to-end speedup measurement.

## Installation

Use Python 3.11 and a CUDA GPU with BF16 support. Run the commands below from the repository root.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch==2.9.1 torchvision==0.24.1
pip install -r requirements.txt
```

Choose a PyTorch wheel compatible with your CUDA environment. `requirements.txt` pins the Transformers and Diffusers commits used for this implementation.

## Models

Download the base model, including its processor and decoding components:

```bash
hf download zai-org/GLM-Image-Decoder --local-dir ./models/GLM-Image-Decoder
```

Generation also requires a **GLM-Image FlashAR checkpoint** from training below. Model weights and training data are not included in this repository. Emu3.5 FlashAR checkpoints are not compatible.

## Quick Start

```bash
CUDA_VISIBLE_DEVICES=0 python generate_glm_flashar.py \
  --model_path ./models/GLM-Image-Decoder \
  --ckpt_path ./outputs/glm_flashar/flashar_full_step12000.pt \
  --prompt "a red car parked next to a blue mailbox" \
  --out_path ./outputs/sample.png
```

Defaults: 1024 × 1024, 50 diffusion steps, guidance scale 5.0, seed 6666. Override them with `--height`, `--width`, `--num_inference_steps`, `--guidance_scale` and `--seed`.

The pipeline currently supports text-to-image generation with eager attention. Diagonal decoding uses recomputation; the experimental KV-cache path is disabled because it does not yet match the training forward pass.

## Training

### 1. Pretokenize

Prepare a JSONL manifest (a JSON list also works). Image paths are relative to `--image_root`:

```json
{"image": "000001.png", "text": "A red panda reading a book beside a window."}
{"image": "animals/000002.jpg", "text": "Two cranes flying over a lake."}
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m glmflashar.data.pretokenize_glm \
  --model_path ./models/GLM-Image-Decoder \
  --json_path ./data/train.jsonl \
  --image_root ./data/images \
  --output_dir ./data/pretok \
  --split train \
  --height 1024 --width 1024 \
  --write_preview_tokens
```

Shards are written to `data/pretok/train_partfirst/`. Each sample contains large-grid tokens, preview tokens, a caption and metadata. Keep `--write_preview_tokens` enabled for the default training configuration.

### 2. Train

```bash
CUDA_VISIBLE_DEVICES=0 python train_glm_flashar.py \
  --config_json configs/train_glm_flashar.default.json
```

The default config uses the paths above, Adafactor, 12,000 steps and a 1,000-step vertical-branch warmup. After warmup, it also trains the backbone. The FlashAR learning rate is `5e-5`; the backbone learning rate is `5e-6`.

Edit the config or override individual fields on the command line:

```bash
python train_glm_flashar.py \
  --config_json configs/train_glm_flashar.default.json \
  --model_path /path/to/GLM-Image-Decoder \
  --pretok_glob '/path/to/pretok/train_partfirst/*.tar' \
  --save_dir ./outputs/my_run \
  --max_steps 1000
```

For multiple GPUs, replace `python` with `torchrun --standalone --nproc_per_node=8`.

Checkpoints are saved under `save_dir`:

- `flashar_full_step*.pt`: FlashAR and trained backbone weights for generation.
- `flashar_resume_step*.pt`: full training state; pass with `--resume_ckpt` to resume a single-process run.
- `flashar_heads_only_step*.pt`: added parameters only; insufficient for generation after backbone training.

Use `--init_ckpt /path/to/flashar_full_step12000.pt` to start a new training run from saved weights. `--init_ckpt` and `--resume_ckpt` are mutually exclusive.

## Repository Layout

```text
glmflashar/
├── data/                   Pretokenization and dataset loading
├── inference/              GLM-Image FlashAR pipeline
├── model/                  FlashAR model and backbone patch
└── utils/                  Text and preview-grid helpers
configs/                    Default training configuration
generate_glm_flashar.py      Single-image generation
train_glm_flashar.py         Training entry point
requirements.txt            Dependencies
```

## Citation

```bibtex
@article{zhou2026flashar,
  title={FlashAR: Efficient Post-Training Acceleration for Autoregressive Image Generation},
  author={Zhou, Junkang and He, Yefei and Chen, Feng and Wang, Weijie and Zhuang, Bohan},
  journal={arXiv preprint arXiv:2605.09430},
  year={2026}
}
```

Built on GLM-Image and the FlashAR method.
