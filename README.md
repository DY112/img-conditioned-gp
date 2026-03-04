# Image-Conditioned Generative Photography (img2img Extension)

This extension adds **image input support** to [Generative Photography](https://github.com/pandayuanyu/generative-photography) via SDEdit-style latent initialization (Approach 2: img2img).

Instead of generating a scene from a text prompt, you can now provide an **input image** and apply camera parameter variations (bokeh, focal length, shutter speed, color temperature) while preserving the original scene content.

## What Changed

### Modified file: `genphoto/pipelines/pipeline_animation.py`

Three new methods added to `GenPhotoPipeline`:

| Method | Purpose |
|---|---|
| `encode_image()` | Encodes an input image into VAE latent space |
| `get_img2img_timesteps()` | Computes truncated timestep schedule based on `strength` |
| `prepare_img2img_latents()` | Replicates image latent across 5 frames and adds noise |

The `__call__()` method now accepts two new optional parameters:
- `strength` (float, default=1.0): Controls the denoising strength. 1.0 = pure text-to-image (original behavior, fully backward compatible). Lower values preserve more of the input image.
- `init_image_latents` (tensor, default=None): Pre-encoded image latents from `encode_image()`.

**Backward compatibility is fully preserved.** All existing inference scripts (`inference_bokehK.py`, etc.) and `app.py` work without any changes.

### New file: `inference_image.py`

Unified inference script supporting all 4 camera parameter types with image input.

## Installation

No additional dependencies are needed beyond the original `environment.yaml`.

## How to Apply

```bash
# Option A: Replace the file directly
cp genphoto/pipelines/pipeline_animation.py /your/repo/genphoto/pipelines/
cp inference_image.py /your/repo/

# Option B: Apply the patch
cd /your/repo
git apply pipeline_animation.patch
cp inference_image.py .
```

## Usage

### Image-conditioned inference (img2img)

```bash
# Bokeh effect from an input image
python inference_image.py \
    --image ./my_photo.jpg \
    --camera_type bokeh \
    --camera_values "[2.44, 8.3, 10.1, 17.2, 24.0]" \
    --config configs/inference_genphoto/adv3_256_384_genphoto_relora_bokehK.yaml \
    --strength 0.6

# Focal length from an input image
python inference_image.py \
    --image ./my_photo.jpg \
    --camera_type focal_length \
    --camera_values "[25.0, 35.0, 45.0, 55.0, 65.0]" \
    --config configs/inference_genphoto/adv3_256_384_genphoto_relora_focal_length.yaml \
    --strength 0.5

# Shutter speed from an input image with a descriptive prompt
python inference_image.py \
    --image ./my_photo.jpg \
    --camera_type shutter_speed \
    --camera_values "[0.1, 0.3, 0.52, 0.7, 0.8]" \
    --config configs/inference_genphoto/adv3_256_384_genphoto_relora_shutter_speed.yaml \
    --strength 0.7 \
    --prompt "A modern bathroom with a mirror and soft lighting."

# Color temperature from an input image
python inference_image.py \
    --image ./my_photo.jpg \
    --camera_type color_temperature \
    --camera_values "[5455.0, 5155.0, 5555.0, 6555.0, 7555.0]" \
    --config configs/inference_genphoto/adv3_256_384_genphoto_relora_color_temperature.yaml \
    --strength 0.5
```

### Pure text-to-image (original behavior)

```bash
python inference_image.py \
    --camera_type bokeh \
    --camera_values "[2.44, 8.3, 10.1, 17.2, 24.0]" \
    --config configs/inference_genphoto/adv3_256_384_genphoto_relora_bokehK.yaml \
    --prompt "A young boy wearing an orange jacket is standing on a crosswalk." \
    --strength 1.0
```

## Key Parameters

### `--strength` (most important)

Controls the balance between input image fidelity and camera effect intensity.

| Strength | Behavior |
|---|---|
| 0.3 | Very faithful to input image, subtle camera effects |
| 0.5 | Balanced — good starting point |
| 0.6 | **Recommended default** — noticeable camera effects with reasonable scene preservation |
| 0.7 | Strong camera effects, some scene details may shift |
| 0.8 | Very strong camera effects, scene structure preserved but details change |
| 1.0 | Pure text-to-image (ignores input image entirely) |

### `--prompt` (optional in img2img mode)

When using `--image`, the prompt is optional. If omitted, a generic prompt `"a high quality photograph"` is used — the latent initialization from the image dominates the scene structure. However, providing a descriptive prompt that matches your image will improve quality, since the cross-attention mechanism still uses it during denoising.

### Other parameters

| Parameter | Default | Description |
|---|---|---|
| `--num_inference_steps` | 25 | Total denoising steps |
| `--guidance_scale` | 8.0 | Classifier-free guidance scale |
| `--height` | 256 | Output height |
| `--width` | 384 | Output width |
| `--seed` | 42 | Random seed |
| `--negative_prompt` | None | Negative prompt for CFG |
| `--output_dir` | auto | Output directory |

## Output

The script produces:
- `{camera_type}_img2img.gif` — Animated GIF of all 5 parameter variants
- `{camera_type}_frames/` — Individual PNG frames for each parameter value
- `input_resized.png` — The input image resized to model resolution (for comparison)

## How It Works

1. **Encode** the input image through the SD1.5 VAE encoder → clean latent `z₀`
2. **Replicate** across 5 frames: `z₀ → [z₀, z₀, z₀, z₀, z₀]`
3. **Add noise** to an intermediate timestep determined by `strength`
4. **Denoise** from this partially-noised state (skipping early steps), with:
   - Camera adaptor injecting ISP parameter variations via temporal/self-attention
   - Text prompt providing scene semantics via cross-attention
5. **Decode** each frame back to pixel space via VAE decoder

The key insight is that the camera parameter pathway (Camera Adaptor) and the scene content pathway (CLIP text encoder → cross-attention) are **architecturally independent**. The img2img latent initialization replaces the text-based scene conditioning with direct visual information, while the camera adaptor continues to function exactly as designed.
