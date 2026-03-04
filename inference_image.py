"""
Image-conditioned Generative Photography inference script.

Given an input image and camera parameters, generates ISP parameter variant
images while preserving the original scene content via img2img latent
initialization (Approach 2: SDEdit-style partial denoising).

Supports all 4 camera settings: bokeh, focal_length, shutter_speed, color_temperature.

Usage examples:
    # Bokeh rendering from image
    python inference_image.py \
        --image ./my_photo.jpg \
        --camera_type bokeh \
        --camera_values "[2.44, 8.3, 10.1, 17.2, 24.0]" \
        --config configs/inference_genphoto/adv3_256_384_genphoto_relora_bokehK.yaml \
        --strength 0.6

    # Focal length from image
    python inference_image.py \
        --image ./my_photo.jpg \
        --camera_type focal_length \
        --camera_values "[25.0, 35.0, 45.0, 55.0, 65.0]" \
        --config configs/inference_genphoto/adv3_256_384_genphoto_relora_focal_length.yaml \
        --strength 0.5

    # With custom prompt (overrides auto-generated description)
    python inference_image.py \
        --image ./my_photo.jpg \
        --camera_type shutter_speed \
        --camera_values "[0.1, 0.3, 0.52, 0.7, 0.8]" \
        --config configs/inference_genphoto/adv3_256_384_genphoto_relora_shutter_speed.yaml \
        --strength 0.7 \
        --prompt "A modern bathroom with a mirror and soft lighting."

    # Strength = 1.0 falls back to pure text-to-image (requires --prompt)
    python inference_image.py \
        --camera_type bokeh \
        --camera_values "[2.44, 8.3, 10.1, 17.2, 24.0]" \
        --config configs/inference_genphoto/adv3_256_384_genphoto_relora_bokehK.yaml \
        --prompt "A young boy wearing an orange jacket." \
        --strength 1.0
"""

import os
import torch
import logging
import argparse
import json
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler
from einops import rearrange
import torchvision.transforms as T
import torchvision

from genphoto.pipelines.pipeline_animation import GenPhotoPipeline
from genphoto.models.unet import UNet3DConditionModelCameraCond
from genphoto.models.camera_adaptor import CameraCameraEncoder, CameraAdaptor
from genphoto.utils.util import save_videos_grid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
#  Camera Embedding Builders (unified from the 4 original inference scripts)
# ============================================================================

def create_bokehK_embedding(bokehK_values, target_height, target_width):
    f = bokehK_values.shape[0]
    bokehK_embedding = torch.zeros((f, 3, target_height, target_width), dtype=bokehK_values.dtype)
    for i in range(f):
        K_value = bokehK_values[i].item()
        kernel_size = max(K_value, 1)
        sigma = K_value / 3.0
        ax = np.linspace(-(kernel_size / 2), kernel_size / 2, int(np.ceil(kernel_size)))
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel /= np.sum(kernel)
        scale = kernel[int(np.ceil(kernel_size) / 2), int(np.ceil(kernel_size) / 2)]
        bokehK_embedding[i] = scale
    return bokehK_embedding


def create_focal_length_embedding(focal_length_values, target_height, target_width,
                                   base_focal_length=24.0, sensor_height=24.0, sensor_width=36.0):
    device = 'cpu'
    focal_length_values = focal_length_values.to(device)
    f = focal_length_values.shape[0]
    sensor_width_t = torch.tensor(sensor_width, device=device)
    sensor_height_t = torch.tensor(sensor_height, device=device)
    base_focal_length_t = torch.tensor(base_focal_length, device=device)
    base_fov_x = 2.0 * torch.atan(sensor_width_t * 0.5 / base_focal_length_t)
    base_fov_y = 2.0 * torch.atan(sensor_height_t * 0.5 / base_focal_length_t)
    target_fov_x = 2.0 * torch.atan(sensor_width_t * 0.5 / focal_length_values)
    target_fov_y = 2.0 * torch.atan(sensor_height_t * 0.5 / focal_length_values)
    crop_ratio_xs = target_fov_x / base_fov_x
    crop_ratio_ys = target_fov_y / base_fov_y
    center_h, center_w = target_height // 2, target_width // 2
    focal_length_embedding = torch.zeros((f, 3, target_height, target_width), dtype=torch.float32)
    for i in range(f):
        crop_h = torch.round(crop_ratio_ys[i] * target_height).int().item()
        crop_w = torch.round(crop_ratio_xs[i] * target_width).int().item()
        crop_h = max(1, min(target_height, crop_h))
        crop_w = max(1, min(target_width, crop_w))
        focal_length_embedding[i, :,
            center_h - crop_h // 2: center_h + crop_h // 2,
            center_w - crop_w // 2: center_w + crop_w // 2] = 1.0
    return focal_length_embedding


def create_shutter_speed_embedding(shutter_speed_values, target_height, target_width, base_exposure=0.5):
    f = shutter_speed_values.shape[0]
    fwc = 32000
    scales = (shutter_speed_values / base_exposure) * (fwc / (fwc + 0.0001))
    scales = scales.unsqueeze(2).unsqueeze(3).expand(f, 3, target_height, target_width)
    return scales


def kelvin_to_rgb(kelvin):
    if torch.is_tensor(kelvin):
        kelvin = kelvin.cpu().item()
    temp = kelvin / 100.0
    if temp <= 66:
        red = 255
        green = 99.4708025861 * np.log(temp) - 161.1195681661 if temp > 0 else 0
        if temp <= 19:
            blue = 0
        else:
            blue = 138.5177312231 * np.log(temp - 10) - 305.0447927307
    elif 66 < temp <= 88:
        red = 0.5 * (255 + 329.698727446 * ((temp - 60) ** -0.19332047592))
        green = 0.5 * (288.1221695283 * ((temp - 60) ** -0.1155148492) +
                       (99.4708025861 * np.log(temp) - 161.1195681661 if temp > 0 else 0))
        blue = 0.5 * (138.5177312231 * np.log(temp - 10) - 305.0447927307 + 255)
    else:
        red = 329.698727446 * ((temp - 60) ** -0.19332047592)
        green = 288.1221695283 * ((temp - 60) ** -0.1155148492)
        blue = 255
    return np.array([red, green, blue], dtype=np.float32) / 255.0


def create_color_temperature_embedding(color_temperature_values, target_height, target_width,
                                        min_color_temperature=2000, max_color_temperature=10000):
    f = color_temperature_values.shape[0]
    rgb_factors = []
    for ct in color_temperature_values.squeeze():
        kelvin = min_color_temperature + (ct * (max_color_temperature - min_color_temperature))
        rgb = kelvin_to_rgb(kelvin)
        rgb_factors.append(rgb)
    rgb_factors = torch.tensor(rgb_factors).float()
    color_temperature_embedding = rgb_factors.unsqueeze(2).unsqueeze(3).expand(f, 3, target_height, target_width)
    return color_temperature_embedding


# ============================================================================
#  Unified Camera Embedding class
# ============================================================================

# Prompt templates per camera type (must match the originals exactly)
CAMERA_PROMPT_TEMPLATES = {
    "bokeh":              lambda v: f"<bokeh kernel size: {v}>",
    "focal_length":       lambda v: f"<focal length: {v}>",
    "shutter_speed":      lambda v: f"<exposure: {v}>",
    "color_temperature":  lambda v: f"<color temperature: {v}>",
}

PHYSICAL_EMBEDDING_FN = {
    "bokeh":             create_bokehK_embedding,
    "focal_length":      create_focal_length_embedding,
    "shutter_speed":     create_shutter_speed_embedding,
    "color_temperature": create_color_temperature_embedding,
}


class UnifiedCameraEmbedding:
    """Builds the 6-channel camera embedding (3ch physical + 3ch CCL) for any camera type."""

    def __init__(self, camera_type, camera_values, tokenizer, text_encoder,
                 device, sample_size=(256, 384)):
        assert camera_type in CAMERA_PROMPT_TEMPLATES, \
            f"Unknown camera_type '{camera_type}'. Choose from {list(CAMERA_PROMPT_TEMPLATES.keys())}"
        self.camera_type = camera_type
        self.camera_values = camera_values.to(device)
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device
        self.sample_size = list(sample_size)

    def build(self):
        if len(self.camera_values) != 5:
            raise ValueError("Expected exactly 5 camera parameter values")

        # --- CCL embedding (identical logic across all types) ---
        prompt_fn = CAMERA_PROMPT_TEMPLATES[self.camera_type]
        prompts = [prompt_fn(v.item()) for v in self.camera_values]

        with torch.no_grad():
            prompt_ids = self.tokenizer(
                prompts, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.device)
            encoder_hidden_states = self.text_encoder(input_ids=prompt_ids).last_hidden_state

        # Consecutive differences + wrap-around
        differences = []
        for i in range(1, encoder_hidden_states.size(0)):
            diff = (encoder_hidden_states[i] - encoder_hidden_states[i - 1]).unsqueeze(0)
            differences.append(diff)
        final_diff = (encoder_hidden_states[-1] - encoder_hidden_states[0]).unsqueeze(0)
        differences.append(final_diff)

        concatenated_differences = torch.cat(differences, dim=0)
        frame = concatenated_differences.size(0)
        concatenated_differences = torch.cat(differences, dim=0)

        pad_length = 128 - concatenated_differences.size(1)
        if pad_length > 0:
            concatenated_differences = F.pad(concatenated_differences, (0, 0, 0, pad_length))

        ccl_embedding = concatenated_differences.reshape(frame, self.sample_size[0], self.sample_size[1])
        ccl_embedding = ccl_embedding.unsqueeze(1).expand(-1, 3, -1, -1).to(self.device)

        # --- Physical embedding (type-specific) ---
        physical_fn = PHYSICAL_EMBEDDING_FN[self.camera_type]
        physical_embedding = physical_fn(self.camera_values, self.sample_size[0], self.sample_size[1]).to(self.device)

        # --- Concatenate: [physical(3ch), ccl(3ch)] -> 6ch ---
        camera_embedding = torch.cat((physical_embedding, ccl_embedding), dim=1)
        return camera_embedding


# ============================================================================
#  Image preprocessing
# ============================================================================

def load_and_preprocess_image(image_path, height=256, width=384):
    """Load an image and preprocess it for VAE encoding.

    Returns:
        image_tensor: [1, 3, H, W] in range [-1, 1]
    """
    image = Image.open(image_path).convert("RGB")

    transform = T.Compose([
        T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),            # [0, 1]
        T.Normalize([0.5], [0.5])  # [-1, 1]
    ])
    image_tensor = transform(image).unsqueeze(0)  # [1, 3, H, W]
    return image_tensor


def save_individual_frames(video_tensor, output_dir, prefix="frame"):
    """Save each frame of the generated video as an individual image.

    Args:
        video_tensor: [1, C, F, H, W] tensor in [0, 1]
        output_dir: directory to save frames
        prefix: filename prefix
    """
    os.makedirs(output_dir, exist_ok=True)
    video = video_tensor[0]  # [C, F, H, W]
    for i in range(video.shape[1]):
        frame = video[:, i, :, :]  # [C, H, W]
        frame_path = os.path.join(output_dir, f"{prefix}_{i:02d}.png")
        torchvision.utils.save_image(frame, frame_path)
        logger.info(f"  Saved {frame_path}")


# ============================================================================
#  Model loading (identical to original scripts)
# ============================================================================

def load_models(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(cfg.noise_scheduler_kwargs))
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_path, subfolder="vae").to(device)
    vae.requires_grad_(False)
    tokenizer = CLIPTokenizer.from_pretrained(cfg.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.pretrained_model_path, subfolder="text_encoder").to(device)
    text_encoder.requires_grad_(False)
    unet = UNet3DConditionModelCameraCond.from_pretrained_2d(
        cfg.pretrained_model_path,
        subfolder=cfg.unet_subfolder,
        unet_additional_kwargs=cfg.unet_additional_kwargs
    ).to(device)
    unet.requires_grad_(False)

    camera_encoder = CameraCameraEncoder(**cfg.camera_encoder_kwargs).to(device)
    camera_encoder.requires_grad_(False)
    camera_adaptor = CameraAdaptor(unet, camera_encoder)
    camera_adaptor.requires_grad_(False)
    camera_adaptor.to(device)

    logger.info("Setting the attention processors")
    unet.set_all_attn_processor(
        add_spatial_lora=cfg.lora_ckpt is not None,
        add_motion_lora=cfg.motion_lora_rank > 0,
        lora_kwargs={"lora_rank": cfg.lora_rank, "lora_scale": cfg.lora_scale},
        motion_lora_kwargs={"lora_rank": cfg.motion_lora_rank, "lora_scale": cfg.motion_lora_scale},
        **cfg.attention_processor_kwargs
    )

    if cfg.lora_ckpt is not None:
        logger.info(f"Loading LoRA checkpoint from {cfg.lora_ckpt}")
        lora_checkpoints = torch.load(cfg.lora_ckpt, map_location=unet.device)
        if 'lora_state_dict' in lora_checkpoints.keys():
            lora_checkpoints = lora_checkpoints['lora_state_dict']
        _, lora_u = unet.load_state_dict(lora_checkpoints, strict=False)
        assert len(lora_u) == 0
        logger.info("LoRA loading done")

    if cfg.motion_module_ckpt is not None:
        logger.info(f"Loading motion module from {cfg.motion_module_ckpt}")
        mm_checkpoints = torch.load(cfg.motion_module_ckpt, map_location=unet.device)
        _, mm_u = unet.load_state_dict(mm_checkpoints, strict=False)
        assert len(mm_u) == 0
        logger.info("Motion module loading done")

    if cfg.camera_adaptor_ckpt is not None:
        logger.info(f"Loading camera adaptor from {cfg.camera_adaptor_ckpt}")
        camera_adaptor_checkpoint = torch.load(cfg.camera_adaptor_ckpt, map_location=device)
        camera_encoder_state_dict = camera_adaptor_checkpoint['camera_encoder_state_dict']
        attention_processor_state_dict = camera_adaptor_checkpoint['attention_processor_state_dict']
        camera_enc_m, camera_enc_u = camera_adaptor.camera_encoder.load_state_dict(
            camera_encoder_state_dict, strict=False)
        assert len(camera_enc_m) == 0 and len(camera_enc_u) == 0
        _, attention_processor_u = camera_adaptor.unet.load_state_dict(
            attention_processor_state_dict, strict=False)
        assert len(attention_processor_u) == 0
        logger.info("Camera adaptor loading done")

    pipeline = GenPhotoPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        unet=unet, scheduler=noise_scheduler, camera_encoder=camera_encoder
    ).to(device)
    pipeline.enable_vae_slicing()

    return pipeline, device


# ============================================================================
#  Inference entry point
# ============================================================================

def run_inference(pipeline, device, args):
    """Run image-conditioned (or text-only) generative photography inference."""

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    height, width = args.height, args.width
    strength = args.strength
    camera_type = args.camera_type

    # --- Parse camera values ---
    camera_values_list = json.loads(args.camera_values)
    camera_values = torch.tensor(camera_values_list)
    if camera_values.ndim == 1:
        camera_values = camera_values.unsqueeze(1)

    # --- Build camera embedding ---
    cam_emb = UnifiedCameraEmbedding(
        camera_type=camera_type,
        camera_values=camera_values,
        tokenizer=pipeline.tokenizer,
        text_encoder=pipeline.text_encoder,
        device=device,
        sample_size=(height, width),
    )
    camera_embedding = cam_emb.build()
    camera_embedding = rearrange(camera_embedding.unsqueeze(0), "b f c h w -> b c f h w")

    # --- Image encoding (img2img) or pure text2img ---
    init_image_latents = None
    prompt = args.prompt

    if args.image and strength < 1.0:
        # img2img mode
        logger.info(f"Loading input image: {args.image}")
        image_tensor = load_and_preprocess_image(args.image, height, width).to(device)
        logger.info(f"Encoding image to latent space...")
        init_image_latents = pipeline.encode_image(image_tensor)
        logger.info(f"  Image latent shape: {init_image_latents.shape}")

        if not prompt:
            # Use a generic prompt — the latent initialization carries scene info
            prompt = "a high quality photograph"
            logger.info(f"  No --prompt given, using generic prompt: '{prompt}'")
    else:
        # Pure text2img (original behavior)
        if not prompt:
            raise ValueError(
                "Either provide --image with strength < 1.0 for img2img, "
                "or provide --prompt for text-to-image."
            )
        if strength < 1.0:
            logger.warning("strength < 1.0 but no --image provided. Running as text2img (strength=1.0).")
            strength = 1.0

    logger.info(f"Running inference: camera_type={camera_type}, strength={strength:.2f}")
    logger.info(f"  Camera values: {camera_values_list}")
    logger.info(f"  Prompt: '{prompt}'")

    with torch.no_grad():
        sample = pipeline(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            camera_embedding=camera_embedding,
            video_length=5,
            height=height,
            width=width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            strength=strength,
            init_image_latents=init_image_latents,
        ).videos[0]

    # --- Save outputs ---
    # Save as animated GIF
    gif_path = os.path.join(output_dir, f"{camera_type}_img2img.gif")
    save_videos_grid(sample[None, ...], gif_path)
    logger.info(f"Saved GIF to {gif_path}")

    # Save individual frames
    frames_dir = os.path.join(output_dir, f"{camera_type}_frames")
    save_individual_frames(sample.unsqueeze(0), frames_dir, prefix=camera_type)

    # Save the input image resized (for comparison)
    if args.image:
        input_save = os.path.join(output_dir, "input_resized.png")
        img = load_and_preprocess_image(args.image, height, width)
        torchvision.utils.save_image(img * 0.5 + 0.5, input_save)  # de-normalize
        logger.info(f"Saved resized input to {input_save}")

    logger.info("Done!")


# ============================================================================
#  CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Image-conditioned Generative Photography inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config (e.g. configs/inference_genphoto/adv3_256_384_genphoto_relora_bokehK.yaml)")
    parser.add_argument("--camera_type", type=str, required=True,
                        choices=["bokeh", "focal_length", "shutter_speed", "color_temperature"],
                        help="Type of camera parameter to vary")
    parser.add_argument("--camera_values", type=str, required=True,
                        help='JSON list of 5 parameter values, e.g. "[2.44, 8.3, 10.1, 17.2, 24.0]"')

    # Image input (for img2img)
    parser.add_argument("--image", type=str, default=None,
                        help="Path to input image. Required for img2img mode.")

    # Scene description (for cross-attention)
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt for scene description. "
                             "If omitted in img2img mode, a generic prompt is used.")
    parser.add_argument("--negative_prompt", type=str, default=None,
                        help="Negative prompt for classifier-free guidance")

    # img2img control
    parser.add_argument("--strength", type=float, default=0.6,
                        help="Denoising strength for img2img (0.0-1.0). "
                             "Lower = more faithful to input image, less camera effect. "
                             "Higher = more camera effect, less scene fidelity. "
                             "1.0 = pure text-to-image (ignores input image). "
                             "Recommended: 0.4-0.7. Default: 0.6")

    # Generation parameters
    parser.add_argument("--num_inference_steps", type=int, default=25,
                        help="Number of denoising steps (default: 25)")
    parser.add_argument("--guidance_scale", type=float, default=8.0,
                        help="Classifier-free guidance scale (default: 8.0)")
    parser.add_argument("--height", type=int, default=256,
                        help="Output height in pixels (default: 256)")
    parser.add_argument("--width", type=int, default=384,
                        help="Output width in pixels (default: 384)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: inference_output/img2img_{camera_type})")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Default output dir
    if args.output_dir is None:
        mode = "img2img" if (args.image and args.strength < 1.0) else "txt2img"
        args.output_dir = f"inference_output/{mode}_{args.camera_type}"

    # Load config and models
    cfg = OmegaConf.load(args.config)
    logger.info("Loading models...")
    pipeline, device = load_models(cfg)
    logger.info("Models loaded successfully")

    # Run inference
    run_inference(pipeline, device, args)


if __name__ == "__main__":
    main()
