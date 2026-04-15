"""
Fast Multi-GPU Test Evaluation Script for ADP-DiT-G.

Reads test.csv, runs inference on all samples using 8 GPUs,
computes metrics (PSNR, MSE, SSIM), and saves results to CSV.

Usage:
    python -m adpdit.evaluate_test_csv \
        --test-csv /path/to/test.csv \
        --checkpoint /path/to/checkpoint \
        --output-csv ./test_results.csv
"""

import os
import sys
import csv
import json
import argparse
import gc
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message="Token indices sequence length is longer")

# Suppress transformers/diffusers logging
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adpdit.utils.metrics import compute_psnr, compute_mse, compute_ssim
from adpdit.constants import SAMPLER_FACTORY


# ============================================================================
# Error Map and Comparison Image Functions
# ============================================================================

def create_colorbar(height, width=40, cmap_name='inferno', z_min=0, z_max=1, num_ticks=5):
    """Create a colorbar image for the error map."""
    actual_height = max(10, height - 20)

    colorbar_array = np.linspace(1, 0, actual_height).reshape(-1, 1)
    colorbar_array = np.repeat(colorbar_array, width, axis=1)

    cmap = plt.colormaps[cmap_name]
    colorbar_colored = cmap(colorbar_array)
    colorbar_rgb = (colorbar_colored[:, :, :3] * 255).astype(np.uint8)

    colorbar_img = Image.fromarray(colorbar_rgb)

    extended_width = width + 60
    extended_colorbar = Image.new('RGB', (extended_width, height), 'white')

    y_offset = 10
    extended_colorbar.paste(colorbar_img, (0, y_offset))

    draw = ImageDraw.Draw(extended_colorbar)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font = ImageFont.load_default()

    tick_positions = np.linspace(0, actual_height-1, num_ticks).astype(int) + y_offset
    tick_values = np.linspace(z_max, z_min, num_ticks)

    for pos, val in zip(tick_positions, tick_values):
        draw.line([(width, pos), (width + 3, pos)], fill='black', width=1)
        label = f"{val:.2f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_height = bbox[3] - bbox[1]
        draw.text((width + 5, pos - text_height//2), label, fill='black', font=font)

    return extended_colorbar


def compute_error_map(target_array, output_array):
    """
    Compute error map between target and output images.
    Returns error map RGB image and statistics.
    """
    # Convert to grayscale
    if len(target_array.shape) == 3:
        target_gray = np.mean(target_array.astype(np.float32), axis=2)
        output_gray = np.mean(output_array.astype(np.float32), axis=2)
    else:
        target_gray = target_array.astype(np.float32)
        output_gray = output_array.astype(np.float32)

    # Normalize to 0-1 range
    target_gray = (target_gray - target_gray.min()) / (target_gray.max() - target_gray.min() + 1e-8)
    output_gray = (output_gray - output_gray.min()) / (output_gray.max() - output_gray.min() + 1e-8)

    # Compute MSE map
    mse_map = (target_gray - output_gray) ** 2

    # Normalize MSE to 0-1 range
    mse_min = mse_map.min()
    mse_max = mse_map.max()

    if mse_max - mse_min > 1e-8:
        mse_normalized = (mse_map - mse_min) / (mse_max - mse_min)
    else:
        mse_normalized = np.zeros_like(mse_map)

    # Apply gamma correction (gamma > 1 suppresses small differences, highlights large ones)
    gamma = 0.9
    mse_gamma = np.power(mse_normalized, gamma)

    # Apply inferno colormap
    inferno_cmap = plt.colormaps['inferno']
    error_map_colored = inferno_cmap(mse_gamma)

    # Convert RGBA to RGB
    error_map_rgb = (error_map_colored[:, :, :3] * 255).astype(np.uint8)

    # Compute statistics
    avg_mse = np.mean(mse_map)
    std_mse = np.std(mse_map)

    # z_min, z_max for visualization range
    z_min = 0.0
    z_max = 1.0

    return error_map_rgb, avg_mse, std_mse, z_min, z_max


def create_comparison_image(input_img, output_img, target_img, save_path,
                           patient_info="", metrics_text=""):
    """
    Create 4-column comparison image: Input | Target | Output | Error Map

    Args:
        input_img: PIL Image (input)
        output_img: PIL Image (generated output)
        target_img: PIL Image (ground truth)
        save_path: Path to save comparison image
        patient_info: Text with patient/sample info
        metrics_text: Text with metrics info
    """
    try:
        # Convert to grayscale first, then to RGB for display
        input_img = input_img.convert('L').convert('RGB')
        output_img = output_img.convert('L').convert('RGB')
        target_img = target_img.convert('L').convert('RGB')

        # Resize to uniform size
        min_width = min(input_img.width, output_img.width, target_img.width)
        min_height = min(input_img.height, output_img.height, target_img.height)

        input_img = input_img.resize((min_width, min_height), Image.LANCZOS)
        output_img = output_img.resize((min_width, min_height), Image.LANCZOS)
        target_img = target_img.resize((min_width, min_height), Image.LANCZOS)

        # Compute error map (output vs target)
        target_array = np.array(target_img)
        output_array = np.array(output_img)
        error_map_rgb, avg_mse, std_mse, z_min, z_max = compute_error_map(target_array, output_array)
        img_error = Image.fromarray(error_map_rgb)

        # Create colorbar
        colorbar_img = create_colorbar(min_height, cmap_name='inferno', z_min=z_min, z_max=z_max)

        # Layout setup
        label_height = 40
        colorbar_width = colorbar_img.width
        gap = 10
        total_width = min_width * 4 + gap * 3 + colorbar_width + gap
        total_height = min_height + label_height

        # Create composite image
        comparison_img = Image.new('RGB', (total_width, total_height), 'white')

        # Paste images: Input | Target | Output | Error Map
        comparison_img.paste(input_img, (0, label_height))
        comparison_img.paste(target_img, (min_width + gap, label_height))
        comparison_img.paste(output_img, (min_width * 2 + gap * 2, label_height))
        comparison_img.paste(img_error, (min_width * 3 + gap * 3, label_height))
        comparison_img.paste(colorbar_img, (min_width * 4 + gap * 4, label_height))

        # Add text labels
        draw = ImageDraw.Draw(comparison_img)

        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            font_info = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        except:
            font_title = ImageFont.load_default()
            font_info = ImageFont.load_default()

        # Add info text at top
        if patient_info and metrics_text:
            combined_info = f"{patient_info} | {metrics_text}"
            draw.text((10, 5), combined_info, fill='black', font=font_info)
        elif patient_info:
            draw.text((10, 5), patient_info, fill='black', font=font_info)
        elif metrics_text:
            draw.text((10, 5), metrics_text, fill='black', font=font_info)

        # Column labels: Input | Target | Output | Error Map
        labels = ['Input', 'Target', 'Output', 'Error Map']
        positions = [
            min_width // 2,
            min_width + gap + min_width // 2,
            min_width * 2 + gap * 2 + min_width // 2,
            min_width * 3 + gap * 3 + min_width // 2
        ]
        colors = ['blue', 'green', 'red', 'orange']

        for label, pos, color in zip(labels, positions, colors):
            bbox = draw.textbbox((0, 0), label, font=font_title)
            text_width = bbox[2] - bbox[0]
            x = pos - text_width // 2
            draw.text((x, label_height - 20), label, fill=color, font=font_title)

        # Save image
        comparison_img.save(save_path, 'PNG')
        return True

    except Exception as e:
        print(f"Failed to create comparison image ({save_path}): {e}")
        return False


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate test.csv with multi-GPU inference')

    # Input/Output paths
    parser.add_argument('--test-csv', type=str, required=True,
                        help='Path to test.csv file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (directory or .pt file)')
    parser.add_argument('--output-csv', type=str, default='./test_results.csv',
                        help='Path to output CSV with metrics')
    parser.add_argument('--output-dir', type=str, default='./test_outputs',
                        help='Directory to save generated images')
    parser.add_argument('--data-root', type=str,
                        default='./dataset/AD_meta',
                        help='Root directory for dataset images')

    # Inference settings
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size per GPU (default: 1 for stability)')
    parser.add_argument('--infer-steps', type=int, default=50,
                        help='Number of inference steps (50 for high quality medical images)')
    parser.add_argument('--strength', type=float, default=0.4,
                        help='Strength for img2img (0.3-0.5 for subtle Alzheimer progression)')
    parser.add_argument('--cfg-scale', type=float, default=4.5,
                        help='Classifier-free guidance scale (4-5 for natural anatomical changes)')
    parser.add_argument('--sampler', type=str, default='dpmpp_2m_karras',
                        choices=['ddpm', 'ddim', 'dpmms', 'dpmpp_2m_sde_exp', 'dpmpp_3m_sde_karras', 'dpmpp_2m_karras'],
                        help='Sampler type (dpmpp_2m_karras recommended)')

    # GPU settings
    parser.add_argument('--num-gpus', type=int, default=8,
                        help='Number of GPUs to use')

    # Options
    parser.add_argument('--save-images', action='store_true',
                        help='Save generated images to output-dir/generated/')
    parser.add_argument('--save-comparisons', action='store_true',
                        help='Save 4-column comparison images to output-dir/comparisons/')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous run (skip completed samples)')

    return parser.parse_args()


def extract_diagnosis_from_prompt(prompt: str) -> str:
    """Extract diagnosis (CN, MCI, AD) from prompt."""
    prompt_lower = prompt.lower()
    if 'cognitive normal' in prompt_lower:
        return 'CN'
    elif 'mild cognitive impairment' in prompt_lower:
        return 'MCI'
    elif 'alzheimer' in prompt_lower:
        return 'AD'
    return 'Unknown'


def extract_subject_id(image_path: str) -> str:
    """Extract subject ID from image path (e.g., '013_S_4580' from './images/test/013_S_4580_0_100.png')."""
    filename = os.path.basename(image_path)
    # Pattern: {site}_{S}_{id}_{visit}_{slice}.png
    parts = filename.replace('.png', '').split('_')
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}_{parts[2]}"
    return filename


def extract_months_from_prompt(prompt: str) -> int:
    """Extract months from first visit from prompt."""
    # Check for 'first visit' pattern
    if 'first visit' in prompt.lower() and 'months from first visit' not in prompt.lower():
        return 0

    # Check for 'X months from first visit' pattern
    match = re.search(r'(\d+)\s*months?\s+from\s+first\s+visit', prompt.lower())
    if match:
        return int(match.group(1))
    return 0


def load_test_csv(csv_path: str) -> Tuple[List[Dict], Dict[str, str]]:
    """
    Load test.csv and return list of sample dicts with progression type.

    Returns:
        Tuple of (samples list, subject_baseline_diagnosis dict)
    """
    # First pass: collect all first visit diagnoses per subject
    subject_baseline = {}  # subject_id -> diagnosis at first visit
    all_rows = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            all_rows.append((idx, row))

            prompt = row['edit_prompt']
            subject_id = extract_subject_id(row['input_image'])
            months = extract_months_from_prompt(prompt)
            diagnosis = extract_diagnosis_from_prompt(prompt)

            # If this is a first visit row (input == edited, or months == 0 with 'first visit')
            if row['input_image'] == row['edited_image'] and 'first visit' in prompt.lower():
                subject_baseline[subject_id] = diagnosis

    # Second pass: build samples with progression type
    samples = []
    for idx, row in all_rows:
        subject_id = extract_subject_id(row['input_image'])
        target_diagnosis = extract_diagnosis_from_prompt(row['edit_prompt'])
        months = extract_months_from_prompt(row['edit_prompt'])

        # Get baseline diagnosis for this subject
        baseline_diagnosis = subject_baseline.get(subject_id, 'Unknown')

        # Determine progression type
        if baseline_diagnosis != 'Unknown' and target_diagnosis != 'Unknown':
            progression_type = f"{baseline_diagnosis}to{target_diagnosis}"
        else:
            progression_type = 'Unknown'

        # Check if this is a first visit (baseline) sample
        is_first_visit = (row['input_image'] == row['edited_image'] and
                         'first visit' in row['edit_prompt'].lower() and
                         'months from first visit' not in row['edit_prompt'].lower())

        samples.append({
            'csv_row_index': idx,
            'input_image': row['input_image'],
            'edited_image': row['edited_image'],
            'edit_prompt': row['edit_prompt'],
            'subject_id': subject_id,
            'baseline_diagnosis': baseline_diagnosis,
            'target_diagnosis': target_diagnosis,
            'progression_type': progression_type,
            'months_from_first_visit': months,
            'is_first_visit': is_first_visit
        })

    print(f"Loaded {len(samples)} samples")
    print(f"Found baseline diagnoses for {len(subject_baseline)} subjects")

    # Print progression type distribution
    progression_counts = {}
    for s in samples:
        pt = s['progression_type']
        progression_counts[pt] = progression_counts.get(pt, 0) + 1
    print("Progression type distribution:")
    for pt, count in sorted(progression_counts.items()):
        print(f"  {pt}: {count}")

    return samples, subject_baseline


def worker_process(gpu_id: int, samples_subset: List[Dict], args, results_queue):
    """
    Worker process that runs inference on a single GPU.

    Args:
        gpu_id: GPU device ID
        samples_subset: Subset of samples to process
        args: Command line arguments
        results_queue: Queue to put results
    """
    try:
        # Set device
        device = f'cuda:{gpu_id}'
        torch.cuda.set_device(gpu_id)

        print(f"[GPU {gpu_id}] Starting worker with {len(samples_subset)} samples")

        # Import modules
        from adpdit.modules.models import ADPDiT
        from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer
        from adpdit.diffusion.pipeline import StableDiffusionPipeline
        from torchvision import transforms

        # Create model config
        class ModelConfig:
            def __init__(self):
                self.image_size = 1024
                self.patch_size = 2
                self.depth = 40
                self.hidden_size = 1408
                self.num_heads = 16
                self.mlp_ratio = 4.3637
                self.learn_sigma = True
                self.text_states_dim = 1280
                self.text_states_dim_t5 = 4096
                self.text_len = 77
                self.text_len_t5 = 256
                self.norm = 'layer'
                self.infer_mode = 'torch'
                self.use_flash_attn = False
                self.qk_norm = True
                self.size_cond = None
                self.use_style_cond = False
                self.use_style_encoder = False
                self.use_img_meta = False
                self.rope_img = 'extend'
                self.rope_real = True

        model_args = ModelConfig()

        # Load checkpoint
        print(f"[GPU {gpu_id}] Loading model...")
        if os.path.isdir(args.checkpoint):
            ckpt_path = os.path.join(args.checkpoint, 'mp_rank_00_model_states.pt')
        else:
            ckpt_path = args.checkpoint

        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        # Initialize model
        model = ADPDiT(
            args=model_args,
            input_size=(model_args.image_size // 8, model_args.image_size // 8),
            patch_size=model_args.patch_size,
            in_channels=4,
            hidden_size=model_args.hidden_size,
            depth=model_args.depth,
            num_heads=model_args.num_heads,
            mlp_ratio=model_args.mlp_ratio,
            log_fn=print
        )

        # Load state dict
        if 'module' in checkpoint:
            state_dict = checkpoint['module']
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=False)
        model = model.to(device)
        model.eval()
        model.args.size_cond = None
        model.args.use_style_cond = False

        # Load VAE and text encoders
        print(f"[GPU {gpu_id}] Loading VAE and text encoders...")
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
        vae.eval()

        clip_tokenizer = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
        clip_encoder = CLIPTextModel.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k").to(device)
        clip_encoder.eval()

        t5_tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
        t5_encoder = T5EncoderModel.from_pretrained("google/t5-v1_1-xxl").to(device)
        t5_encoder.eval()

        # T5 embedder wrapper
        class T5Embedder:
            def __init__(self, model, tokenizer, max_length):
                self.model = model
                self.tokenizer = tokenizer
                self.max_length = max_length

        t5_embedder = T5Embedder(t5_encoder, t5_tokenizer, model_args.text_len_t5)

        # Scheduler setup
        sampler_config = SAMPLER_FACTORY[args.sampler]
        scheduler_class_name = sampler_config['scheduler']
        scheduler_kwargs = sampler_config['kwargs'].copy()

        scheduler_classes = {
            'DDPMScheduler': DDPMScheduler,
            'DDIMScheduler': DDIMScheduler,
            'DPMSolverMultistepScheduler': DPMSolverMultistepScheduler,
        }

        scheduler_class = scheduler_classes.get(scheduler_class_name)
        if scheduler_class is None:
            raise ValueError(f"Unknown scheduler class: {scheduler_class_name}")

        if scheduler_class_name == 'DPMSolverMultistepScheduler' and 'config' in sampler_config:
            scheduler_kwargs.update(sampler_config['config'])

        scheduler = scheduler_class(**scheduler_kwargs)

        # Create pipeline
        pipeline = StableDiffusionPipeline(
            vae=vae,
            text_encoder=clip_encoder,
            tokenizer=clip_tokenizer,
            unet=model,
            scheduler=scheduler,
            embedder_t5=t5_embedder,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        )
        # Disable internal progress bar for cleaner output
        pipeline.set_progress_bar_config(disable=True)

        # Setup RoPE
        from adpdit.modules.posemb_layers import get_2d_rotary_pos_embed
        latent_size = model_args.image_size // 8
        patch_grid_size = latent_size // model_args.patch_size
        head_dim = model_args.hidden_size // model_args.num_heads

        freqs_cis_img = get_2d_rotary_pos_embed(
            head_dim,
            (patch_grid_size, patch_grid_size),
            use_real=model_args.rope_real
        )

        if model_args.rope_real:
            freqs_cis_img = tuple([freq.to(device) for freq in freqs_cis_img])
        else:
            freqs_cis_img = freqs_cis_img.to(device)

        print(f"[GPU {gpu_id}] Pipeline ready, starting inference...")

        # Transforms
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize([0.5], [0.5])

        # Process samples in batches
        results = []
        batch_size = args.batch_size
        num_batches = (len(samples_subset) + batch_size - 1) // batch_size

        pbar = tqdm(total=len(samples_subset), desc=f"GPU {gpu_id}", position=gpu_id)

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(samples_subset))
            batch_samples = samples_subset[batch_start:batch_end]
            actual_batch_size = len(batch_samples)

            try:
                # Load and preprocess images for the batch
                input_imgs = []
                gt_imgs = []
                clip_prompts = []
                valid_indices = []  # Track which samples loaded successfully

                for i, sample in enumerate(batch_samples):
                    try:
                        # Remap paths
                        input_path = sample['input_image']
                        edited_path = sample['edited_image']

                        # Handle relative paths
                        if input_path.startswith('./images/'):
                            input_path = os.path.join(args.data_root, input_path[2:])
                            edited_path = os.path.join(args.data_root, edited_path[2:])
                        elif not os.path.isabs(input_path):
                            input_path = os.path.join(args.data_root, input_path)
                            edited_path = os.path.join(args.data_root, edited_path)

                        # Load images
                        input_img = Image.open(input_path).convert('RGB')
                        gt_img = Image.open(edited_path).convert('RGB')

                        # Resize to 1024x1024
                        if input_img.size != (model_args.image_size, model_args.image_size):
                            input_img = input_img.resize((model_args.image_size, model_args.image_size), Image.LANCZOS)
                        if gt_img.size != (model_args.image_size, model_args.image_size):
                            gt_img = gt_img.resize((model_args.image_size, model_args.image_size), Image.LANCZOS)

                        # Prepare prompt
                        full_prompt = sample['edit_prompt']
                        clip_tokens = clip_tokenizer.encode(full_prompt)
                        if len(clip_tokens) > 77:
                            clip_prompt = clip_tokenizer.decode(clip_tokens[:76])
                        else:
                            clip_prompt = full_prompt

                        input_imgs.append(input_img)
                        gt_imgs.append(gt_img)
                        clip_prompts.append(clip_prompt)
                        valid_indices.append(i)

                    except Exception as e:
                        print(f"[GPU {gpu_id}] Error loading sample {batch_start + i}: {e}")
                        continue

                if not valid_indices:
                    pbar.update(actual_batch_size)
                    continue

                # Stack into batch tensor [B, 3, 1024, 1024]
                input_tensors = torch.stack([normalize(to_tensor(img)) for img in input_imgs]).to(device)

                # Batch VAE encoding
                with torch.no_grad():
                    input_latents_batch = vae.encode(input_tensors).latent_dist.sample() * 0.18215

                # Run batch inference
                with torch.no_grad():
                    output = pipeline(
                        height=model_args.image_size,
                        width=model_args.image_size,
                        prompt=clip_prompts,  # List[str] for batch
                        num_inference_steps=args.infer_steps,
                        guidance_scale=args.cfg_scale,
                        strength=args.strength,
                        image_latents=input_latents_batch,  # [B, 4, 128, 128]
                        input_image=input_latents_batch,  # Clean input for skip connections
                        freqs_cis_img=freqs_cis_img,
                        negative_prompt=[""] * len(valid_indices),  # List for batch
                        return_dict=True
                    )

                if output is None:
                    print(f"[GPU {gpu_id}] Pipeline returned None for batch {batch_idx}")
                    pbar.update(actual_batch_size)
                    continue

                # Process each sample in the batch
                for i, valid_idx in enumerate(valid_indices):
                    sample = batch_samples[valid_idx]
                    generated_img = output.images[i]
                    input_img = input_imgs[i]
                    gt_img = gt_imgs[i]

                    # Compute metrics
                    gen_np = np.array(generated_img)
                    gt_np = np.array(gt_img)

                    psnr = compute_psnr(gen_np, gt_np)
                    mse = compute_mse(gen_np, gt_np)
                    try:
                        ssim = compute_ssim(gen_np, gt_np)
                    except Exception as e:
                        ssim = 0.0

                    # Save generated image if requested
                    saved_image_path = None
                    saved_comparison_path = None
                    if args.save_images or args.save_comparisons:
                        # Extract filename from target (edited) image path
                        target_filename = os.path.basename(sample['edited_image'])
                        base_name, ext = os.path.splitext(target_filename)

                        # Save generated image to generated/ subdirectory
                        if args.save_images:
                            generated_dir = os.path.join(args.output_dir, 'generated')
                            os.makedirs(generated_dir, exist_ok=True)
                            img_filename = f"{base_name}.png"
                            img_save_path = os.path.join(generated_dir, img_filename)
                            # Resize to 256x256 and convert to grayscale before saving
                            generated_img_resized = generated_img.resize((256, 256), Image.LANCZOS)
                            generated_img_gray = generated_img_resized.convert('L')
                            generated_img_gray.save(img_save_path)
                            saved_image_path = img_save_path

                        # Save 4-column comparison image to comparisons/ subdirectory
                        if args.save_comparisons:
                            comparisons_dir = os.path.join(args.output_dir, 'comparisons')
                            os.makedirs(comparisons_dir, exist_ok=True)
                            comparison_filename = f"{base_name}_comparison.png"
                            comparison_save_path = os.path.join(comparisons_dir, comparison_filename)

                            # Create patient/sample info text
                            prog_type = sample.get('progression_type', '')
                            months = sample.get('months_from_first_visit', 0)
                            patient_info = f"{prog_type} | {sample.get('subject_id', '')} | {months} months"

                            # Create metrics text
                            metrics_text = f"PSNR: {psnr:.2f} dB | MSE: {mse:.6f} | SSIM: {ssim:.4f}"

                            # Create comparison image
                            create_comparison_image(
                                input_img, generated_img, gt_img,
                                comparison_save_path,
                                patient_info, metrics_text
                            )
                            saved_comparison_path = comparison_save_path

                    # Store result with progression info
                    result = {
                        'csv_row_index': sample['csv_row_index'],
                        'subject_id': sample.get('subject_id', ''),
                        'input_image': sample['input_image'],
                        'edited_image': sample['edited_image'],
                        'edit_prompt': sample['edit_prompt'],
                        'progression_type': sample.get('progression_type', ''),
                        'months_from_first_visit': sample.get('months_from_first_visit', 0),
                        'is_first_visit': sample.get('is_first_visit', False),
                        'psnr': float(psnr),
                        'mse': float(mse),
                        'ssim': float(ssim),
                        'generated_image_path': saved_image_path,
                        'comparison_image_path': saved_comparison_path
                    }
                    results.append(result)

                pbar.update(actual_batch_size)

                # Cleanup every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            except Exception as e:
                print(f"[GPU {gpu_id}] Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                pbar.update(actual_batch_size)
                continue

        pbar.close()

        # Send results back
        results_queue.put((gpu_id, results))
        print(f"[GPU {gpu_id}] Completed {len(results)} samples")

    except Exception as e:
        print(f"[GPU {gpu_id}] Worker error: {e}")
        import traceback
        traceback.print_exc()
        results_queue.put((gpu_id, []))


def main():
    args = parse_args()

    print("=" * 80)
    print("ADP-DiT-G Test CSV Evaluation")
    print("=" * 80)
    print(f"Test CSV: {args.test_csv}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Num GPUs: {args.num_gpus}")
    print(f"Inference Steps: {args.infer_steps}")
    print(f"Strength: {args.strength}")
    print(f"CFG Scale: {args.cfg_scale}")
    print(f"Save Images: {args.save_images}")
    print(f"Save Comparisons: {args.save_comparisons}")
    print("=" * 80)

    # Load test CSV
    print("Loading test.csv...")
    samples, subject_baseline = load_test_csv(args.test_csv)
    total_samples = len(samples)
    print(f"Total samples: {total_samples}")

    # Check for resume
    completed_indices = set()
    if args.resume and os.path.exists(args.output_csv):
        print("Resuming from previous run...")
        try:
            existing_df = pd.read_csv(args.output_csv)
            completed_indices = set(existing_df['csv_row_index'].tolist())
            print(f"Found {len(completed_indices)} completed samples")
        except Exception as e:
            print(f"Could not load existing results: {e}")

    # Filter out completed samples
    if completed_indices:
        samples = [s for s in samples if s['csv_row_index'] not in completed_indices]
        print(f"Remaining samples to process: {len(samples)}")

    if not samples:
        print("No samples to process!")
        return

    # Create output directory
    if args.save_images:
        os.makedirs(args.output_dir, exist_ok=True)

    # Distribute samples across GPUs
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    print(f"Using {num_gpus} GPUs")

    samples_per_gpu = len(samples) // num_gpus
    gpu_samples = []
    for i in range(num_gpus):
        start_idx = i * samples_per_gpu
        if i == num_gpus - 1:
            end_idx = len(samples)
        else:
            end_idx = start_idx + samples_per_gpu
        gpu_samples.append(samples[start_idx:end_idx])

    # Create multiprocessing queue and spawn workers
    mp.set_start_method('spawn', force=True)
    results_queue = mp.Queue()

    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, gpu_samples[gpu_id], args, results_queue)
        )
        p.start()
        processes.append(p)

    # Collect results
    all_results = []
    for _ in range(num_gpus):
        gpu_id, results = results_queue.get()
        all_results.extend(results)
        print(f"Collected {len(results)} results from GPU {gpu_id}")

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print(f"Total results collected: {len(all_results)}")

    # Sort by csv_row_index to maintain original order
    all_results.sort(key=lambda x: x['csv_row_index'])

    # If resuming, merge with existing results
    if completed_indices:
        try:
            existing_df = pd.read_csv(args.output_csv)
            existing_results = existing_df.to_dict('records')
            all_results = existing_results + all_results
            all_results.sort(key=lambda x: x['csv_row_index'])
        except Exception as e:
            print(f"Could not merge with existing results: {e}")

    # Save to CSV
    print(f"Saving results to {args.output_csv}...")
    df = pd.DataFrame(all_results)

    # Reorder columns with progression info
    column_order = [
        'csv_row_index', 'subject_id', 'input_image', 'edited_image', 'edit_prompt',
        'progression_type', 'months_from_first_visit', 'is_first_visit',
        'psnr', 'mse', 'ssim', 'generated_image_path', 'comparison_image_path'
    ]
    df = df[[c for c in column_order if c in df.columns]]

    df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")

    # Helper function for formatting to 4 digits
    def format_value(x):
        """Format value to 4 total digits."""
        if x == 0 or np.isnan(x):
            return "0.00"

        rounded = round(x, 4)

        if abs(rounded) >= 100:
            # 100+: 123.4 (4 significant digits)
            return f"{round(rounded, 1):.1f}"
        elif abs(rounded) >= 10:
            # 10~99: 29.32 (4 significant digits)
            return f"{round(rounded, 2):.2f}"
        elif abs(rounded) >= 1:
            # 1~9: 6.811 (4 significant digits)
            return f"{round(rounded, 3):.3f}"
        else:
            # 0.xxxx: 0.0024 (4 significant digits)
            return f"{round(rounded, 4):.4f}"

    def format_metric(mean_val, std_val):
        """Format metric with mean ± std, both formatted to 4 digits."""
        return f"{format_value(mean_val)} ± {format_value(std_val)}"

    # Build results content for both printing and saving to file
    lines = []

    # Overall summary statistics
    lines.append("=" * 80)
    lines.append("Overall Summary Statistics")
    lines.append("=" * 80)
    lines.append(f"Total samples processed: {len(all_results)}")
    lines.append(f"PSNR:  {format_metric(df['psnr'].mean(), df['psnr'].std())}")
    lines.append(f"MSE:   {format_metric(df['mse'].mean(), df['mse'].std())}")
    lines.append(f"SSIM:  {format_metric(df['ssim'].mean(), df['ssim'].std())}")
    lines.append("")

    # Per-progression-type statistics
    if 'progression_type' in df.columns:
        lines.append("-" * 80)
        lines.append("Per-Progression-Type Statistics")
        lines.append("-" * 80)
        for prog_type in sorted(df['progression_type'].unique()):
            subset = df[df['progression_type'] == prog_type]
            lines.append("")
            lines.append(f"{prog_type} (n={len(subset)}):")
            lines.append(f"  PSNR:  {format_metric(subset['psnr'].mean(), subset['psnr'].std())}")
            lines.append(f"  MSE:   {format_metric(subset['mse'].mean(), subset['mse'].std())}")
            lines.append(f"  SSIM:  {format_metric(subset['ssim'].mean(), subset['ssim'].std())}")
        lines.append("")

    # Per-interval statistics
    if 'months_from_first_visit' in df.columns:
        lines.append("-" * 80)
        lines.append("Per-Interval Statistics (months from first visit)")
        lines.append("-" * 80)

        # Define interval bins: 0≤Interval<12, 12≤Interval<24, 24≤Interval<36, 36≤Interval
        def get_interval_label(months):
            if months < 12:
                return "0≤Interval<12"
            elif months < 24:
                return "12≤Interval<24"
            elif months < 36:
                return "24≤Interval<36"
            else:
                return "36≤Interval"

        df['interval'] = df['months_from_first_visit'].apply(get_interval_label)

        interval_order = ["0≤Interval<12", "12≤Interval<24", "24≤Interval<36", "36≤Interval"]
        for interval in interval_order:
            subset = df[df['interval'] == interval]
            if len(subset) > 0:
                lines.append("")
                lines.append(f"{interval} (n={len(subset)}):")
                lines.append(f"  PSNR:  {format_metric(subset['psnr'].mean(), subset['psnr'].std())}")
                lines.append(f"  MSE:   {format_metric(subset['mse'].mean(), subset['mse'].std())}")
                lines.append(f"  SSIM:  {format_metric(subset['ssim'].mean(), subset['ssim'].std())}")

    # Print to console
    print("\n" + "\n".join(lines))
    print("\n" + "=" * 80)

    # Save to results.md file
    results_md_path = os.path.join(args.output_dir, 'results.md')
    with open(results_md_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Results summary saved to {results_md_path}")


if __name__ == '__main__':
    main()
