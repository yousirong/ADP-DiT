import gc
import json
import os
import random
import sys
import time
import shutil
from functools import partial
from glob import glob
import os, logging, warnings
import csv
import re
from PIL import Image
import torchvision.transforms as transforms

# Enable PyTorch memory fragmentation management to prevent OOM from reserved memory
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Suppress all FutureWarning messages
warnings.filterwarnings("ignore", category=FutureWarning)

# Silence DeepSpeed's internal real_accelerator logger
logging.getLogger("deepspeed.runtime.accelerator.real_accelerator").setLevel(logging.WARNING)

# Silence TorchCheckpointEngine logger
logging.getLogger("TorchCheckpointEngine").setLevel(logging.WARNING)

# Optionally suppress the entire deepspeed logger to WARNING and above
logging.getLogger("deepspeed").setLevel(logging.WARNING)

import deepspeed
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
import torch.distributed as dist
from diffusers import DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler
from diffusers.models import AutoencoderKL
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer, logging as tf_logging
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from types import SimpleNamespace

from IndexKits.index_kits import ResolutionGroup
from IndexKits.index_kits.sampler import DistributedSamplerWithStartIndex, BlockDistributedSampler,DistributedRandomReplacementSampler
from adpdit.config import get_args
from adpdit.constants import VAE_EMA_PATH, TEXT_ENCODER, TOKENIZER, T5_ENCODER, SAMPLER_FACTORY, NEGATIVE_PROMPT
from adpdit.modules.text_encoder import T5Embedder
from adpdit.data_loader.arrow_load_stream import TextImageArrowStream
from adpdit.diffusion import create_diffusion
from adpdit.diffusion.gaussian_diffusion import _extract_into_tensor
from adpdit.diffusion.pipeline import StableDiffusionPipeline
from adpdit.ds_config import deepspeed_config_from_args
from adpdit.lr_scheduler import CosineAnnealingWarmupRestarts, add_tuning_arguments
from adpdit.modules.ema import EMA
from adpdit.modules.fp16_layers import Float16Module
from adpdit.modules.models import ADP_DIT_MODELS, ADPDiT
from adpdit.modules.posemb_layers import init_image_posemb
from adpdit.utils.tools import create_exp_folder, model_resume, get_trainable_params

# ============================================================================
# Validation Patients by Disease Progression Scenario
# ============================================================================
# Representative patient per scenario (first patient of each)
# - CNtoCN: Cognitive Normal maintained
# - CNtoMCI: Cognitive Normal -> Mild Cognitive Impairment
# - MCItoMCI: MCI maintained
# - MCItoAD: MCI -> Alzheimer Disease
# - ADtoAD: AD maintained
VALIDATION_PATIENTS_BY_SCENARIO = {
    'CNtoCN': '002_S_4213',
    'CNtoMCI': '002_S_4270',
    'MCItoMCI': '002_S_2010',
    'MCItoAD': '002_S_4521',
    'ADtoAD': '002_S_0729',
}


def create_validation_dataset_from_train(train_csv_path, validation_patients, output_dir, logger=None):
    """
    Extract validation patient samples from the train CSV and create a new Arrow file and JSON index.

    Args:
        train_csv_path: Path to the train CSV file (image_text.csv)
        validation_patients: Dict mapping scenario names to patient IDs
        output_dir: Output directory
        logger: Logger (optional)

    Returns:
        tuple: (validation_json_path, validation_csv_path, num_samples)
    """
    log = logger.info if logger else print

    # Create output directories
    val_dir = os.path.join(output_dir, 'validation_from_train')
    os.makedirs(val_dir, exist_ok=True)
    arrows_dir = os.path.join(val_dir, 'arrows')
    os.makedirs(arrows_dir, exist_ok=True)

    # Read CSV
    log(f"Loading train CSV: {train_csv_path}")
    df = pd.read_csv(train_csv_path)
    log(f"  Total samples in train: {len(df):,}")

    # Function to extract patient ID from path
    def extract_patient_id(path):
        match = re.search(r'(\d{3}_S_\d{4})', path)
        return match.group(1) if match else None

    df['patient_id'] = df['input_image'].apply(extract_patient_id)

    # Filter samples for validation patients
    patient_ids = list(validation_patients.values())
    val_df = df[df['patient_id'].isin(patient_ids)].copy()

    log(f"  Filtered validation samples: {len(val_df):,}")
    for scenario, patient_id in validation_patients.items():
        count = len(val_df[val_df['patient_id'] == patient_id])
        log(f"    {scenario} ({patient_id}): {count} samples")

    if len(val_df) == 0:
        log("WARNING: No validation samples found!")
        return None, None, 0

    # Drop patient_id column (not needed when saving to Arrow)
    val_df = val_df.drop(columns=['patient_id'])

    # Save validation CSV
    val_csv_path = os.path.join(val_dir, 'validation_samples.csv')
    val_df.to_csv(val_csv_path, index=False)
    log(f"  Saved validation CSV: {val_csv_path}")

    # Create Arrow files in the same format as the training Arrow files
    # Read images and store them in Arrow format
    data_root = os.path.dirname(os.path.dirname(train_csv_path))  # dataset/AD_meta

    records = []
    for _, row in val_df.iterrows():
        try:
            input_path = row['input_image']
            edited_path = row['edited_image']
            prompt = row['edit_prompt']

            # Resolve paths
            if input_path.startswith('./'):
                input_full = os.path.join(data_root, input_path[2:])
                edited_full = os.path.join(data_root, edited_path[2:])
            else:
                input_full = os.path.join(data_root, input_path)
                edited_full = os.path.join(data_root, edited_path)

            # Read images as binary
            with open(input_full, 'rb') as f:
                input_bytes = f.read()
            with open(edited_full, 'rb') as f:
                edited_bytes = f.read()

            # Check image dimensions
            img = Image.open(input_full)
            width, height = img.size

            # Generate MD5 hash (simple path-based approach)
            import hashlib
            input_md5 = hashlib.md5(input_path.encode()).hexdigest()
            edited_md5 = hashlib.md5(edited_path.encode()).hexdigest()

            records.append({
                'edit_prompt': prompt,
                'input_md5': input_md5,
                'edited_md5': edited_md5,
                'width': width,
                'height': height,
                'input_image': input_bytes,
                'edited_image': edited_bytes,
            })
        except Exception as e:
            log(f"  Warning: Failed to process {row['input_image']}: {e}")
            continue

    log(f"  Processed {len(records)} valid samples")

    # Save Arrow files
    if records:
        table = pa.Table.from_pydict({
            'edit_prompt': [r['edit_prompt'] for r in records],
            'input_md5': [r['input_md5'] for r in records],
            'edited_md5': [r['edited_md5'] for r in records],
            'width': [r['width'] for r in records],
            'height': [r['height'] for r in records],
            'input_image': [r['input_image'] for r in records],
            'edited_image': [r['edited_image'] for r in records],
        })

        arrow_path = os.path.join(arrows_dir, '00000.arrow')
        with pa.ipc.new_file(arrow_path, table.schema) as writer:
            writer.write_table(table)
        log(f"  Saved Arrow file: {arrow_path}")

        # Create JSON index file
        json_data = {
            "data_type": [
                "height>=256 (default=256)",
                "width>=256 (default=256)",
                "Validation subset from train"
            ],
            "config_file": "",
            "indices_file": "",
            "arrow_files": [
                f"./validation_from_train/arrows/00000.arrow"
            ],
            "cum_length": [len(records)]
        }

        json_path = os.path.join(val_dir, 'validation_index.json')
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        log(f"  Saved JSON index: {json_path}")

        return json_path, val_csv_path, len(records)

    return None, None, 0


@torch.no_grad()
def generate_samples(args, rank, logger, model, vae, text_encoder, text_encoder_t5,
                     tokenizer, tokenizer_t5, diffusion, freqs_cis_img, device,
                     train_steps, results_dir, validation_csv_path=None):
    """
    Generate sample images from the model during training (inference-style sampling with legacy fallback).

    Selects representative slices from the 5 validation patients (one per scenario).
    Loads from validation_csv_path if provided, otherwise falls back to the train CSV.
    Uses the current model without additional VRAM overhead.
    """
    if rank != 0:  # Run only on the main process
        return

    logger.info(f"=" * 60)
    logger.info(f"Generating samples at step {train_steps}...")
    logger.info(f"=" * 60)

    # Create sample output directory
    sample_dir = os.path.join(results_dir, 'samples', f'step_{train_steps:07d}')
    os.makedirs(sample_dir, exist_ok=True)

    # Prefer validation CSV; fall back to train CSV if not available
    if validation_csv_path and os.path.exists(validation_csv_path):
        csv_path = validation_csv_path
        logger.info(f"  Using validation CSV: {csv_path}")
    else:
        csv_path = getattr(args, 'train_csv_path', './dataset/AD_meta/csvfile/image_text.csv')
        logger.info(f"  Using train CSV: {csv_path}")

    if not os.path.exists(csv_path):
        logger.warning(f"CSV not found: {csv_path}")
        return

    # Use one representative patient per 5 scenarios
    samples_to_generate = []
    target_slice = 128  # Central slice

    # Load CSV
    df = pd.read_csv(csv_path)

    for scenario, patient_id in VALIDATION_PATIENTS_BY_SCENARIO.items():
        # Find samples for this patient
        patient_rows = df[df['input_image'].str.contains(patient_id)]

        if len(patient_rows) == 0:
            logger.warning(f"  No samples found for {scenario} ({patient_id})")
            continue

        # Find the sample closest to the target slice
        # Prefer progression data (input != target, containing "months from first visit")
        progression_rows = patient_rows[patient_rows['edit_prompt'].str.contains('months from first visit', case=False)]
        if len(progression_rows) > 0:
            patient_rows = progression_rows

        # Extract slice numbers and select the one closest to the target slice
        def get_slice_num(path):
            match = re.search(r'_(\d+)\.png', path)
            return int(match.group(1)) if match else 0

        patient_rows = patient_rows.copy()
        patient_rows['slice_num'] = patient_rows['input_image'].apply(get_slice_num)
        patient_rows['slice_diff'] = abs(patient_rows['slice_num'] - target_slice)
        best_row = patient_rows.loc[patient_rows['slice_diff'].idxmin()]

        samples_to_generate.append({
            'input_image': best_row['input_image'],
            'edited_image': best_row['edited_image'],
            'edit_prompt': best_row['edit_prompt'],
            'patient_id': patient_id,
            'scenario': scenario
        })

        logger.info(f"  {scenario}: {patient_id}, slice {best_row['slice_num']}")

    if not samples_to_generate:
        logger.warning("No matching samples found for generation")
        return

    logger.info(f"  Selected {len(samples_to_generate)} samples (5 scenarios)")

    # Switch model to eval mode
    if hasattr(model, 'module'):
        if hasattr(model.module, 'module'):
            eval_model = model.module.module
        else:
            eval_model = model.module
    else:
        eval_model = model

    was_training = eval_model.training
    eval_model.eval()

    # Determine actual dtype of each model component
    model_dtype = next(eval_model.parameters()).dtype
    vae_dtype = next(vae.parameters()).dtype
    text_encoder_dtype = next(text_encoder.parameters()).dtype
    text_encoder_t5_dtype = next(text_encoder_t5.parameters()).dtype
    logger.info(f"  Model dtype: {model_dtype}")
    logger.info(f"  VAE dtype: {vae_dtype}")
    logger.info(f"  CLIP dtype: {text_encoder_dtype}")
    logger.info(f"  T5 dtype: {text_encoder_t5_dtype}")

    # Inference-style sampling pipeline (CFG + scheduler)
    pipeline = None
    try:
        sampler_key = getattr(args, "sampler", "ddpm")
        sampler_cfg = SAMPLER_FACTORY.get(sampler_key, SAMPLER_FACTORY["ddpm"])
        scheduler_classes = {
            "DDPMScheduler": DDPMScheduler,
            "DDIMScheduler": DDIMScheduler,
            "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
        }
        scheduler_class = scheduler_classes.get(sampler_cfg["scheduler"])
        if scheduler_class is None:
            raise ValueError(f"Unknown scheduler class: {sampler_cfg['scheduler']}")

        scheduler_kwargs = sampler_cfg["kwargs"].copy()
        if getattr(args, "predict_type", None) is not None:
            scheduler_kwargs["prediction_type"] = args.predict_type
        if getattr(args, "noise_schedule", None) is not None:
            scheduler_kwargs["beta_schedule"] = args.noise_schedule
        if getattr(args, "beta_start", None) is not None:
            scheduler_kwargs["beta_start"] = args.beta_start
        if getattr(args, "beta_end", None) is not None:
            scheduler_kwargs["beta_end"] = args.beta_end
        if sampler_cfg["scheduler"] == "DPMSolverMultistepScheduler" and "config" in sampler_cfg:
            scheduler_kwargs.update(sampler_cfg["config"])

        scheduler = scheduler_class(**scheduler_kwargs)
        t5_embedder = SimpleNamespace(
            model=text_encoder_t5,
            tokenizer=tokenizer_t5,
            max_length=args.text_len_t5,
        )
        pipeline = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=eval_model,
            scheduler=scheduler,
            embedder_t5=t5_embedder,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            infer_mode=getattr(args, "infer_mode", "torch"),
        )
        pipeline.set_progress_bar_config(disable=True)
        pipeline_device = f"cuda:{device}" if isinstance(device, int) else device
        pipeline.to(pipeline_device)
        logger.info(f"  Sample pipeline: {sampler_cfg['name']} ({sampler_key})")
    except Exception as e:
        logger.warning(f"  ⚠️ Failed to build inference pipeline, falling back to legacy sampling: {e}")

    # Set up image transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    data_root = os.path.dirname(os.path.dirname(csv_path))

    try:
        for idx, sample in enumerate(samples_to_generate):
            try:
                # Resolve image paths
                input_path = sample['input_image']
                edited_path = sample['edited_image']

                if input_path.startswith('./images/'):
                    input_path = os.path.join(data_root, input_path[2:])
                    edited_path = os.path.join(data_root, edited_path[2:])

                # Load images
                input_img = Image.open(input_path).convert('RGB')
                target_img = Image.open(edited_path).convert('RGB')

                input_tensor = transform(input_img).unsqueeze(0).to(device)

                # VAE encoding (cast to VAE dtype)
                vae_scaling_factor = vae.config.scaling_factor
                with torch.no_grad():
                    input_latents = vae.encode(input_tensor.to(vae_dtype)).latent_dist.sample().mul_(vae_scaling_factor)

                prompt = sample['edit_prompt']

                # Positional embedding
                reso = "256x256"
                if reso not in freqs_cis_img:
                    reso = list(freqs_cis_img.keys())[0]
                cos_cis_img, sin_cis_img = freqs_cis_img[reso]

                # Sampling params
                num_steps = args.sample_infer_steps
                strength = args.sample_strength
                cfg_scale = args.sample_cfg_scale

                if pipeline is not None:
                    negative_prompt = getattr(args, "negative", NEGATIVE_PROMPT)
                    output = pipeline(
                        height=input_tensor.shape[2],
                        width=input_tensor.shape[3],
                        prompt=prompt,
                        negative_prompt=negative_prompt if cfg_scale > 1.0 else None,
                        num_inference_steps=num_steps,
                        guidance_scale=cfg_scale,
                        strength=strength,
                        image_latents=input_latents,
                        input_image=input_latents,
                        freqs_cis_img=(cos_cis_img, sin_cis_img),
                        return_dict=True,
                        use_fp16=bool(getattr(args, "use_fp16", False) or getattr(args, "extra_fp16", False)),
                        learn_sigma=getattr(args, "learn_sigma", True),
                        progress=False,
                    )
                    output_img = output.images[0].convert('L')
                    del output
                else:
                    # Text encoding (CLIP)
                    text_inputs = tokenizer(
                        prompt, padding="max_length", max_length=args.text_len,
                        truncation=True, return_tensors="pt"
                    )
                    text_embedding = text_inputs.input_ids.to(device)
                    text_embedding_mask = text_inputs.attention_mask.to(device)
                    with torch.no_grad():
                        encoder_hidden_states = text_encoder(text_embedding, attention_mask=text_embedding_mask)[0]

                    # Text encoding (T5)
                    text_inputs_t5 = tokenizer_t5(
                        prompt, padding="max_length", max_length=args.text_len_t5,
                        truncation=True, return_tensors="pt"
                    )
                    text_embedding_t5 = text_inputs_t5.input_ids.to(device)
                    text_embedding_mask_t5 = text_inputs_t5.attention_mask.to(device)
                    with torch.no_grad():
                        output_t5 = text_encoder_t5(
                            input_ids=text_embedding_t5,
                            attention_mask=text_embedding_mask_t5 if T5_ENCODER['attention_mask'] else None,
                            output_hidden_states=True
                        )
                        encoder_hidden_states_t5 = output_t5['hidden_states'][T5_ENCODER['layer_index']].clone()

                    # Configure timesteps
                    init_timestep = min(int(num_steps * strength), num_steps)
                    t_start = max(num_steps - init_timestep, 0)

                    timesteps = list(range(diffusion.num_timesteps))[::-1]
                    step_ratio = diffusion.num_timesteps // num_steps
                    timesteps = timesteps[::step_ratio][:num_steps]
                    timesteps = timesteps[t_start:]

                    # Add noise
                    if strength < 1.0:
                        noise = torch.randn_like(input_latents)
                        z = diffusion.q_sample(input_latents, torch.tensor([timesteps[0]], device=device), noise=noise)
                    else:
                        z = torch.randn_like(input_latents)

                    # Cast all tensors to match model dtype
                    z = z.to(model_dtype)
                    input_latents = input_latents.to(model_dtype)
                    encoder_hidden_states = encoder_hidden_states.to(model_dtype)
                    encoder_hidden_states_t5 = encoder_hidden_states_t5.to(model_dtype)
                    cos_cis_img = cos_cis_img.to(model_dtype)
                    sin_cis_img = sin_cis_img.to(model_dtype)

                    # Model kwargs
                    model_kwargs = dict(
                        encoder_hidden_states=encoder_hidden_states,
                        text_embedding_mask=text_embedding_mask,
                        encoder_hidden_states_t5=encoder_hidden_states_t5,
                        text_embedding_mask_t5=text_embedding_mask_t5,
                        image_meta_size=None,
                        style=None,
                        cos_cis_img=cos_cis_img,
                        sin_cis_img=sin_cis_img,
                        input_image=input_latents,
                        auxiliary_data=None,
                    )

                    # Sampling loop
                    with torch.no_grad():
                        if args.predict_type == 'sample':
                            # Direct X0 prediction mode: multi-step DDIM sampling
                            for t in timesteps:
                                t_batch = torch.tensor([t], device=device)
                                if z.dtype != model_dtype:
                                    z = z.to(model_dtype)
                                out = diffusion.ddim_sample(
                                    model=eval_model,
                                    x=z,
                                    t=t_batch,
                                    clip_denoised=False,
                                    model_kwargs=model_kwargs,
                                    eta=0.0,
                                )
                                z = out["sample"].to(model_dtype)
                        else:
                            # Epsilon or v-prediction mode: DDIM sampling
                            for t in timesteps:
                                t_batch = torch.tensor([t], device=device)

                                # Model prediction
                                out_dict = eval_model(z, t_batch, **model_kwargs)
                                model_output = out_dict['x'] if isinstance(out_dict, dict) else out_dict

                                if model_output.shape[1] == 8:
                                    model_output = model_output[:, :4]

                                # DDIM step - recover X0 from epsilon or v prediction
                                alpha_t = _extract_into_tensor(diffusion.alphas_cumprod, t_batch, z.shape).to(model_dtype)
                                sqrt_alpha_t = torch.sqrt(alpha_t)
                                sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
                                if args.predict_type == 'v_prediction':
                                    pred_x0 = sqrt_alpha_t * z - sqrt_one_minus_alpha_t * model_output
                                else:
                                    pred_x0 = (z - sqrt_one_minus_alpha_t * model_output) / sqrt_alpha_t

                                # Next timestep
                                t_idx = timesteps.index(t)
                                if t_idx < len(timesteps) - 1:
                                    t_next = timesteps[t_idx + 1]
                                    alpha_next = _extract_into_tensor(
                                        diffusion.alphas_cumprod,
                                        torch.tensor([t_next], device=device),
                                        z.shape
                                    ).to(model_dtype)
                                    # DDIM update (eta=0, deterministic)
                                    z = torch.sqrt(alpha_next) * pred_x0 + torch.sqrt(1 - alpha_next) * \
                                        (z - torch.sqrt(alpha_t) * pred_x0) / torch.sqrt(1 - alpha_t)
                                else:
                                    z = pred_x0

                    # VAE decoding (cast to VAE dtype)
                    output_latents = z / vae_scaling_factor
                    with torch.no_grad():
                        output_tensor = vae.decode(output_latents.to(vae_dtype)).sample

                    # Tensor to PIL Image
                    output_tensor = (output_tensor.clamp(-1, 1) + 1) / 2
                    output_np = (output_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    output_img = Image.fromarray(output_np).convert('L')

                # Create comparison image (Input | Target | Output)
                input_gray = input_img.convert('L').resize((256, 256))
                target_gray = target_img.convert('L').resize((256, 256))
                output_gray = output_img.resize((256, 256))

                # 3-column comparison image
                gap = 5
                comparison = Image.new('L', (256 * 3 + gap * 2, 256 + 30), 255)

                # Place images
                comparison.paste(input_gray, (0, 30))
                comparison.paste(target_gray, (256 + gap, 30))
                comparison.paste(output_gray, (256 * 2 + gap * 2, 30))

                # Add labels
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(comparison)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
                except:
                    font = ImageFont.load_default()

                labels = ['Input', 'Target', 'Output']
                for i, label in enumerate(labels):
                    x = i * (256 + gap) + 128
                    draw.text((x - 20, 10), label, fill=0, font=font)

                # Save with scenario information
                patient_id = sample['patient_id']
                scenario = sample.get('scenario', f'sample{idx}')
                filename = f"{scenario}_{patient_id}.png"
                comparison.save(os.path.join(sample_dir, filename))

                logger.info(f"  Generated: {filename} ({scenario})")

                # Memory cleanup
                del input_tensor, input_latents
                if pipeline is None:
                    del z, output_latents, output_tensor
                torch.cuda.empty_cache()

            except Exception as e:
                import traceback
                logger.error(f"  ❌ Failed to generate sample {idx}: {e}")
                logger.error(f"  Traceback: {traceback.format_exc()}")
                continue

    except Exception as e:
        logger.error(f"Sample generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    finally:
        # Restore model to training mode
        if was_training:
            eval_model.train()

        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()

    logger.info(f"Sample generation complete. Saved to: {sample_dir}")
    logger.info(f"=" * 60)


def deepspeed_initialize(args, logger, model, opt, deepspeed_config):
    logger.info(f"Initialize deepspeed...")
    logger.info(f"    Using deepspeed optimizer")

    def get_learning_rate_scheduler(optimizer):
        """Use CosineAnnealingWarmupRestarts for the scheduler"""
        return CosineAnnealingWarmupRestarts(
            optimizer=optimizer,
            total_steps=args.max_training_steps,
            num_cycles=args.num_cycles,
            warmup_ratio=args.warmup_ratio,
            max_lr=args.max_lr,
            min_lr=args.min_lr,
            t_mult=getattr(args, 't_mult', 2.0),  # Cycle length multiplier
            gamma=getattr(args, 'gamma', 0.5),    # max_lr decay factor (halved each restart)
        )

    logger.info(f"    Building scheduler with total_steps={args.max_training_steps}, num_cycles={args.num_cycles}, warmup_ratio={args.warmup_ratio}, t_mult={getattr(args, 't_mult', 2.0)}, gamma={getattr(args, 'gamma', 0.5)}")
    model, opt, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=get_trainable_params(model),
        config_params=deepspeed_config,
        args=args,
        lr_scheduler=partial(get_learning_rate_scheduler)
    )
    return model, opt, scheduler


@torch.no_grad()
def run_validation(args, rank, logger, model, valid_loader, device, vae,
                   text_encoder, text_encoder_t5, freqs_cis_img,
                   train_steps, writer):
    """
    Run distributed classification validation to detect overfitting.

    Memory Management Strategy:
    - Distribute validation batches across all GPUs
    - Each GPU processes its assigned batches independently
    - Clear CUDA cache before validation (prevent accumulation)
    - Clean up batch tensors after each forward pass
    - Aggressive cleanup between batches
    - Final cleanup before returning to restore memory for training

    Returns:
        tuple: (accuracy, per_class_metrics_dict)
    """
    logger.info(f"Running distributed validation at step {train_steps}... (Rank {rank})")

    # 🔧 MEMORY CLEANUP: Clear cache before validation starts
    logger.info(f"🧹 Clearing CUDA cache before validation (prevent memory accumulation)...")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    # Set model to eval mode
    # Extract the actual model from DeepSpeed wrapper
    if hasattr(model, 'module'):
        if hasattr(model.module, 'module'):
            # FP16 wrapped: model.module.module
            eval_model = model.module.module
        else:
            # No FP16: model.module
            eval_model = model.module
    else:
        eval_model = model

    was_training = eval_model.training
    eval_model.eval()

    logger.info(f"Validation model type: {type(eval_model).__name__} (rank {rank})")

    all_preds = []
    all_labels = []
    total_samples = 0

    logger.info(f"Starting validation loop for rank {rank}...")

    for batch_idx, batch in enumerate(valid_loader):
        try:
            # Prepare inputs (same as training)
            latents, model_kwargs, cls_labels = prepare_model_inputs(
                args, batch, device, vae, text_encoder,
                text_encoder_t5, freqs_cis_img
            )

            if cls_labels is None:
                if batch_idx == 0:
                    logger.warning(f"Rank {rank}: No cls_labels in batch. Validation dataset may not have labels.")
                continue

            # Create dummy timesteps (use t=0 for validation)
            t = torch.zeros(latents.shape[0], dtype=torch.long, device=device)

            # ⚡ MEMORY OPTIMIZATION: Use no_grad() to prevent gradient computation during validation
            with torch.no_grad():
                # Forward pass to get classification logits
                # Pass return_dict=True to get cls_logits from the model
                try:
                    # Try with return_dict parameter
                    output = eval_model(latents, t, return_dict=True, **model_kwargs)
                except TypeError:
                    # If return_dict not supported, try without it
                    logger.warning("Model does not support return_dict parameter, trying without it")
                    output = eval_model(latents, t, **model_kwargs)

                # Extract classification logits from model output
                cls_logits = None
                if isinstance(output, dict) and 'cls_logits' in output:
                    cls_logits = output['cls_logits'].detach()  # Detach to save memory
                    noise_pred = output.get('x', None)
                elif isinstance(output, dict):
                    logger.warning(f"Model returned dict but no 'cls_logits' key. Available keys: {list(output.keys())}")
                    break
                else:
                    logger.warning(f"Model did not return dict with cls_logits. Output type: {type(output)}")
                    break

                if cls_logits is None:
                    logger.warning("Failed to extract cls_logits from model output")
                    break

                # Get predictions
                _, predicted = torch.max(cls_logits.data, 1)

                # Collect predictions and labels (convert to CPU immediately to free GPU memory)
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(cls_labels.cpu().numpy().tolist())
                total_samples += cls_labels.size(0)

            # 🔧 MEMORY CLEANUP: Delete batch-specific tensors to prevent accumulation
            del latents, model_kwargs, cls_labels, cls_logits, output, t, predicted
            if 'noise_pred' in locals():
                del noise_pred

            # Every 5 batches, do aggressive cleanup to prevent memory buildup
            if (batch_idx + 1) % 5 == 0:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
                logger.debug(f"  Validation batch {batch_idx + 1}: Aggressive memory cleanup")

        except Exception as e:
            logger.error(f"Validation batch {batch_idx} failed: {e}")
            continue

    # 🔧 MEMORY CLEANUP: Aggressive cleanup after validation loop to restore memory
    logger.info(f"🧹 Cleaning up validation memory (after {total_samples} samples processed on rank {rank})...")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    # 🔄 Gather predictions from all ranks
    logger.info(f"Gathering validation results from all ranks... (local: {len(all_preds)} samples on rank {rank})")

    # Convert to tensors for distributed gathering
    local_preds = torch.tensor(all_preds, dtype=torch.long, device='cpu')
    local_labels = torch.tensor(all_labels, dtype=torch.long, device='cpu')
    local_sample_count = torch.tensor([total_samples], dtype=torch.long, device='cpu')

    # Initialize world_size for logging
    world_size = 1

    # Gather from all ranks (only rank 0 collects final results)
    if dist.is_available() and dist.is_initialized():
        # Get world size
        world_size = dist.get_world_size()

        if rank == 0:
            # Rank 0 collects all predictions
            all_gathered_preds = [None] * world_size
            all_gathered_labels = [None] * world_size
            all_gathered_counts = [None] * world_size

            dist.gather_object(local_preds, all_gathered_preds, dst=0)
            dist.gather_object(local_labels, all_gathered_labels, dst=0)
            dist.gather_object(local_sample_count, all_gathered_counts, dst=0)

            # Concatenate results from all ranks
            y_pred = np.concatenate([p.numpy() if isinstance(p, torch.Tensor) else p for p in all_gathered_preds])
            y_true = np.concatenate([l.numpy() if isinstance(l, torch.Tensor) else l for l in all_gathered_labels])
            total_samples = sum([c.item() if isinstance(c, torch.Tensor) else c[0] for c in all_gathered_counts])

            logger.info(f"✅ Gathered validation results from {world_size} GPUs: {total_samples} total samples")
        else:
            # Non-rank-0 GPUs just send their data
            dist.gather_object(local_preds, None, dst=0)
            dist.gather_object(local_labels, None, dst=0)
            dist.gather_object(local_sample_count, None, dst=0)
            return None, None
    else:
        # Single GPU case
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

    # Calculate metrics (only on rank 0)
    if rank == 0:
        if len(y_pred) == 0:
            logger.warning("No validation samples processed!")
            accuracy, per_class_metrics = 0.0, {}
        else:
            # Overall accuracy
            accuracy = 100.0 * np.sum(y_true == y_pred) / len(y_true)

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

            # Per-class metrics
            per_class_metrics = {}
            for class_idx, class_name in enumerate(['CN', 'MCI', 'AD']):
                tp = cm[class_idx, class_idx]
                fn = np.sum(cm[class_idx, :]) - tp
                fp = np.sum(cm[:, class_idx]) - tp
                tn = np.sum(cm) - tp - fn - fp

                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                f1 = 2 * (precision * sensitivity) / (precision + sensitivity) \
                     if (precision + sensitivity) > 0 else 0.0

                per_class_metrics[class_name] = {
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'precision': precision,
                    'f1': f1,
                    'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
                }

                # Log to TensorBoard
                if writer is not None:
                    writer.add_scalar(f'Validation/{class_name}/Sensitivity', sensitivity, train_steps)
                    writer.add_scalar(f'Validation/{class_name}/Specificity', specificity, train_steps)
                    writer.add_scalar(f'Validation/{class_name}/Precision', precision, train_steps)
                    writer.add_scalar(f'Validation/{class_name}/F1Score', f1, train_steps)

            # Log overall accuracy
            if writer is not None:
                writer.add_scalar('Validation/Accuracy', accuracy, train_steps)

            logger.info(f"Validation Results (step {train_steps}):")
            logger.info(f"  Overall Accuracy: {accuracy:.2f}% ({total_samples} samples from {world_size} GPUs)")
            for class_name, metrics in per_class_metrics.items():
                logger.info(f"  {class_name}: Sens={metrics['sensitivity']:.4f}, "
                           f"Spec={metrics['specificity']:.4f}, "
                           f"Prec={metrics['precision']:.4f}, F1={metrics['f1']:.4f}")

    # Restore training mode (all ranks)
    if was_training:
        eval_model.train()

    # 🔧 MEMORY CLEANUP: Final cleanup before returning to training
    if rank == 0:
        logger.info(f"🧹 Final cleanup before returning to training...")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    if rank == 0:
        logger.info(f"✅ Validation complete. Memory restored for training (step {train_steps}).")
        return accuracy, per_class_metrics
    else:
        return None, None


@torch.no_grad()
def run_test_inference(args, rank, logger, model, ema, vae, text_encoder, text_encoder_t5, tokenizer, tokenizer_t5,
                       diffusion, freqs_cis_img, device, epoch, train_steps, writer):
    """
    Run inference on a specific test sample and log to TensorBoard
    """
    if rank != 0:  # Only run on main process
        return

    # Inference configuration (DDPM with CFG)
    # Use fewer steps to reduce memory usage during training
    num_inference_steps = 10  # Minimal steps for testing
    cfg_scale = 3.5
    strength = 1.0
    negative_prompt = "noise, blur, anatomically incorrect, shrinking ventricles, cortical thickening, non-progressive atrophy, temporal inconsistency, non-AD patterns"

    # Test sample configuration
    test_input_path = "./dataset/AD_meta/test_images/input_image/003465.png"
    test_edited_path = "./dataset/AD_meta/test_images/edited_image/003465.png"
    test_prompt = "Alzheimer Disease, Female, 75.20 years old, 50 months from first visit, slice 133, CDRSB 4.5 ADAS11 21.0 ADAS13 33.0 ADASQ4 10.0 MMSE 21.0 RAVLT_immediate 22.0 RAVLT_learning 4.0 RAVLT_forgetting 6.0 RAVLT_perc_forgetting 100.0 LDELTOTAL 0.0 TRABSCOR 185.0 FAQ 24.0 MOCA 13.0"

    try:
        # Clear cache before inference to free up memory
        logger.info(f"Clearing CUDA cache before inference...")
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        # Move text encoders and VAE to CPU temporarily to save GPU memory
        text_encoder_device = next(text_encoder.parameters()).device
        text_encoder_t5_device = next(text_encoder_t5.parameters()).device
        vae_device = next(vae.parameters()).device
        text_encoder.cpu()
        text_encoder_t5.cpu()
        vae.cpu()
        torch.cuda.empty_cache()

        logger.info(f"Running test inference (DDPM, CFG={cfg_scale}, steps={num_inference_steps}, strength={strength})...")

        # Load and preprocess input image
        input_image = Image.open(test_input_path).convert('RGB')
        edited_image = Image.open(test_edited_path).convert('RGB')

        # Resize to smaller size to save memory (256x256 instead of original size)
        target_size = 256
        input_image = input_image.resize((target_size, target_size), Image.LANCZOS)
        edited_image = edited_image.resize((target_size, target_size), Image.LANCZOS)

        # Get image size
        img_width, img_height = target_size, target_size

        # Transform to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        input_tensor = transform(input_image).unsqueeze(0).to(device)
        edited_tensor = transform(edited_image).unsqueeze(0).to(device)

        if args.extra_fp16:
            input_tensor = input_tensor.half()
            edited_tensor = edited_tensor.half()

        # Move text encoders back to GPU for encoding
        text_encoder.to(device)
        text_encoder_t5.to(device)

        # ==== Encode POSITIVE prompt (CLIP) ====
        text_inputs_pos = tokenizer(
            test_prompt,
            padding="max_length",
            max_length=args.text_len,
            truncation=True,
            return_tensors="pt"
        )
        text_embedding_pos = text_inputs_pos.input_ids.to(device)
        text_embedding_mask_pos = text_inputs_pos.attention_mask.to(device)

        with torch.no_grad():
            encoder_hidden_states_pos = text_encoder(
                text_embedding_pos,
                attention_mask=text_embedding_mask_pos,
            )[0]

        # Encode POSITIVE prompt (T5)
        text_inputs_t5_pos = tokenizer_t5(
            test_prompt,
            padding="max_length",
            max_length=args.text_len_t5,
            truncation=True,
            return_tensors="pt"
        )
        text_embedding_t5_pos = text_inputs_t5_pos.input_ids.to(device)
        text_embedding_mask_t5_pos = text_inputs_t5_pos.attention_mask.to(device)

        with torch.no_grad():
            output_t5_pos = text_encoder_t5(
                input_ids=text_embedding_t5_pos,
                attention_mask=text_embedding_mask_t5_pos if T5_ENCODER['attention_mask'] else None,
                output_hidden_states=True
            )
            encoder_hidden_states_t5_pos = output_t5_pos['hidden_states'][T5_ENCODER['layer_index']].clone()

        # ==== Encode NEGATIVE prompt (CLIP) ====
        text_inputs_neg = tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=args.text_len,
            truncation=True,
            return_tensors="pt"
        )
        text_embedding_neg = text_inputs_neg.input_ids.to(device)
        text_embedding_mask_neg = text_inputs_neg.attention_mask.to(device)

        with torch.no_grad():
            encoder_hidden_states_neg = text_encoder(
                text_embedding_neg,
                attention_mask=text_embedding_mask_neg,
            )[0]

        # Encode NEGATIVE prompt (T5)
        text_inputs_t5_neg = tokenizer_t5(
            negative_prompt,
            padding="max_length",
            max_length=args.text_len_t5,
            truncation=True,
            return_tensors="pt"
        )
        text_embedding_t5_neg = text_inputs_t5_neg.input_ids.to(device)
        text_embedding_mask_t5_neg = text_inputs_t5_neg.attention_mask.to(device)

        with torch.no_grad():
            output_t5_neg = text_encoder_t5(
                input_ids=text_embedding_t5_neg,
                attention_mask=text_embedding_mask_t5_neg if T5_ENCODER['attention_mask'] else None,
                output_hidden_states=True
            )
            encoder_hidden_states_t5_neg = output_t5_neg['hidden_states'][T5_ENCODER['layer_index']].clone()

        # Move text encoders back to CPU to save memory
        text_encoder.cpu()
        text_encoder_t5.cpu()
        torch.cuda.empty_cache()

        # Move VAE to GPU for encoding
        vae.to(device)

        # Encode input image to latent
        vae_scaling_factor = vae.config.scaling_factor
        input_latents = vae.encode(input_tensor).latent_dist.sample().mul_(vae_scaling_factor)

        # Move VAE back to CPU to save memory
        vae.cpu()
        torch.cuda.empty_cache()

        # Prepare positional embeddings
        reso = f"{img_height}x{img_width}"
        if reso not in freqs_cis_img:
            # Use closest resolution
            reso = list(freqs_cis_img.keys())[0]
            logger.warning(f"Resolution {img_height}x{img_width} not found, using {reso}")
        cos_cis_img, sin_cis_img = freqs_cis_img[reso]

        # Use EMA model if available, otherwise use current model
        inference_model = ema.ema_model if (args.use_ema and ema is not None) else model.module
        if args.use_fp16 and not args.use_ema:
            inference_model = model.module.module

        inference_model.eval()

        # DDPM sampling with CFG
        latent_shape = input_latents.shape

        # Calculate starting timestep based on strength
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)

        # Create timesteps (reverse order for denoising)
        timesteps = list(range(diffusion.num_timesteps))[::-1]
        # Select subset based on num_inference_steps
        step_ratio = diffusion.num_timesteps // num_inference_steps
        timesteps = timesteps[::step_ratio][:num_inference_steps]

        # Apply strength by starting from t_start
        timesteps = timesteps[t_start:]

        # Add noise to input latents based on strength
        if strength < 1.0:
            noise = torch.randn_like(input_latents)
            z = diffusion.q_sample(input_latents, torch.tensor([timesteps[0]], device=device), noise=noise)
        else:
            # strength = 1.0: start from pure noise
            z = torch.randn(latent_shape, device=device)

        if args.extra_fp16:
            z = z.half()

        # DDPM sampling loop with CFG
        for i, t in enumerate(timesteps):
            t_batch = torch.tensor([t], device=device)

            # Prepare unconditional kwargs
            model_kwargs_uncond = dict(
                encoder_hidden_states=encoder_hidden_states_neg,
                text_embedding_mask=text_embedding_mask_neg,
                encoder_hidden_states_t5=encoder_hidden_states_t5_neg,
                text_embedding_mask_t5=text_embedding_mask_t5_neg,
                image_meta_size=None,
                style=None,
                cos_cis_img=cos_cis_img,
                sin_cis_img=sin_cis_img,
                input_image=input_latents,
                auxiliary_data=None,
            )

            # Prepare conditional kwargs
            model_kwargs_cond = dict(
                encoder_hidden_states=encoder_hidden_states_pos,
                text_embedding_mask=text_embedding_mask_pos,
                encoder_hidden_states_t5=encoder_hidden_states_t5_pos,
                text_embedding_mask_t5=text_embedding_mask_t5_pos,
                image_meta_size=None,
                style=None,
                cos_cis_img=cos_cis_img,
                sin_cis_img=sin_cis_img,
                input_image=input_latents,
                auxiliary_data=None,
            )

            # Predict noise unconditionally
            noise_pred_uncond = inference_model(z, t_batch, **model_kwargs_uncond)

            # Predict noise conditionally
            noise_pred_cond = inference_model(z, t_batch, **model_kwargs_cond)

            # Apply CFG
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

            # Clean up intermediate outputs
            del noise_pred_uncond, noise_pred_cond

            # Manually compute DDPM step with CFG-adjusted noise prediction
            # Predict x0 from noise
            alpha_t = diffusion._extract_into_tensor(diffusion.alphas_cumprod, t_batch, z.shape)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)

            if args.predict_type == 'epsilon':
                pred_x0 = (z - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
            else:
                pred_x0 = noise_pred

            # Clip prediction
            pred_x0 = torch.clamp(pred_x0, -1, 1)

            # Compute mean for previous timestep
            if t > 0:
                # Compute posterior mean
                posterior_mean = (
                    diffusion._extract_into_tensor(diffusion.posterior_mean_coef1, t_batch, z.shape) * pred_x0
                    + diffusion._extract_into_tensor(diffusion.posterior_mean_coef2, t_batch, z.shape) * z
                )

                # Get model variance
                posterior_variance = diffusion._extract_into_tensor(diffusion.posterior_variance, t_batch, z.shape)

                # Add noise for DDPM (not DDIM)
                noise = torch.randn_like(z)
                z = posterior_mean + torch.sqrt(posterior_variance) * noise

                # Clean up
                del posterior_mean, posterior_variance, noise
            else:
                z = pred_x0

            del noise_pred, pred_x0, alpha_t, sqrt_alpha_t, sqrt_one_minus_alpha_t

        # Move VAE back to GPU for decoding
        vae.to(device)

        # Decode latent to image
        generated_latents = z / vae_scaling_factor
        with torch.no_grad():
            generated_image = vae.decode(generated_latents).sample

        # Move VAE back to CPU
        vae.cpu()
        torch.cuda.empty_cache()

        # Denormalize images for visualization
        def denormalize(tensor):
            return (tensor * 0.5 + 0.5).clamp(0, 1)

        input_vis = denormalize(input_tensor)
        edited_vis = denormalize(edited_tensor)
        generated_vis = denormalize(generated_image)

        # Log images to TensorBoard
        if writer is not None:
            writer.add_image('Test_Inference/Input', input_vis[0], train_steps)
            writer.add_image('Test_Inference/GroundTruth', edited_vis[0], train_steps)
            writer.add_image('Test_Inference/Generated', generated_vis[0], train_steps)

            # Create comparison grid
            comparison = torch.cat([input_vis[0], edited_vis[0], generated_vis[0]], dim=2)  # Concatenate horizontally
            writer.add_image('Test_Inference/Comparison', comparison, train_steps)

        logger.info(f"Test inference completed (DDPM, CFG={cfg_scale}, steps={num_inference_steps}, step={train_steps})")

        # Set model back to train mode
        if not args.use_ema:
            inference_model.train()

        # Clean up memory aggressively
        del z, generated_latents, generated_image
        del input_latents, input_tensor, edited_tensor
        del encoder_hidden_states_pos, encoder_hidden_states_neg
        del encoder_hidden_states_t5_pos, encoder_hidden_states_t5_neg
        del input_vis, edited_vis, generated_vis, comparison

        # Restore text encoders and VAE to original device
        text_encoder.to(text_encoder_device)
        text_encoder_t5.to(text_encoder_t5_device)
        vae.to(vae_device)

        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        logger.info(f"Memory cleaned after inference")

    except Exception as e:
        logger.error(f"Test inference failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

        # Restore text encoders and VAE to original device even on error
        try:
            text_encoder.to(text_encoder_device)
            text_encoder_t5.to(text_encoder_t5_device)
            vae.to(vae_device)
        except:
            pass

        # Clean up memory even on error
        import gc
        gc.collect()
        torch.cuda.empty_cache()


def save_checkpoint(args, rank, logger, model, ema, epoch, train_steps, checkpoint_dir, by='step'):
    def save_lora_weight(checkpoint_dir, client_state, tag=f"{train_steps:07d}.pt"):
        cur_ckpt_save_dir = f"{checkpoint_dir}/{tag}"
        if rank == 0:
            if args.use_fp16:
                model.module.module.save_pretrained(cur_ckpt_save_dir)
            else:
                model.module.save_pretrained(cur_ckpt_save_dir)

    def save_model_weight(client_state, tag):
        checkpoint_path = f"{checkpoint_dir}/{tag}"
        try:
            if args.training_parts == "lora":
                save_lora_weight(checkpoint_dir, client_state, tag=tag)
            else:
                model.save_checkpoint(checkpoint_dir, client_state=client_state, tag=tag)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Saved failed to {checkpoint_path}. {type(e)}: {e}")
            return False, ''
        return True, checkpoint_path

    client_state = {
        "steps": train_steps,
        "epoch": epoch,
        "args": args
    }
    if ema is not None:
        client_state['ema'] = ema.state_dict()

    # Save model weights by epoch or step
    dst_paths = []
    if by == 'epoch':
        tag = f"e{epoch:04d}.pt"
        dst_paths.append(save_model_weight(client_state, tag))
    elif by == 'step':
        if train_steps % args.ckpt_every == 0:
            tag = f"{train_steps:07d}.pt"
            dst_paths.append(save_model_weight(client_state, tag))
        if train_steps % args.ckpt_latest_every == 0 or train_steps == args.max_training_steps:
            tag = "latest.pt"
            dst_paths.append(save_model_weight(client_state, tag))
    elif by == 'final':
        tag = "final.pt"
        dst_paths.append(save_model_weight(client_state, tag))
    elif by == 'emergency':
        tag = f"emergency_step{train_steps:07d}.pt"
        dst_paths.append(save_model_weight(client_state, tag))
    else:
        if by not in ['epoch', 'step', 'final', 'best', 'emergency']:
            raise ValueError(f"Unknown save checkpoint method: {by}")

    saved = any([state for state, _ in dst_paths])
    if not saved:
        return False

    # Maybe clear optimizer states
    if not args.save_optimizer_state:
        dist.barrier()
        if rank == 0 and len(dst_paths) > 0:
            # Delete optimizer states to avoid occupying too much disk space.
            for state, dst_path in dst_paths:
                if state:
                    for opt_state_path in glob(f"{dst_path}/zero_*_optim_states.pt"):
                        os.remove(opt_state_path)

    return True


@torch.no_grad()
def prepare_model_inputs(args, batch, device, vae, text_encoder, text_encoder_t5, freqs_cis_img):
    """
    Prepare model inputs for text-guided image-to-image training.

    batch contains:
        - input_image: condition/source image
        - edited_image: target/ground truth image
        - text_embedding: CLIP text tokens
        - text_embedding_mask: CLIP attention mask
        - text_embedding_t5: T5 text tokens
        - text_embedding_mask_t5: T5 attention mask
        - kwargs: additional metadata
    """
    input_image, edited_image, text_embedding, text_embedding_mask, text_embedding_t5, text_embedding_mask_t5, kwargs = batch

    # clip & T5 text embedding
    text_embedding = text_embedding.to(device)
    text_embedding_mask = text_embedding_mask.to(device)
    encoder_hidden_states = text_encoder(
        text_embedding.to(device),
        attention_mask=text_embedding_mask.to(device),
    )[0]
    text_embedding_t5 = text_embedding_t5.to(device).squeeze(1)
    text_embedding_mask_t5 = text_embedding_mask_t5.to(device).squeeze(1)

    with torch.no_grad():
        output_t5 = text_encoder_t5(
            input_ids=text_embedding_t5,
            attention_mask=text_embedding_mask_t5 if T5_ENCODER['attention_mask'] else None,
            output_hidden_states=True
        )
        encoder_hidden_states_t5 = output_t5['hidden_states'][T5_ENCODER['layer_index']].detach()
        del output_t5  # Release intermediate output

    if args.size_cond:
        image_meta_size = kwargs['image_meta_size'].to(device)
    else:
        image_meta_size = None
    if args.use_style_cond:
        style = kwargs['style'].to(device)
    else:
        style = None

    if args.extra_fp16:
        input_image = input_image.half()
        edited_image = edited_image.half()

    input_image = input_image.to(device)
    edited_image = edited_image.to(device)

    # Extract resolution from original images before encoding
    _, _, height, width = input_image.shape
    reso = f"{height}x{width}"
    cos_cis_img, sin_cis_img = freqs_cis_img[reso]

    vae_scaling_factor = vae.config.scaling_factor

    # Encode input image (condition) to latent space
    input_latents = vae.encode(input_image).latent_dist.sample().mul_(vae_scaling_factor)
    del input_image  # Release original image immediately after encoding

    # Encode edited image (target/ground truth) to latent space
    target_latents = vae.encode(edited_image).latent_dist.sample().mul_(vae_scaling_factor)
    del edited_image  # Release original image immediately after encoding

    # Parse auxiliary metadata if enabled
    auxiliary_data = None
    if args.use_auxiliary_encoder and 'raw_text' in kwargs:
        from adpdit.modules.auxiliary_encoder import AuxiliaryMetadataEncoder

        # Get raw text prompts from batch
        raw_texts = kwargs['raw_text']  # List of strings

        # Parse prompts to extract auxiliary metadata
        clinical_data, cognitive_scores = AuxiliaryMetadataEncoder.parse_prompts(raw_texts)

        auxiliary_data = {
            'clinical_data': clinical_data,
            'cognitive_scores': cognitive_scores,
        }

    # Extract classification labels if available (only when classification is enabled)
    cls_labels = None
    if args.use_classification and 'cls_label' in kwargs:
        cls_labels = kwargs['cls_label'].to(device)

    model_kwargs = dict(
        encoder_hidden_states=encoder_hidden_states,
        text_embedding_mask=text_embedding_mask,
        encoder_hidden_states_t5=encoder_hidden_states_t5,
        text_embedding_mask_t5=text_embedding_mask_t5,
        image_meta_size=image_meta_size,
        style=style,
        cos_cis_img=cos_cis_img,
        sin_cis_img=sin_cis_img,
        input_image=input_latents,  # Condition latents for img2img
        auxiliary_data=auxiliary_data,  # Add auxiliary metadata
    )

    return target_latents, model_kwargs, cls_labels


def main(args):
    if args.training_parts == "lora":
        args.use_ema = False

    os.environ["OMP_NUM_THREADS"] = "128"
    os.environ["MKL_NUM_THREADS"] = "128"
    torch.set_num_threads(128)
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    batch_size = args.batch_size
    grad_accu_steps = args.grad_accu_steps
    global_batch_size = world_size * batch_size * grad_accu_steps

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world_size + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(device)
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    deepspeed_config = deepspeed_config_from_args(args, global_batch_size)

    # Setup an experiment folder
    experiment_dir, checkpoint_dir, logger = create_exp_folder(args, rank)

    # TensorBoard Writer (only on rank 0 to avoid duplicate logs)
    writer = SummaryWriter(log_dir=os.path.join(experiment_dir, "tensorboard_logs")) if rank == 0 else None

    # Log all arguments; save args.json only on rank 0
    logger.info(sys.argv)
    logger.info(str(args))
    if rank == 0:
        args_dict = vars(args)
        args_dict['world_size'] = world_size
        with open(f"{experiment_dir}/args.json", 'w') as f:
            json.dump(args_dict, f, indent=4)

    tf_logging.set_verbosity_error()

    logger.info("Building ADP-DiT Model.")

    image_size = args.image_size
    if len(image_size) == 1:
        image_size = [image_size[0], image_size[0]]
    if len(image_size) != 2:
        raise ValueError(f"Invalid image size: {args.image_size}")
    assert image_size[0] % 8 == 0 and image_size[1] % 8 == 0, "Image size must be divisible by 8"

    latent_size = [image_size[0] // 8, image_size[1] // 8]

    assert args.deepspeed, f"Must enable deepspeed in this script: train_deepspeed.py"
    with deepspeed.zero.Init(
        data_parallel_group=torch.distributed.group.WORLD,
        remote_device=None if args.remote_device == 'none' else args.remote_device,
        config_dict_or_path=deepspeed_config,
        mpu=None,
        enabled=args.zero_stage == 3
    ):
        model = ADP_DIT_MODELS[args.model](args, input_size=latent_size, log_fn=logger.info)

    if args.multireso:
        resolutions = ResolutionGroup(image_size[0], align=16, step=args.reso_step, target_ratios=args.target_ratios).data
    else:
        resolutions = ResolutionGroup(image_size[0], align=16, target_ratios=['1:1']).data

    freqs_cis_img = init_image_posemb(
        args.rope_img,
        resolutions=resolutions,
        patch_size=model.patch_size,
        hidden_size=model.hidden_size,
        num_heads=model.num_heads,
        log_fn=logger.info,
        rope_real=args.rope_real
    )

    ema = None
    if args.use_ema:
        ema = EMA(args, model, device, logger)

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    if args.use_fp16:
        model = Float16Module(model, args)
    logger.info(f"    Using main model with data type {'fp16' if args.use_fp16 else 'fp32'}")

    diffusion = create_diffusion(
        noise_schedule=args.noise_schedule,
        predict_type=args.predict_type,
        learn_sigma=args.learn_sigma,
        mse_loss_weight_type=args.mse_loss_weight_type,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        noise_offset=args.noise_offset,
    )

    logger.info(f"    Using Mediffusion: Joint Diffusion for Self-Explainable Semi-Supervised Classification")
    logger.info(f"    Loss = MSE + λ * CE (joint loss with fixed lambda weighting)")
    logger.info(f"    λ: Fixed classification weight = {args.cls_loss_weight}")
    logger.info(f"    Warmup: Classification loss starts after {args.cls_warmup_steps} steps")
    logger.info(f"    - Before warmup: Loss = MSE only (let diffusion learn)")
    logger.info(f"    - After warmup: Loss = MSE + λ * CE (joint training)")
    logger.info(f"    Classifier regularization: dropout={getattr(args, 'cls_dropout', 0.3)}, label_smoothing={getattr(args, 'label_smoothing', 0.1)}")

    logger.info(f"    Loading vae from {VAE_EMA_PATH}")
    vae = AutoencoderKL.from_pretrained(VAE_EMA_PATH)
    logger.info(f"    Loading CLIP-G text encoder from {TEXT_ENCODER}")
    text_encoder = CLIPTextModel.from_pretrained(TEXT_ENCODER)
    logger.info(f"    Loading CLIP-G tokenizer from {TOKENIZER}")
    tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER)

    t5_path = T5_ENCODER['T5']
    embedder_t5 = T5Embedder(t5_path, torch_dtype=T5_ENCODER['torch_dtype'], max_length=args.text_len_t5)
    tokenizer_t5 = embedder_t5.tokenizer
    text_encoder_t5 = embedder_t5.model

    if args.extra_fp16:
        logger.info(f"    Using fp16 for extra modules: vae, text_encoder")
        vae = vae.half().to(device)
        text_encoder = text_encoder.half().to(device)
        text_encoder_t5 = text_encoder_t5.half().to(device)
    else:
        vae = vae.to(device)
        text_encoder = text_encoder.to(device)
        text_encoder_t5 = text_encoder_t5.to(device)

    logger.info(f"    Optimizer parameters: lr={args.lr}, weight_decay={args.weight_decay}")
    logger.info("    Using deepspeed optimizer")
    opt = None

    logger.info(f"Building Streaming Dataset.")
    logger.info(f"    Loading index file {args.index_file} (v2)")

    dataset = TextImageArrowStream(
        args=args,
        resolution=image_size[0],
        random_flip=args.random_flip,
        log_fn=logger.info,
        index_file=args.index_file,
        multireso=args.multireso,
        batch_size=batch_size,
        world_size=world_size,
        random_shrink_size_cond=args.random_shrink_size_cond,
        merge_src_cond=args.merge_src_cond,
        uncond_p=args.uncond_p,
        text_ctx_len=args.text_len,
        tokenizer=tokenizer,
        uncond_p_t5=args.uncond_p_t5,
        text_ctx_len_t5=args.text_len_t5,
        tokenizer_t5=tokenizer_t5
    )

    if args.multireso:
        sampler = BlockDistributedSampler(dataset, num_replicas=world_size, rank=rank, seed=args.global_seed,
                                          shuffle=False, drop_last=True, batch_size=batch_size)
    else:
        sampler = DistributedSamplerWithStartIndex(dataset, num_replicas=world_size, rank=rank,
                                                   seed=args.global_seed, shuffle=False, drop_last=True)
        # sampler = DistributedRandomReplacementSampler(dataset, num_replicas=world_size, rank=rank, num_samples_per_replica=5000,seed=args.global_seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True,persistent_workers=True)

    logger.info(f"    Dataset contains {len(dataset):,} images.")
    logger.info(f"    Index file: {args.index_file}.")
    if args.multireso:
        logger.info(f'    Using MultiResolutionBucketIndexV2 with step {dataset.index_manager.step} '
                    f'and base size {dataset.index_manager.base_size}')
        logger.info(f'\n  {dataset.index_manager.resolutions}')

    # Build Validation Dataset (if provided and classification is enabled)
    valid_loader = None
    validation_csv_path = None  # Store for sample generation

    # Determine validation index file
    valid_index_file_to_use = args.valid_index_file

    if args.use_classification:
        # Option 1: Create validation from train data (5 representative patients)
        if getattr(args, 'valid_from_train', False) and rank == 0:
            logger.info("=" * 60)
            logger.info("Creating Validation Dataset from Train Data (5 patients)")
            logger.info("=" * 60)

            val_json_path, val_csv_path, num_samples = create_validation_dataset_from_train(
                train_csv_path=args.train_csv_path,
                validation_patients=VALIDATION_PATIENTS_BY_SCENARIO,
                output_dir=experiment_dir,
                logger=logger
            )

            if val_json_path:
                valid_index_file_to_use = val_json_path
                validation_csv_path = val_csv_path
                logger.info(f"  Created validation dataset with {num_samples} samples")
                logger.info(f"  Scenarios: {list(VALIDATION_PATIENTS_BY_SCENARIO.keys())}")
            else:
                logger.warning("Failed to create validation dataset from train data")

        # Broadcast validation index file path to all ranks
        if dist.is_initialized():
            if rank == 0:
                valid_index_paths = [valid_index_file_to_use, validation_csv_path]
            else:
                valid_index_paths = [None, None]
            dist.broadcast_object_list(valid_index_paths, src=0)
            valid_index_file_to_use = valid_index_paths[0]
            validation_csv_path = valid_index_paths[1]

    if args.use_classification and valid_index_file_to_use:
        logger.info(f"Building Validation Dataset.")
        # Use smaller batch size for validation to avoid OOM
        # Priority: 1) args.val_batch_size if provided, 2) 50% of training batch, 3) min 32
        if args.val_batch_size is not None:
            validation_batch_size = args.val_batch_size
        else:
            validation_batch_size = max(32, batch_size // 2)

        logger.info(f"  Training batch size: {batch_size} per GPU")
        logger.info(f"  Validation batch size: {validation_batch_size} per GPU")

        valid_dataset = TextImageArrowStream(
            args=args,
            resolution=image_size[0],
            random_flip=False,  # No augmentation for validation
            log_fn=logger.info,
            index_file=valid_index_file_to_use,
            multireso=args.multireso,
            batch_size=validation_batch_size,  # ← Reduced for memory safety
            world_size=world_size,
            random_shrink_size_cond=args.random_shrink_size_cond,
            merge_src_cond=args.merge_src_cond,
            uncond_p=0.0,  # No unconditional for validation
            text_ctx_len=args.text_len,
            tokenizer=tokenizer,
            uncond_p_t5=0.0,
            text_ctx_len_t5=args.text_len_t5,
            tokenizer_t5=tokenizer_t5,
            enable_medical_augment=False  # No augmentation for validation
        )

        # Create validation sampler and loader
        valid_sampler = DistributedSamplerWithStartIndex(
            valid_dataset, num_replicas=world_size, rank=rank,
            seed=args.global_seed, shuffle=False, drop_last=False
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=validation_batch_size, shuffle=False,  # ← Reduced batch size
            sampler=valid_sampler, num_workers=args.num_workers,
            pin_memory=True, drop_last=False
        )

        logger.info(f"    Validation dataset contains {len(valid_dataset):,} images.")
        logger.info(f"    Validation index file: {valid_index_file_to_use}.")
        if getattr(args, 'valid_from_train', False):
            logger.info(f"    Validation patients: {list(VALIDATION_PATIENTS_BY_SCENARIO.values())}")
    elif rank == 0 and not args.use_classification:
        logger.info(f"⏭️  Skipping validation dataset (classification disabled for pure generative training)")

    logger.info(f"Loading parameter")
    start_epoch = 0
    start_epoch_step = 0
    train_steps = 0

    if args.resume:
        model, ema, start_epoch, start_epoch_step, train_steps = model_resume(args, model, ema, logger, len(loader))

    if args.training_parts == "lora":
        loraconfig = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            target_modules=args.target_modules
        )
        if args.use_fp16:
            model.module = get_peft_model(model.module, loraconfig)
        else:
            model = get_peft_model(model, loraconfig)

    logger.info(f"    Training parts: {args.training_parts}")

    model, opt, scheduler = deepspeed_initialize(args, logger, model, opt, deepspeed_config)

    model.train()
    if args.use_ema:
        ema.eval()

    logger.info(f"    Worker {rank} ready.")
    dist.barrier()

    iters_per_epoch = len(loader)

    # Calculate warmup steps for logging and training
    # LR scheduler uses warmup_ratio to determine warmup steps
    if args.num_cycles == 1:
        first_cycle_steps = args.max_training_steps
    elif getattr(args, 't_mult', 1.0) == 1.0:
        first_cycle_steps = args.max_training_steps // args.num_cycles
    else:
        t_mult = getattr(args, 't_mult', 2.0)
        first_cycle_steps = int(args.max_training_steps * (t_mult - 1) / (t_mult ** args.num_cycles - 1))

    warmup_steps = int(first_cycle_steps * args.warmup_ratio)

    logger.info(" ****************************** Running training ******************************")
    logger.info(f"      Number GPUs:               {world_size}")
    logger.info(f"      Number training samples:   {len(dataset):,}")
    logger.info(f"      Number parameters:         {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"      Number trainable params:   {sum(p.numel() for p in get_trainable_params(model)):,}")
    logger.info("    ------------------------------------------------------------------------------")
    logger.info(f"      Iters per epoch:           {iters_per_epoch:,}")
    logger.info(f"      Batch size per device:     {batch_size}")
    logger.info(f"      Batch size all device:     {batch_size * world_size * grad_accu_steps:,}")
    logger.info(f"      Gradient Accu steps:       {args.grad_accu_steps}")
    logger.info(f"      Total optimization steps:  {args.epochs * iters_per_epoch // grad_accu_steps:,}")
    logger.info(f"      Training epochs:           {start_epoch}/{args.epochs}")
    logger.info(f"      Training epoch steps:      {start_epoch_step:,}/{iters_per_epoch:,}")
    logger.info(f"      Training total steps:      {train_steps:,}/"
                f"{min(args.max_training_steps, args.epochs * iters_per_epoch):,}")
    logger.info("    ------------------------------------------------------------------------------")
    logger.info(f"      LR Scheduler:              CosineAnnealingWarmupRestarts")
    logger.info(f"      - Max LR:                  {args.max_lr}")
    logger.info(f"      - Min LR:                  {args.min_lr}")
    logger.info(f"      - Warmup ratio:            {args.warmup_ratio}")
    logger.info(f"      - Warmup steps:            {warmup_steps:,}")
    logger.info(f"      - Number of cycles:        {args.num_cycles}")
    if args.num_cycles > 1:
        logger.info(f"      - Cycle multiplier (t_mult): {getattr(args, 't_mult', 2.0)}")
        logger.info(f"      - LR decay per cycle (gamma): {getattr(args, 'gamma', 0.5)}")
        logger.info(f"      - 🔄 Restarts enabled:   LR will periodically reset to max_lr")
    logger.info("    ------------------------------------------------------------------------------")
    logger.info(f"      Noise schedule:            {args.noise_schedule}")
    logger.info(f"      Beta limits:               ({args.beta_start}, {args.beta_end})")
    logger.info(f"      Learn sigma:               {args.learn_sigma}")
    logger.info(f"      Prediction type:           {args.predict_type}")
    logger.info(f"      Noise offset:              {args.noise_offset}")
    logger.info("    ------------------------------------------------------------------------------")
    logger.info(f"      Using EMA model:           {args.use_ema} ({args.ema_dtype})")
    if args.use_ema:
        logger.info(f"      Using EMA decay:           {ema.max_value if args.use_ema else None}")
        logger.info(f"      Using EMA warmup power:    {ema.power if args.use_ema else None}")
    logger.info(f"      Using main model fp16:     {args.use_fp16}")
    logger.info(f"      Using extra modules fp16:  {args.extra_fp16}")
    logger.info("    ------------------------------------------------------------------------------")

    # Log classification training mode
    if args.use_classification:
        logger.info(f"      Training Mode:             🎯 Joint Training (Generation + Classification)")
        logger.info(f"      Classification class weights: CN=2.57, MCI=1.00, AD=3.07")
        logger.info(f"      (Addressing class imbalance: CN 22.7%, MCI 58.3%, AD 19.0%)")
        logger.info("    ------------------------------------------------------------------------------")
        logger.info(f"      Mediffusion Loss Strategy:")
        logger.info(f"      - Loss formula:            L_total = L_diff + λ * L_class")
        logger.info(f"      - Lambda (λ):              {args.cls_loss_weight}")
        logger.info(f"      - Classification warmup:   {args.cls_warmup_steps} steps")
        if args.cls_warmup_steps > 0:
            logger.info(f"      - Warmup strategy:         Train diffusion only for first {args.cls_warmup_steps} steps,")
            logger.info(f"                                 then enable classification (λ={args.cls_loss_weight})")
        else:
            logger.info(f"      - Warmup disabled:         Classification enabled from step 0")
    else:
        logger.info(f"      Training Mode:             🚀 Pure Generation (MSE Loss Only)")
        logger.info(f"      - Classification:          DISABLED")
        logger.info(f"      - Loss function:           MSE (L2 pixel distance)")
        logger.info(f"      - Focus:                   Diffusion model quality")
        logger.info(f"      - Memory usage:            Optimized (no classifier overhead)")
    logger.info("    ------------------------------------------------------------------------------")
    if valid_loader is not None:
        logger.info(f"      Validation enabled:        Yes")
        logger.info(f"      - Validation every:        {args.val_every} steps")
        logger.info(f"      - Validation batch size:   {validation_batch_size} (reduced for memory safety)")
        logger.info(f"      - Early stopping patience: {args.val_patience} validations")
        logger.info(f"      - Validation samples:      {len(valid_dataset):,}")
        logger.info(f"      - Min accuracy for freeze: 85.0%")
        logger.info(f"      🔄 Overfitting strategy:   Freeze classifier when plateau detected")
        logger.info(f"         (LR restarts continue for generation model)")
    else:
        logger.info(f"      Validation enabled:        No")
    logger.info("    ------------------------------------------------------------------------------")
    logger.info(f"      Experiment directory:      {experiment_dir}")
    logger.info("    *******************************************************************************")

    if args.gc_interval > 0:
        gc.disable()
        gc.collect()

    log_steps = 0
    running_loss = 0
    running_mse_loss = 0
    running_ce_loss = 0
    running_mse_weight = 0
    running_ce_weight = 0
    running_correct = 0  # Correct classification count
    running_total = 0    # Total classification samples

    # Classification metrics for every 100 steps (for 3-class: CN=0, MCI=1, AD=2)
    step_all_preds = []
    step_all_labels = []

    start_time = time.time()
    best_loss = float('inf')
    best_checkpoint_path = None

    # Validation tracking variables
    best_val_accuracy = 0.0
    val_accuracy_history = []
    patience_counter = 0

    # Classifier freezing flag
    classifier_frozen = False

    def freeze_classifier(model, logger):
        """
        Freeze classifier-related parameters while keeping MSE/generation parameters trainable.
        This prevents overfitting of the classifier while continuing to train the generation model.

        🔄 Cosine Annealing Restart Scheduler Compatibility:
        - When classifier is frozen, LR restarts only affect the generation model
        - This allows the diffusion model to continue benefiting from periodic LR increases
        - Classifier remains frozen to prevent overfitting, even during LR restarts
        """
        if args.use_fp16:
            actual_model = model.module.module
        else:
            actual_model = model.module

        # Freeze classification-related parameters
        frozen_params = []
        trainable_params = []

        for name, param in actual_model.named_parameters():
            # Freeze classification projection, MLP, and CE uncertainty parameter
            if 'cls_projection' in name or 'cls_mlp' in name or name == 'log_var_ce':
                param.requires_grad = False
                frozen_params.append(name)
            elif param.requires_grad:
                trainable_params.append(name)

        logger.info(f"🔒 Classifier frozen! Frozen parameters ({len(frozen_params)}): {', '.join(frozen_params)}")
        logger.info(f"    ✅ Generation model remains trainable ({len(trainable_params)} params)")
        logger.info(f"    🔄 LR scheduler will continue to update generation model parameters")
        if args.num_cycles > 1:
            logger.info(f"    📊 Cosine restarts will continue for diffusion model training")
        return frozen_params

    # Note: warmup_steps already calculated above for logging purposes
    # Loss monitor removed - not needed for Mediffusion approach
    # Mediffusion uses fixed lambda weighting, no need for uncertainty monitoring

    epoch = start_epoch
    while epoch < args.epochs:
        shuffle_seed = args.global_seed + epoch
        logger.info(f"    Start random shuffle with seed={shuffle_seed}")
        dataset.shuffle(seed=shuffle_seed, fast=True)
        logger.info(f"    End of random shuffle")

        if not args.multireso:
            start_index = start_epoch_step * world_size * batch_size
            if start_index != sampler.start_index:
                sampler.start_index = start_index
                start_epoch_step = 0
                logger.info(f"      Iters left this epoch: {len(loader):,}")

        epoch_loss_sum = 0.0
        epoch_steps = 0

        logger.info(f"    Beginning epoch {epoch}...")
        for batch in loader:
            latents, model_kwargs, cls_labels = prepare_model_inputs(args, batch, device, vae, text_encoder, text_encoder_t5, freqs_cis_img)
            loss_dict = diffusion.training_losses(model=model, x_start=latents, model_kwargs=model_kwargs, cls_labels=cls_labels)

            # Use Mediffusion-style joint loss with lambda weighting and warm-up
            # Reference: "Mediffusion: Joint Diffusion for Self-Explainable Semi-Supervised
            #             Classification and Medical Image Generation"
            # Loss = L_diff + lambda * L_class
            # - lambda: Fixed hyperparameter for scale balancing (0.00005 ~ 0.001)
            # - Warm-up: Classification loss starts after N steps to let diffusion learn features first
            if args.use_classification and cls_labels is not None and 'ce' in loss_dict:
                # Diffusion loss (MSE) - always applied
                mse_loss = loss_dict["mse"].mean()

                # Classification loss (CE) - applied to labeled data
                ce_loss_raw = loss_dict["ce"].mean()

                # ✅ Clamp CE loss to prevent numerical instability when classifier saturates
                ce_loss = torch.clamp(ce_loss_raw, min=1e-4)

                # Apply warm-up strategy: classification loss starts after warmup steps
                if train_steps < args.cls_warmup_steps:
                    # Warm-up phase: Only train diffusion model (classification weight = 0)
                    # IMPORTANT: Do NOT include ce_loss in computational graph during warmup
                    cls_weight = 0.0
                    effective_ce_loss = 0.0
                    loss = mse_loss  # No CE loss in graph → no gradient to classifier
                else:
                    # Normal training: Apply lambda weight to classification loss
                    cls_weight = args.cls_loss_weight
                    effective_ce_loss = cls_weight * ce_loss
                    loss = mse_loss + effective_ce_loss  # Mediffusion-style: L = MSE + λ*CE

                # Store individual losses for logging
                loss_dict["mse_loss"] = mse_loss
                loss_dict["ce_loss"] = ce_loss  # Raw CE loss (before weighting)
                loss_dict["loss"] = loss

                # Store weights for logging
                loss_dict["mse_weight"] = torch.tensor(1.0)  # MSE always weighted at 1.0
                loss_dict["ce_weight"] = torch.tensor(cls_weight)  # CE weighted by lambda (or 0 during warmup)

                # Normalized alphas for visualization (what percentage each task contributes)
                total_contribution = 1.0 + cls_weight  # 1.0 (MSE) + lambda (CE)
                loss_dict["mse_alpha"] = torch.tensor(1.0 / total_contribution if total_contribution > 0 else 1.0)
                loss_dict["ce_alpha"] = torch.tensor(cls_weight / total_contribution if total_contribution > 0 else 0.0)
            else:
                # Pure generative training (MSE loss only)
                loss = loss_dict["loss"].mean()

            model.backward(loss)
            last_batch_iteration = (train_steps + 1) // (global_batch_size // (batch_size * world_size))
            model.step(lr_kwargs={'last_batch_iteration': last_batch_iteration})

            if args.use_ema:
                if args.use_fp16:
                    ema.update(model.module.module, step=train_steps)
                else:
                    ema.update(model.module, step=train_steps)
            # ===========================================================================
            # Log loss values and check for anomalies:
            # ===========================================================================
            running_loss += loss.item()

            if cls_labels is not None and 'ce' in loss_dict:
                running_mse_loss += loss_dict["mse_loss"].item()
                running_ce_loss += loss_dict["ce_loss"].item()
                running_mse_weight += loss_dict["mse_weight"].item()
                running_ce_weight += loss_dict["ce_weight"].item()
                # Note: sigma parameters removed (Mediffusion uses direct summation, not uncertainty weighting)

                # Calculate classification accuracy (when classification is enabled)
                if args.use_classification and "cls_logits" in loss_dict and "cls_labels" in loss_dict:
                    cls_logits = loss_dict["cls_logits"]
                    cls_labels_batch = loss_dict["cls_labels"]

                    # Get predicted classes (argmax of logits)
                    _, predicted = torch.max(cls_logits.data, 1)

                    # Count correct predictions
                    correct = (predicted == cls_labels_batch).sum().item()
                    total = cls_labels_batch.size(0)

                    running_correct += correct
                    running_total += total

                    # Collect predictions and labels for step-level metrics (every 100 steps)
                    step_all_preds.extend(predicted.cpu().numpy().tolist())
                    step_all_labels.extend(cls_labels_batch.cpu().numpy().tolist())

                # Anomaly detection removed - Mediffusion uses fixed lambda, simpler and more stable
                # No need for alpha monitoring or loss explosion detection with fixed weighting

            # Clean up intermediate tensors to prevent memory accumulation
            del latents, model_kwargs
            if 'loss_dict' in locals():
                del loss_dict
            torch.cuda.empty_cache()

            log_steps += 1
            train_steps += 1
            epoch_loss_sum += loss.item()
            epoch_steps += 1

            if  train_steps % args.log_every == 0:
                # Clear cache before logging to prevent memory fragmentation during logging
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()

                # Ensure synchronization across GPUs before timing/logging
                torch.cuda.synchronize()
                # Calculate elapsed time for steps
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)

                # Compute the average loss across all processes
                avg_loss_tensor = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss_tensor.item() / world_size

                update_step = train_steps // args.grad_accu_steps  # Based on optimizer steps

                if rank == 0: # Only log on the main process
                    # Retrieve current learning rate
                    current_lr = opt.param_groups[0]['lr']
                    samples_per_sec = int(steps_per_sec * batch_size * world_size)

                    # Log metrics to TensorBoard
                    writer.add_scalar('Loss/TrainStep', avg_loss, train_steps)
                    writer.add_scalar('Loss/UpdateStep', avg_loss, update_step)

                    # Log classification-specific metrics if available and classification is enabled
                    if args.use_classification and running_mse_loss > 0:
                        avg_mse = running_mse_loss / log_steps
                        avg_ce = running_ce_loss / log_steps
                        avg_mse_weight = running_mse_weight / log_steps
                        avg_ce_weight = running_ce_weight / log_steps
                        # Mediffusion: Fixed weights (1.0 each), no uncertainty parameters
                        # Normalized alphas for compatibility
                        avg_mse_alpha = avg_mse_weight / (avg_mse_weight + avg_ce_weight)
                        avg_ce_alpha = avg_ce_weight / (avg_mse_weight + avg_ce_weight)

                        writer.add_scalar('Loss/MSE', avg_mse, train_steps)
                        writer.add_scalar('Loss/CrossEntropy', avg_ce, train_steps)
                        writer.add_scalar('Loss/MSE_Weight', avg_mse_weight, train_steps)
                        writer.add_scalar('Loss/CE_Weight', avg_ce_weight, train_steps)  # Lambda (λ)
                        writer.add_scalar('Loss/MSE_Alpha', avg_mse_alpha, train_steps)  # Normalized weight
                        writer.add_scalar('Loss/CE_Alpha', avg_ce_alpha, train_steps)    # Normalized weight

                        # Log warm-up status
                        if train_steps < args.cls_warmup_steps:
                            warmup_progress = train_steps / args.cls_warmup_steps
                            writer.add_scalar('Mediffusion/Warmup_Progress', warmup_progress, train_steps)
                            writer.add_scalar('Mediffusion/Classification_Active', 0, train_steps)
                        else:
                            writer.add_scalar('Mediffusion/Classification_Active', 1, train_steps)

                        # Log classification accuracy if available
                        if running_total > 0:
                            accuracy = 100.0 * running_correct / running_total
                            writer.add_scalar('Accuracy/Train', accuracy, train_steps)
                            writer.add_scalar('Accuracy/TrainUpdateStep', accuracy, update_step)

                            # Check for classifier overfitting frequently (every log_every steps)
                            # Freeze if: (1) accuracy >= 95%, or (2) CE loss at minimum threshold with high accuracy
                            should_freeze = (accuracy >= 95.0) or (avg_ce <= 1.1e-4 and accuracy >= 90.0)
                            if not classifier_frozen and should_freeze and train_steps > args.cls_warmup_steps:
                                logger.warning(f"⚠️  Classifier overfitting detected! Accuracy={accuracy:.2f}%, CE={avg_ce:.6f}")
                                frozen_params = freeze_classifier(model, logger)
                                classifier_frozen = True
                                writer.add_scalar('Classifier/Frozen', 1, train_steps)

                    writer.add_scalar('LearningRate/TrainStep', current_lr, train_steps)
                    writer.add_scalar('LearningRate/UpdateStep', current_lr, update_step)
                    writer.add_scalar('Performance/StepsPerSec', steps_per_sec, train_steps)
                    writer.add_scalar('Performance/SamplesPerSec', samples_per_sec, train_steps)

                    # -- Console Log --
                    log_msg = (f"(step={train_steps:07d}) "
                              f"(update_step={update_step:07d}) "
                              f"Train Loss: {avg_loss:.4f}, ")

                    # Only log classification metrics when classification is enabled
                    if args.use_classification and running_mse_loss > 0:
                        log_msg += (f"MSE: {avg_mse:.4f}, "
                                   f"CE: {avg_ce:.4f}, ")

                        # Show classification weight (lambda) and warm-up status
                        if train_steps < args.cls_warmup_steps:
                            warmup_progress = 100.0 * train_steps / args.cls_warmup_steps
                            log_msg += f"λ: 0.0 (warmup {warmup_progress:.0f}%), "
                        else:
                            log_msg += f"λ: {args.cls_loss_weight:.5f}, "

                        if running_total > 0:
                            accuracy = 100.0 * running_correct / running_total
                            log_msg += f"Acc: {accuracy:.2f}%, "

                    log_msg += (f"Lr: {current_lr:.6g}, "
                               f"Steps/Sec: {steps_per_sec:.2f}, "
                               f"Samples/Sec: {samples_per_sec:d}")
                    logger.info(log_msg)

                running_loss = 0
                running_mse_loss = 0
                running_ce_loss = 0
                running_mse_weight = 0
                running_ce_weight = 0
                running_correct = 0  # Reset accuracy counters
                running_total = 0
                log_steps = 0
                start_time = time.time()

            # Calculate classification metrics every 100 steps (only after classification warmup)
            # Skip during warmup period to let generation model stabilize first
            should_compute_cls_metrics = (
                args.use_classification and
                train_steps % 100 == 0 and
                train_steps > 0 and
                train_steps > args.cls_warmup_steps  # Only after warmup
            )

            if should_compute_cls_metrics:
                if rank == 0:
                    logger.info(f"    📊 Computing confusion matrix at step {train_steps} (collected {len(step_all_preds)} predictions)")
                if rank == 0 and len(step_all_preds) > 0 and len(step_all_labels) > 0:
                    # Convert to numpy arrays
                    y_true = np.array(step_all_labels)
                    y_pred = np.array(step_all_preds)

                    # Overall accuracy
                    step_accuracy = 100.0 * np.sum(y_true == y_pred) / len(y_true)
                    if writer is not None:
                        writer.add_scalar('Step/Accuracy', step_accuracy, train_steps)

                    # Confusion matrix for 3 classes (CN=0, MCI=1, AD=2)
                    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

                    # Calculate per-class metrics
                    for class_idx, class_name in enumerate(['CN', 'MCI', 'AD']):
                        # For multi-class, we use one-vs-rest approach
                        # True Positives: correctly predicted as this class
                        tp = cm[class_idx, class_idx]

                        # False Negatives: this class predicted as other classes
                        fn = np.sum(cm[class_idx, :]) - tp

                        # False Positives: other classes predicted as this class
                        fp = np.sum(cm[:, class_idx]) - tp

                        # True Negatives: other classes correctly predicted as other classes
                        tn = np.sum(cm) - tp - fn - fp

                        # Sensitivity (Recall, True Positive Rate)
                        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                        # Specificity (True Negative Rate)
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

                        # Precision
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

                        # F1 Score
                        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

                        # Log to TensorBoard
                        if writer is not None:
                            writer.add_scalar(f'Step/{class_name}/TP', tp, train_steps)
                            writer.add_scalar(f'Step/{class_name}/TN', tn, train_steps)
                            writer.add_scalar(f'Step/{class_name}/FP', fp, train_steps)
                            writer.add_scalar(f'Step/{class_name}/FN', fn, train_steps)
                            writer.add_scalar(f'Step/{class_name}/Sensitivity', sensitivity, train_steps)
                            writer.add_scalar(f'Step/{class_name}/Specificity', specificity, train_steps)
                            writer.add_scalar(f'Step/{class_name}/Precision', precision, train_steps)
                            writer.add_scalar(f'Step/{class_name}/F1Score', f1, train_steps)

                        # Log to console
                        logger.info(f"    Step {train_steps} - {class_name}: "
                                   f"TP={tp}, TN={tn}, FP={fp}, FN={fn}, "
                                   f"Sens={sensitivity:.4f}, Spec={specificity:.4f}, "
                                   f"Prec={precision:.4f}, F1={f1:.4f}")

                    # Log confusion matrix values
                    logger.info(f"    Step {train_steps} Confusion Matrix:")
                    logger.info(f"              Pred_CN  Pred_MCI  Pred_AD")
                    logger.info(f"    True_CN   {cm[0,0]:6d}   {cm[0,1]:6d}   {cm[0,2]:6d}")
                    logger.info(f"    True_MCI  {cm[1,0]:6d}   {cm[1,1]:6d}   {cm[1,2]:6d}")
                    logger.info(f"    True_AD   {cm[2,0]:6d}   {cm[2,1]:6d}   {cm[2,2]:6d}")

                    # Reset step-level metrics
                    step_all_preds.clear()
                    step_all_labels.clear()

                    # Clean up memory after step-level metrics computation
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    gc.collect()

            # Run validation to detect overfitting (validation-based early stopping)
            # Compatible with cosine annealing restart scheduler
            # Only run when classification is enabled
            # Start validation after --val-start-step to avoid OOM during early training
            val_start_step = args.val_start_step  # Default 0 if not specified in config
            if (args.use_classification and valid_loader is not None and
                train_steps % args.val_every == 0 and
                train_steps > args.cls_warmup_steps and
                train_steps >= val_start_step):

                val_accuracy, val_metrics = run_validation(
                    args, rank, logger, model, valid_loader, device,
                    vae, text_encoder, text_encoder_t5, freqs_cis_img,
                    train_steps, writer
                )

                # Clean up memory after validation
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()

                if rank == 0 and val_accuracy is not None:
                    val_accuracy_history.append(val_accuracy)

                    # Cosine Annealing Restart scheduler compatibility logic:
                    # - Consider restart cycles: overfitting is judged within each cycle
                    # - Best accuracy is tracked over the entire training; patience is applied per cycle

                    # Calculate current cycle position (for cosine annealing restarts)
                    if args.num_cycles > 1:
                        # Calculate which cycle we're in
                        t_mult = getattr(args, 't_mult', 2.0)
                        if t_mult == 1.0:
                            cycle_length = args.max_training_steps // args.num_cycles
                            current_cycle = min(train_steps // cycle_length, args.num_cycles - 1)
                            cycle_start_step = current_cycle * cycle_length
                        else:
                            # Variable cycle length with t_mult
                            # Calculate cumulative cycle lengths to find current cycle
                            cumulative_steps = 0
                            first_cycle_len = int(args.max_training_steps * (t_mult - 1) / (t_mult ** args.num_cycles - 1))
                            current_cycle = 0
                            cycle_start_step = 0

                            for i in range(args.num_cycles):
                                cycle_len = int(first_cycle_len * (t_mult ** i))
                                if cumulative_steps <= train_steps < cumulative_steps + cycle_len:
                                    current_cycle = i
                                    cycle_start_step = cumulative_steps
                                    break
                                cumulative_steps += cycle_len

                        steps_in_cycle = train_steps - cycle_start_step
                        logger.info(f"    📊 Cosine cycle {current_cycle+1}/{args.num_cycles}, "
                                   f"step {steps_in_cycle} in current cycle")
                    else:
                        current_cycle = 0
                        cycle_start_step = 0
                        steps_in_cycle = train_steps

                    # Early stopping logic with cycle awareness
                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        patience_counter = 0
                        logger.info(f"✅ New best validation accuracy: {best_val_accuracy:.2f}%")
                    else:
                        patience_counter += 1
                        logger.info(f"⚠️  Validation accuracy did not improve "
                                   f"({patience_counter}/{args.val_patience} patience)")

                    # 🔒 Freeze classifier if overfitting detected
                    # Strategy: Only freeze when we have high accuracy but no improvement
                    # This allows the model to benefit from LR restarts for generation quality
                    freeze_condition = (
                        not classifier_frozen and
                        patience_counter >= args.val_patience and
                        val_accuracy >= 85.0  # Only freeze if accuracy is reasonably high
                    )

                    if freeze_condition:
                        logger.warning(f"🛑 Classifier overfitting detected!")
                        logger.warning(f"   Current validation accuracy: {val_accuracy:.2f}%")
                        logger.warning(f"   Best validation accuracy: {best_val_accuracy:.2f}%")
                        logger.warning(f"   No improvement for {args.val_patience} consecutive validations "
                                      f"({args.val_patience * args.val_every} steps)")
                        logger.warning(f"   🔄 Note: Cosine LR restarts will continue for generation model")
                        frozen_params = freeze_classifier(model, logger)
                        classifier_frozen = True
                        if writer is not None:
                            writer.add_scalar('Classifier/Frozen', 1, train_steps)
                            writer.add_scalar('Classifier/FrozenAtCycle', current_cycle, train_steps)
                            writer.add_scalar('Classifier/BestValAccuracy', best_val_accuracy, train_steps)

                    # Log cycle information for monitoring restart behavior
                    if args.num_cycles > 1 and writer is not None:
                        writer.add_scalar('Scheduler/CurrentCycle', current_cycle, train_steps)
                        writer.add_scalar('Scheduler/StepInCycle', steps_in_cycle, train_steps)

            # Garbage collection and memory cleanup:
            if args.gc_interval > 0 and (train_steps % args.gc_interval == 0):
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            # Save checkpoint
            if (train_steps % args.ckpt_every == 0 or train_steps % args.ckpt_latest_every == 0) and train_steps > 0:
                save_checkpoint(args, rank, logger, model, ema, epoch, train_steps, checkpoint_dir, by='step')

            # Sample generation during training (one patient per 5 scenarios)
            if args.sample_every > 0 and train_steps % args.sample_every == 0 and train_steps > 0:
                generate_samples(
                    args, rank, logger, model, vae, text_encoder, text_encoder_t5,
                    tokenizer, tokenizer_t5, diffusion, freqs_cis_img, device,
                    train_steps, args.results_dir,
                    validation_csv_path=validation_csv_path
                )
                # Memory cleanup after sample generation
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()

            if train_steps >= args.max_training_steps:
                logger.info(f"Breaking step loop at train_steps={train_steps}.")
                break

        if train_steps >= args.max_training_steps:
            logger.info(f"Breaking epoch loop at epoch={epoch}.")
            break

        if rank == 0 and writer is not None:
            writer.add_scalar('Epoch/train', epoch, train_steps)

        # Compute average loss at the end of the epoch
        avg_epoch_loss = epoch_loss_sum / epoch_steps if epoch_steps > 0 else float('inf')

        # Save per-epoch checkpoint
        if args.ckpt_every_n_epoch > 0 and epoch % args.ckpt_every_n_epoch == 0:
            save_checkpoint(args, rank, logger, model, ema, epoch, train_steps, checkpoint_dir, by='epoch')

        epoch += 1
    # Save the final model
    save_checkpoint(args, rank, logger, model, ema, epoch, train_steps, checkpoint_dir, by='final')

    if rank == 0 and writer is not None:
        writer.close()

    dist.destroy_process_group()


if __name__ == "__main__":
    # Start
    main(get_args())
