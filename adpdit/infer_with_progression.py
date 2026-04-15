#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADP-DiT Disease Progression Inference Script (Updated for new CSV format)

CSV Format:
input_image,edited_image,edit_prompt

This script:
1. Reads CSV with input/target image pairs and prompts
2. Groups data by patient (based on 'first visit' marker)
3. Automatically classifies progression type from prompts
4. Runs inference with classification probability output
"""
import csv
import os
import sys
import argparse
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import subprocess
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.progression_utils import (
    extract_condition_from_prompt,
    extract_cls_label_from_prompt,
    is_first_visit,
    create_progression_prompt
)

# Constants
BASE_NEGATIVE = "noise, blur, anatomically incorrect, shrinking ventricles, cortical thickening, non-progressive atrophy, temporal inconsistency, non-AD patterns"

# Model settings (adjust these paths as needed)
DIT_WEIGHT = "./checkpoints/mp_rank_00_model_states.pt"  # Set your checkpoint path
LOAD_KEY = "module"
MODEL = "DiT-g/2"

# Experiment settings
SELECTED_SAMPLERS = ["ddpm", "ddim", "dpmpp_2m_karras"]
STRENGTH_VALUES = [1]
CFG_SCALE_VALUES = [3.5]
INFER_STEPS_VALUES = [100]


def extract_condition_from_filename(filepath: str) -> Optional[str]:
    """
    Try to extract condition from filename if it contains CN, MCI, or AD.

    Args:
        filepath: Path to image file

    Returns:
        Condition string or None
    """
    basename = os.path.basename(filepath).lower()

    if 'ad' in basename or 'alzheimer' in basename:
        return "Alzheimer Disease"
    elif 'mci' in basename:
        return "Mild Cognitive Impairment"
    elif 'cn' in basename or 'normal' in basename:
        return "Cognitive Normal"

    return None


def read_csv_with_conditions(csv_file: str) -> List[Dict]:
    """
    Read CSV and extract conditions for both input and target.

    Args:
        csv_file: Path to CSV file

    Returns:
        List of dicts with input_img, target_img, target_prompt, input_condition, target_condition
    """
    data = []

    if not os.path.isfile(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return data

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            input_img = row['input_image'].strip()
            edited_img = row['edited_image'].strip()
            edit_prompt = row['edit_prompt'].strip()

            # Extract target condition from prompt
            target_condition = extract_condition_from_prompt(edit_prompt)

            # Try to extract input condition from filename
            input_condition = extract_condition_from_filename(input_img)

            # If can't extract from filename, use target condition (stable case)
            if input_condition is None:
                input_condition = target_condition

            # Get classification labels
            input_label = extract_cls_label_from_prompt(input_condition)
            target_label = extract_cls_label_from_prompt(edit_prompt)

            data.append({
                'input_img': input_img,
                'target_img': edited_img,
                'target_prompt': edit_prompt,
                'input_condition': input_condition,
                'target_condition': target_condition,
                'input_label': input_label,
                'target_label': target_label,
                'is_first_visit': is_first_visit(edit_prompt),
                'row_idx': idx
            })

    print(f"✅ Loaded {len(data)} rows from {csv_file}")
    return data


def classify_progression_pair(input_condition: str, target_condition: str) -> str:
    """
    Classify progression type from input and target conditions.

    Args:
        input_condition: Input disease condition
        target_condition: Target disease condition

    Returns:
        Progression type string (e.g., "CNtoMCI")
    """
    # Map to short forms
    condition_map = {
        "Cognitive Normal": "CN",
        "Mild Cognitive Impairment": "MCI",
        "Alzheimer Disease": "AD"
    }

    input_short = condition_map.get(input_condition, "CN")
    target_short = condition_map.get(target_condition, "CN")

    return f"{input_short}to{target_short}"


def group_by_patient(data: List[Dict]) -> List[List[Dict]]:
    """
    Group data by patient based on 'first visit' marker.

    Args:
        data: List of data dicts

    Returns:
        List of patient groups
    """
    patients = []
    current_patient = []

    for row in data:
        if row['is_first_visit']:
            # Start of new patient
            if current_patient:
                patients.append(current_patient)
            current_patient = [row]
        else:
            # Continuation of current patient
            current_patient.append(row)

    # Don't forget last patient
    if current_patient:
        patients.append(current_patient)

    return patients


def organize_by_progression_type(data: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Organize data by progression type.

    Args:
        data: List of data dicts

    Returns:
        Dict mapping progression type to list of samples
    """
    organized = defaultdict(list)

    for row in data:
        prog_type = classify_progression_pair(row['input_condition'], row['target_condition'])
        organized[prog_type].append(row)

    return organized


def run_inference_command(
    row: Dict,
    output_dir: str,
    sampler: str,
    strength: float,
    cfg_scale: float,
    infer_steps: int,
    progression_type: str
) -> bool:
    """
    Run inference for a single sample.

    Args:
        row: Data dict with image paths and prompts
        output_dir: Output directory
        sampler: Sampler name
        strength: Strength value
        cfg_scale: CFG scale value
        infer_steps: Inference steps
        progression_type: Progression type

    Returns:
        True if successful, False otherwise
    """
    input_img = row['input_img']
    target_prompt = row['target_prompt']
    input_condition = row['input_condition']
    target_condition = row['target_condition']
    target_label = row['target_label']

    # Create progression-aware prompt
    enhanced_prompt = create_progression_prompt(target_prompt, input_condition, target_condition)

    # Create output subfolder
    output_subfolder = os.path.join(
        output_dir,
        progression_type,
        sampler,
        f"strength_{strength}_cfg_{cfg_scale}_steps_{infer_steps}"
    )
    os.makedirs(output_subfolder, exist_ok=True)

    # Build inference command
    cmd = [
        "python3", "-m", "adpdit.inference",
        "--image_path", input_img,
        "--prompt", enhanced_prompt,
        "--negative_prompt", BASE_NEGATIVE,
        "--dit_weight", DIT_WEIGHT,
        "--load_key", LOAD_KEY,
        "--model", MODEL,
        "--image_save_path", output_subfolder,
        "--sampler", sampler,
        "--strength", str(strength),
        "--cfg_scale", str(cfg_scale),
        "--infer_steps", str(infer_steps),
    ]

    # Get basename for logging
    basename = os.path.basename(input_img)

    print(f"\n{'='*80}")
    print(f"🔄 Processing: {basename}")
    print(f"   Progression: {progression_type} ({input_condition} → {target_condition})")
    print(f"   Target Label: {target_label} (0:CN, 1:MCI, 2:AD)")
    print(f"   Sampler: {sampler}, Strength: {strength}, CFG: {cfg_scale}, Steps: {infer_steps}")
    print(f"   Prompt: {enhanced_prompt[:100]}...")
    print(f"{'='*80}\n")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Success: {basename}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error processing {basename}:")
        print(f"   {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Disease Progression Inference with automatic progression type classification"
    )
    parser.add_argument("--csv_file", type=str, required=True,
                        help="Path to CSV file (format: input_image,edited_image,edit_prompt)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for generated images")
    parser.add_argument("--filter_progression", type=str, default=None,
                        choices=["CNtoCN", "CNtoMCI", "MCItoMCI", "MCItoAD", "ADtoAD"],
                        help="Filter to specific progression type (optional)")
    parser.add_argument("--samplers", type=str, nargs='+', default=SELECTED_SAMPLERS,
                        help="Samplers to use")
    parser.add_argument("--strength", type=float, nargs='+', default=STRENGTH_VALUES,
                        help="Strength values")
    parser.add_argument("--cfg_scale", type=float, nargs='+', default=CFG_SCALE_VALUES,
                        help="CFG scale values")
    parser.add_argument("--infer_steps", type=int, nargs='+', default=INFER_STEPS_VALUES,
                        help="Inference steps")

    args = parser.parse_args()

    # Read CSV data
    print(f"\n📖 Reading CSV file: {args.csv_file}")
    data = read_csv_with_conditions(args.csv_file)

    if not data:
        print("❌ No data found in CSV file!")
        return

    # Group by patient
    print(f"\n👥 Grouping data by patient...")
    patient_groups = group_by_patient(data)
    print(f"   Found {len(patient_groups)} patients")

    # Organize by progression type
    print(f"\n📊 Organizing by progression type...")
    progression_data = organize_by_progression_type(data)

    # Print statistics
    print(f"\n📈 Progression Type Statistics:")
    total_samples = 0
    for prog_type in sorted(progression_data.keys()):
        count = len(progression_data[prog_type])
        total_samples += count
        print(f"   {prog_type}: {count} samples")
    print(f"   Total: {total_samples} samples")

    # Filter if requested
    if args.filter_progression:
        print(f"\n🔍 Filtering to progression type: {args.filter_progression}")
        if args.filter_progression in progression_data:
            progression_data = {args.filter_progression: progression_data[args.filter_progression]}
        else:
            print(f"❌ No samples found for progression type: {args.filter_progression}")
            return

    # Run inference
    total_processed = 0
    total_success = 0

    for prog_type, samples in sorted(progression_data.items()):
        print(f"\n{'#'*80}")
        print(f"# Processing {prog_type}: {len(samples)} samples")
        print(f"{'#'*80}\n")

        for row in samples:
            for sampler in args.samplers:
                for strength in args.strength:
                    for cfg_scale in args.cfg_scale:
                        for infer_steps in args.infer_steps:
                            success = run_inference_command(
                                row, args.output_dir,
                                sampler, strength, cfg_scale, infer_steps,
                                prog_type
                            )
                            total_processed += 1
                            if success:
                                total_success += 1

    # Print summary
    print(f"\n{'='*80}")
    print(f"✨ Inference Complete!")
    print(f"   Total processed: {total_processed}")
    print(f"   Successful: {total_success}")
    print(f"   Failed: {total_processed - total_success}")
    print(f"   Success rate: {total_success/total_processed*100:.1f}%")
    print(f"   Output directory: {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
