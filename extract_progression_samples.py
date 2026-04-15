#!/usr/bin/env python3
"""
Extract progression images for specific patients from CNtoMCI and MCItoAD scenarios.

Extracts input, target, output, and error map images,
and additionally crops important regions to 256x256.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Configuration - set these paths before running
CSV_PATH = './test_outputs/test_results.csv'
IMAGE_BASE = './dataset/AD_meta/images/test'
OUTPUT_BASE = './test_outputs/generated'
OUTPUT_DIR = './progression_samples'

# ============================================================================
# Manual crop coordinates (for 256x256 source images)
# Format: (left, top, right, bottom)
# ============================================================================
CROP_COORDS = {
    'CNtoMCI': (80, 120, 121, 161),
    'MCItoAD': (74, 145, 107, 178),
}
# ============================================================================


def create_colorbar(height, width=30, cmap_name='inferno', z_min=0, z_max=1, num_ticks=5):
    """Create a colorbar image for the error map."""
    from PIL import ImageDraw, ImageFont

    actual_height = max(10, height - 20)

    colorbar_array = np.linspace(1, 0, actual_height).reshape(-1, 1)
    colorbar_array = np.repeat(colorbar_array, width, axis=1)

    cmap = plt.colormaps[cmap_name]
    colorbar_colored = cmap(colorbar_array)
    colorbar_rgb = (colorbar_colored[:, :, :3] * 255).astype(np.uint8)

    colorbar_img = Image.fromarray(colorbar_rgb)

    extended_width = width + 40
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
        label = f"{val:.1f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_height = bbox[3] - bbox[1]
        draw.text((width + 5, pos - text_height//2), label, fill='black', font=font)

    return extended_colorbar


def compute_error_map(target_array, output_array):
    """
    Compute error map between target and output images.
    Returns error map RGB image.
    (Same function as in evaluate_test_csv.py)
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

    return Image.fromarray(error_map_rgb)


def crop_and_resize(img, crop_coords, target_size=256):
    """
    Crop the image using crop_coords, then resize to target_size.

    Args:
        img: PIL Image
        crop_coords: (left, top, right, bottom) tuple
        target_size: Output image size

    Returns:
        Cropped and resized PIL Image
    """
    left, top, right, bottom = crop_coords
    cropped = img.crop((left, top, right, bottom))
    resized = cropped.resize((target_size, target_size), Image.LANCZOS)
    return resized


def create_comparison_grid(images_dict, slice_num, save_path, cell_size=256, gap=5):
    """
    Create a 2-row x 4-column comparison image with per-row colorbars.

    Row 1: CNtoMCI (input, target, output, errormap) + colorbar
    Row 2: MCItoAD (input, target, output, errormap) + colorbar

    Args:
        images_dict: {
            'CNtoMCI': {'input': img, 'target': img, 'output': img, 'errormap': img},
            'MCItoAD': {'input': img, 'target': img, 'output': img, 'errormap': img}
        }
        slice_num: Slice number (used for title)
        save_path: Output file path
        cell_size: Size of each cell
        gap: Gap between cells
    """
    from PIL import ImageDraw, ImageFont

    # Layout settings
    cols = 4  # input, target, output, errormap
    rows = 2  # CNtoMCI, MCItoAD

    header_height = 30  # Column header height
    row_label_width = 80  # Row label width
    colorbar_width = 70  # Colorbar width

    # Create colorbar for each row
    colorbar_img = create_colorbar(cell_size, width=30, z_min=0.0, z_max=1.0, num_ticks=6)

    total_width = row_label_width + cols * cell_size + (cols - 1) * gap + gap + colorbar_width
    total_height = header_height + rows * cell_size + (rows - 1) * gap

    # White background
    comparison = Image.new('RGB', (total_width, total_height), 'white')
    draw = ImageDraw.Draw(comparison)

    # Font setup
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except:
        font = ImageFont.load_default()

    # Column headers (input, target, output, errormap)
    col_titles = ['Input', 'Target', 'Output', 'Error Map']
    for col_idx, title in enumerate(col_titles):
        x = row_label_width + col_idx * (cell_size + gap) + cell_size // 2
        bbox = draw.textbbox((0, 0), title, font=font)
        text_width = bbox[2] - bbox[0]
        draw.text((x - text_width // 2, 8), title, fill='black', font=font)

    # Place row labels and images
    row_labels = ['CNtoMCI', 'MCItoAD']
    for row_idx, prog_type in enumerate(row_labels):
        # Row label
        y_center = header_height + row_idx * (cell_size + gap) + cell_size // 2
        bbox = draw.textbbox((0, 0), prog_type, font=font)
        text_height = bbox[3] - bbox[1]
        draw.text((5, y_center - text_height // 2), prog_type, fill='black', font=font)

        # Place images
        if prog_type in images_dict:
            img_data = images_dict[prog_type]
            col_keys = ['input', 'target', 'output', 'errormap']

            for col_idx, key in enumerate(col_keys):
                if key in img_data and img_data[key] is not None:
                    img = img_data[key]

                    # Convert to RGB if grayscale
                    if img.mode == 'L':
                        img = img.convert('RGB')

                    # Resize to match cell size
                    img = img.resize((cell_size, cell_size), Image.LANCZOS)

                    x = row_label_width + col_idx * (cell_size + gap)
                    y = header_height + row_idx * (cell_size + gap)
                    comparison.paste(img, (x, y))

        # Place colorbar for each row
        colorbar_x = row_label_width + cols * cell_size + (cols - 1) * gap + gap
        colorbar_y = header_height + row_idx * (cell_size + gap)
        comparison.paste(colorbar_img, (colorbar_x, colorbar_y))

    # Save
    comparison.save(save_path)


def create_combined_comparison_grid(all_prog_data, all_prog_data_cropped, slice_num, save_path, cell_size=256, gap=5):
    """
    Generate a 4-row x 4-column comparison image.

    Row 1: CNtoMCI full
    Row 2: CNtoMCI crop
    Row 3: MCItoAD full
    Row 4: MCItoAD crop

    Each row: Input, Generated, Target, Error Map + colorbar
    """
    from PIL import ImageDraw, ImageFont

    cols = 4  # input, generated, target, errormap
    rows = 4  # CNtoMCI full, CNtoMCI crop, MCItoAD full, MCItoAD crop

    header_height = 30
    row_label_width = 120
    colorbar_width = 70

    colorbar_img = create_colorbar(cell_size, width=30, z_min=0.0, z_max=1.0, num_ticks=6)

    total_width = row_label_width + cols * cell_size + (cols - 1) * gap + gap + colorbar_width
    total_height = header_height + rows * cell_size + (rows - 1) * gap

    comparison = Image.new('RGB', (total_width, total_height), 'white')
    draw = ImageDraw.Draw(comparison)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
    except:
        font = ImageFont.load_default()

    # Column headers: Input, Target, Output, Error Map
    col_titles = ['Input', 'Target', 'Output', 'Error Map']
    for col_idx, title in enumerate(col_titles):
        x = row_label_width + col_idx * (cell_size + gap) + cell_size // 2
        bbox = draw.textbbox((0, 0), title, font=font)
        text_width = bbox[2] - bbox[0]
        draw.text((x - text_width // 2, 8), title, fill='black', font=font)

    # Row data definitions
    row_configs = [
        ('CNtoMCI (Full)', 'CNtoMCI', all_prog_data),
        ('CNtoMCI (Crop)', 'CNtoMCI', all_prog_data_cropped),
        ('MCItoAD (Full)', 'MCItoAD', all_prog_data),
        ('MCItoAD (Crop)', 'MCItoAD', all_prog_data_cropped),
    ]

    for row_idx, (row_label, prog_type, data_dict) in enumerate(row_configs):
        # Row label
        y_center = header_height + row_idx * (cell_size + gap) + cell_size // 2
        bbox = draw.textbbox((0, 0), row_label, font=font)
        text_height = bbox[3] - bbox[1]
        draw.text((5, y_center - text_height // 2), row_label, fill='black', font=font)

        # Place images
        if prog_type in data_dict and slice_num in data_dict[prog_type]:
            img_data = data_dict[prog_type][slice_num]
            col_keys = ['input', 'target', 'output', 'errormap']

            for col_idx, key in enumerate(col_keys):
                if key in img_data and img_data[key] is not None:
                    img = img_data[key]

                    if img.mode == 'L':
                        img = img.convert('RGB')

                    img = img.resize((cell_size, cell_size), Image.LANCZOS)

                    x = row_label_width + col_idx * (cell_size + gap)
                    y = header_height + row_idx * (cell_size + gap)
                    comparison.paste(img, (x, y))

        # Place colorbar for each row
        colorbar_x = row_label_width + cols * cell_size + (cols - 1) * gap + gap
        colorbar_y = header_height + row_idx * (cell_size + gap)
        comparison.paste(colorbar_img, (colorbar_x, colorbar_y))

    comparison.save(save_path)


def create_comparison_grid_with_crop_box(images_dict, images_dict_cropped, crop_coords, slice_num, save_path, cell_size=256, gap=5, inset_position='right'):
    """
    Generate a 1-row x 5-column comparison image (for a single progression type).

    Input, Target, Output: Full image + yellow crop box + cropped image inset at the bottom corner.
    Err(Out-Tgt), Err(In-Out): cropped images only.

    Args:
        images_dict: {'input': img, 'target': img, 'output': img, 'errormap': img, 'errormap_input_out': img}
        images_dict_cropped: same structure as images_dict but with cropped versions
        crop_coords: (left, top, right, bottom) - crop coordinates relative to the original image
        slice_num: slice number (used in title)
        save_path: output file path
        cell_size: size of each cell
        gap: gap between cells
        inset_position: 'right' (bottom-right) or 'left' (bottom-left)
    """
    from PIL import ImageDraw, ImageFont

    # Layout settings
    cols = 5  # input, target, output, errormap_out_target, errormap_input_out
    rows = 1  # single row

    header_height = 30  # column header height
    colorbar_width = 70  # colorbar width

    # Create colorbar
    colorbar_img = create_colorbar(cell_size, width=30, z_min=0.0, z_max=1.0, num_ticks=6)

    total_width = cols * cell_size + (cols - 1) * gap + gap + colorbar_width
    total_height = header_height + cell_size

    # White background
    comparison = Image.new('RGB', (total_width, total_height), 'white')
    draw = ImageDraw.Draw(comparison)

    # Font setup
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
    except:
        font = ImageFont.load_default()

    # Column headers
    col_titles = ['Input', 'Target', 'Output', 'Err(Out-Tgt)', 'Err(In-Out)']
    for col_idx, title in enumerate(col_titles):
        x = col_idx * (cell_size + gap) + cell_size // 2
        bbox = draw.textbbox((0, 0), title, font=font)
        text_width = bbox[2] - bbox[0]
        draw.text((x - text_width // 2, 8), title, fill='black', font=font)

    # Place images
    col_keys = ['input', 'target', 'output', 'errormap', 'errormap_input_out']
    # First 3 (input, target, output): full image + cropped inset; remaining 2: cropped only
    full_with_inset_keys = ['input', 'target', 'output']

    # Inset size for cropped image (40% of cell_size)
    inset_size = int(cell_size * 0.40)
    inset_margin = 0  # flush to the corner

    for col_idx, key in enumerate(col_keys):
        x = col_idx * (cell_size + gap)
        y = header_height

        if key in full_with_inset_keys:
            # Full image + cropped inset
            if key in images_dict and images_dict[key] is not None:
                img = images_dict[key].copy()

                # Convert to RGB if grayscale
                if img.mode == 'L':
                    img = img.convert('RGB')

                # Resize to cell size
                img = img.resize((cell_size, cell_size), Image.LANCZOS)

                # Draw yellow crop box
                img_draw = ImageDraw.Draw(img)
                scale = cell_size / 256.0
                left = int(crop_coords[0] * scale)
                top = int(crop_coords[1] * scale)
                right = int(crop_coords[2] * scale)
                bottom = int(crop_coords[3] * scale)
                # Yellow box (thickness 2)
                for i in range(2):
                    img_draw.rectangle(
                        [left - i, top - i, right + i, bottom + i],
                        outline='yellow'
                    )

                # Insert cropped image at the bottom corner
                if key in images_dict_cropped and images_dict_cropped[key] is not None:
                    cropped_img = images_dict_cropped[key].copy()
                    if cropped_img.mode == 'L':
                        cropped_img = cropped_img.convert('RGB')
                    cropped_img = cropped_img.resize((inset_size, inset_size), Image.LANCZOS)

                    # Add yellow border
                    cropped_draw = ImageDraw.Draw(cropped_img)
                    for i in range(2):
                        cropped_draw.rectangle(
                            [i, i, inset_size - 1 - i, inset_size - 1 - i],
                            outline='yellow'
                        )

                    # Determine inset position based on inset_position
                    if inset_position == 'left':
                        inset_x = inset_margin  # bottom-left
                    else:
                        inset_x = cell_size - inset_size - inset_margin  # bottom-right
                    inset_y = cell_size - inset_size - inset_margin
                    img.paste(cropped_img, (inset_x, inset_y))

                comparison.paste(img, (x, y))
        else:
            # Error map: cropped image only
            if key in images_dict_cropped and images_dict_cropped[key] is not None:
                img = images_dict_cropped[key].copy()

                # Convert to RGB if grayscale
                if img.mode == 'L':
                    img = img.convert('RGB')

                # Resize to cell size
                img = img.resize((cell_size, cell_size), Image.LANCZOS)

                comparison.paste(img, (x, y))

    # Place colorbar
    colorbar_x = cols * cell_size + (cols - 1) * gap + gap
    colorbar_y = header_height
    comparison.paste(colorbar_img, (colorbar_x, colorbar_y))

    # Save
    comparison.save(save_path)


def extract_samples():
    # Load CSV
    df = pd.read_csv(CSV_PATH)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Progression types to process
    progression_types = ['CNtoMCI', 'MCItoAD']

    # Collect data per progression type (for generating comparison images)
    all_prog_data = {}  # {prog_type: {slice_num: {'input': img, 'output': img, ...}}}
    all_prog_data_cropped = {}  # cropped version

    for prog_type in progression_types:
        print(f"\n{'='*60}")
        print(f"Processing: {prog_type}")
        print('='*60)

        # Filter rows for this progression type
        prog_df = df[df['progression_type'] == prog_type]

        if len(prog_df) == 0:
            print(f"No samples found for {prog_type}")
            continue

        # MCItoAD is fixed to a specific subject (013_S_1186, 97 months); others use the maximum months
        if prog_type == 'MCItoAD':
            subject_id = '013_S_1186'
            subject_df = prog_df[prog_df['subject_id'] == subject_id]
            if len(subject_df) == 0:
                print(f"Subject {subject_id} not found for {prog_type}")
                continue
            max_months = subject_df['months_from_first_visit'].max()
            max_samples = subject_df[subject_df['months_from_first_visit'] == max_months]
        else:
            # Find the maximum months
            max_months = prog_df['months_from_first_visit'].max()
            max_samples = prog_df[prog_df['months_from_first_visit'] == max_months]
            # Get subject_id
            subject_id = max_samples['subject_id'].iloc[0]

        print(f"Subject: {subject_id}")
        print(f"Max months: {max_months}")
        print(f"Number of slices: {len(max_samples)}")

        # Retrieve crop coordinates
        crop_coords = CROP_COORDS.get(prog_type, (40, 30, 216, 206))
        print(f"Crop coords: {crop_coords}")

        # Create output subdirectory
        prog_output_dir = os.path.join(OUTPUT_DIR, f"{prog_type}_{subject_id}_{int(max_months)}months")

        # Original image directories
        os.makedirs(os.path.join(prog_output_dir, 'input'), exist_ok=True)
        os.makedirs(os.path.join(prog_output_dir, 'output'), exist_ok=True)
        os.makedirs(os.path.join(prog_output_dir, 'target'), exist_ok=True)
        os.makedirs(os.path.join(prog_output_dir, 'errormap'), exist_ok=True)

        # Cropped image directories
        os.makedirs(os.path.join(prog_output_dir, 'input_cropped'), exist_ok=True)
        os.makedirs(os.path.join(prog_output_dir, 'output_cropped'), exist_ok=True)
        os.makedirs(os.path.join(prog_output_dir, 'target_cropped'), exist_ok=True)
        os.makedirs(os.path.join(prog_output_dir, 'errormap_cropped'), exist_ok=True)

        # Storage for images of this progression type
        all_prog_data[prog_type] = {}
        all_prog_data_cropped[prog_type] = {}

        # Process each slice
        for idx, row in max_samples.iterrows():
            # Extract filenames
            input_filename = os.path.basename(row['input_image'])
            target_filename = os.path.basename(row['edited_image'])
            output_filename = os.path.basename(row['generated_image_path'])

            # Extract slice number (e.g., 116_S_4855_0_100.png -> 100)
            slice_num = input_filename.split('_')[-1].replace('.png', '')

            # Input image path (first visit)
            input_path = os.path.join(IMAGE_BASE, input_filename)

            # Target image path (at max months)
            target_path = os.path.join(IMAGE_BASE, target_filename)

            # Generated image path (uses OUTPUT_BASE, keyed by target filename)
            output_path = os.path.join(OUTPUT_BASE, target_filename)

            # Check file existence
            if not os.path.exists(input_path):
                print(f"Warning: Input not found: {input_path}")
                continue
            if not os.path.exists(target_path):
                print(f"Warning: Target not found: {target_path}")
                continue
            if not os.path.exists(output_path):
                print(f"Warning: Output not found: {output_path}")
                continue

            # Load and resize images to a uniform 256x256
            target_size = 256
            input_img = Image.open(input_path).convert('L').resize((target_size, target_size), Image.LANCZOS)
            target_img = Image.open(target_path).convert('L').resize((target_size, target_size), Image.LANCZOS)
            output_img = Image.open(output_path).convert('L').resize((target_size, target_size), Image.LANCZOS)

            # Generate error map (output vs target) - same method as evaluate_test_csv.py
            target_array = np.array(target_img)
            output_array = np.array(output_img)
            errormap_img = compute_error_map(target_array, output_array)

            # Save original images (using original filenames)
            input_img.save(os.path.join(prog_output_dir, 'input', input_filename))
            output_img.save(os.path.join(prog_output_dir, 'output', output_filename))
            target_img.save(os.path.join(prog_output_dir, 'target', target_filename))
            errormap_img.save(os.path.join(prog_output_dir, 'errormap', input_filename.replace('.png', '_errormap.png')))

            # Save cropped images
            input_cropped = crop_and_resize(input_img, crop_coords, target_size)
            output_cropped = crop_and_resize(output_img, crop_coords, target_size)
            target_cropped = crop_and_resize(target_img, crop_coords, target_size)

            # Recompute error map using cropped images
            target_cropped_array = np.array(target_cropped)
            output_cropped_array = np.array(output_cropped)
            errormap_cropped_img = compute_error_map(target_cropped_array, output_cropped_array)

            input_cropped.save(os.path.join(prog_output_dir, 'input_cropped', input_filename))
            output_cropped.save(os.path.join(prog_output_dir, 'output_cropped', output_filename))
            target_cropped.save(os.path.join(prog_output_dir, 'target_cropped', target_filename))
            errormap_cropped_img.save(os.path.join(prog_output_dir, 'errormap_cropped', input_filename.replace('.png', '_errormap.png')))

            # Compute Input vs Output error map
            input_array = np.array(input_img)
            errormap_input_out_img = compute_error_map(input_array, output_array)

            # Compute cropped Input vs Output error map
            input_cropped_array = np.array(input_cropped)
            errormap_input_out_cropped_img = compute_error_map(input_cropped_array, output_cropped_array)

            # Store data for comparison image generation
            all_prog_data[prog_type][slice_num] = {
                'input': input_img.copy(),
                'output': output_img.copy(),
                'target': target_img.copy(),
                'errormap': errormap_img.copy(),
                'errormap_input_out': errormap_input_out_img.copy()
            }
            all_prog_data_cropped[prog_type][slice_num] = {
                'input': input_cropped.copy(),
                'output': output_cropped.copy(),
                'target': target_cropped.copy(),
                'errormap': errormap_cropped_img.copy(),
                'errormap_input_out': errormap_input_out_cropped_img.copy()
            }

        print(f"\nSaved to: {prog_output_dir}")
        print(f"  Original (256x256):")
        print(f"    - input/: input image (first visit)")
        print(f"    - target/: ground truth target image ({int(max_months)} months later)")
        print(f"    - output/: model-generated image")
        print(f"    - errormap/: output vs target difference map (inferno colormap)")
        print(f"  Cropped (256x256):")
        print(f"    - input_cropped/: cropped and resized")
        print(f"    - target_cropped/: cropped and resized")
        print(f"    - output_cropped/: cropped and resized")
        print(f"    - errormap_cropped/: difference map based on cropped images")

    # Generate comparison images
    print(f"\n{'='*60}")
    print("Creating comparison images...")
    print('='*60)

    comparison_combined_dir = os.path.join(OUTPUT_DIR, 'comparison_combined')
    cntomci_dir = os.path.join(OUTPUT_DIR, 'CNtoMCI')
    mcitoad_dir = os.path.join(OUTPUT_DIR, 'MCItoAD')
    os.makedirs(comparison_combined_dir, exist_ok=True)
    os.makedirs(cntomci_dir, exist_ok=True)
    os.makedirs(mcitoad_dir, exist_ok=True)

    # Find common slice numbers
    if 'CNtoMCI' in all_prog_data and 'MCItoAD' in all_prog_data:
        cn_slices = set(all_prog_data['CNtoMCI'].keys())
        mci_slices = set(all_prog_data['MCItoAD'].keys())
        common_slices = cn_slices & mci_slices

        print(f"Common slices: {len(common_slices)}")

        for slice_num in sorted(common_slices, key=int):
            # 4-row x 4-column combined comparison
            save_path = os.path.join(comparison_combined_dir, f"comparison_slice_{slice_num}.png")
            create_combined_comparison_grid(
                all_prog_data,
                all_prog_data_cropped,
                slice_num,
                save_path
            )

            # CNtoMCI: Full + Cropped comparison (1 row: full with yellow crop box + cropped inset)
            cntomci_crop_coords = CROP_COORDS.get('CNtoMCI', (70, 90, 186, 206))
            save_path_cntomci = os.path.join(cntomci_dir, f"slice_{slice_num}.png")
            create_comparison_grid_with_crop_box(
                all_prog_data['CNtoMCI'][slice_num],
                all_prog_data_cropped['CNtoMCI'][slice_num],
                cntomci_crop_coords,
                slice_num,
                save_path_cntomci
            )

            # MCItoAD: Full + Cropped comparison
            mcitoad_crop_coords = CROP_COORDS.get('MCItoAD', (130, 180, 181, 231))
            save_path_mcitoad = os.path.join(mcitoad_dir, f"slice_{slice_num}.png")
            create_comparison_grid_with_crop_box(
                all_prog_data['MCItoAD'][slice_num],
                all_prog_data_cropped['MCItoAD'][slice_num],
                mcitoad_crop_coords,
                slice_num,
                save_path_mcitoad,
                inset_position='right'
            )

        print(f"Comparison images saved to:")
        print(f"  - {comparison_combined_dir}")
        print(f"    (4 rows: CNtoMCI Full, CNtoMCI Crop, MCItoAD Full, MCItoAD Crop)")
        print(f"    (4 cols: Input, Target, Output, Error Map)")
        print(f"  - {cntomci_dir} (CNtoMCI: Full + Cropped inset)")
        print(f"  - {mcitoad_dir} (MCItoAD: Full + Cropped inset)")
    else:
        print("Warning: Both CNtoMCI and MCItoAD data required for comparison images")


if __name__ == '__main__':
    print("="*60)
    print("Crop coordinate settings (current values):")
    for prog_type, coords in CROP_COORDS.items():
        print(f"  {prog_type}: {coords}")
    print("="*60)
    print("To change coordinates, edit CROP_COORDS at the top of this script.")
    print("Format: (left, top, right, bottom)")
    print("="*60)

    extract_samples()

    print(f"\n{'='*60}")
    print(f"Done! Results saved to: {OUTPUT_DIR}")
    print('='*60)
