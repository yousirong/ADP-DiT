import pandas as pd
import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
from skimage.color import gray2rgb, rgb2gray
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.feature import canny
from scipy.linalg import sqrtm
from torchvision import models
import torch
from torchvision.transforms import ToTensor
from matplotlib.colors import LinearSegmentedColormap
import textwrap
import numpy.ma as ma

def generate_difference_map(
    input_path, output_path, target_path, difference_map_path, prompt="No Prompt"
):
    """
    Load input, output, and target images, visualize them in 4 subplots
    (input/target/output/difference), and compute edge-based difference map
    using a sequential colormap ("hot"). Background (value=0) is shown in gray,
    and the color bar range is 0-100.
    """
    input_img = imread(input_path)
    output_img = imread(output_path)
    target_img = imread(target_path)

    # Convert 2D to 3D if needed
    if input_img.ndim == 2:
        input_img = gray2rgb(input_img)
    if output_img.ndim == 2:
        output_img = gray2rgb(output_img)
    if target_img.ndim == 2:
        target_img = gray2rgb(target_img)

    height, width = input_img.shape[:2]
    extent_val = [0, width, height, 0]

    output_img_f = output_img.astype(np.float32)
    target_img_f = target_img.astype(np.float32)

    # Compute difference (grayscale-based)
    output_gray = rgb2gray(output_img_f)
    target_gray = rgb2gray(target_img_f)
    diff_map_gray = np.abs(output_gray - target_gray)

    # Edge extraction (canny)
    edges = canny(target_gray / 255.0, sigma=1.0)

    # Keep only edge-region differences
    diff_map_edges = diff_map_gray * edges.astype(np.float32)

    if diff_map_edges.shape != (height, width):
        diff_map_edges = resize(diff_map_edges, (height, width), preserve_range=True)

    # Colormap: "hot" with gray for masked (zero) regions
    cmap_seq = plt.get_cmap("hot").copy()
    cmap_seq.set_bad("gray")

    # Visualization (1x4 subplot)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(input_img.astype(np.uint8), extent=extent_val)
    axes[0].set_title("Input Image", pad=15)
    axes[0].axis("off")

    axes[1].imshow(target_img.astype(np.uint8), extent=extent_val)
    axes[1].set_title("Target Image", pad=15)
    axes[1].axis("off")

    axes[2].imshow(output_img.astype(np.uint8), extent=extent_val)
    axes[2].set_title("Output Image", pad=15)
    axes[2].axis("off")

    diff_map_display = diff_map_edges.clip(0, 255).astype(np.float32)
    masked_diff = ma.masked_equal(diff_map_display, 0)

    im = axes[3].imshow(masked_diff, cmap=cmap_seq, vmin=0, vmax=100, extent=extent_val)
    axes[3].set_title("Difference Map", pad=15)
    axes[3].axis("off")

    cb = fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    cb.set_ticks([0, 25, 50, 75, 100])

    wrapped_prompt = "\n".join(textwrap.wrap(prompt, width=200))
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    fig.suptitle(wrapped_prompt, x=0.5, y=0.98, fontsize=14)

    plt.savefig(difference_map_path, dpi=600)
    plt.close(fig)
    print(f"Difference map saved to: {difference_map_path}")


def create_csv_if_not_exists(csv_path, required_columns):
    parent_dir = os.path.dirname(csv_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        empty_df = pd.DataFrame(columns=required_columns)
        empty_df.to_csv(csv_path, index=False)


def generate_evaluation_summary(summary_results, summary_csv_path):
    parent_dir = os.path.dirname(summary_csv_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    try:
        summary_df = pd.DataFrame(summary_results)
        if 'Dataset' in summary_df.columns:
            columns = ['Dataset'] + [col for col in summary_df.columns if col != 'Dataset']
            summary_df = summary_df[columns]

        if os.path.exists(summary_csv_path):
            summary_df.to_csv(summary_csv_path, mode='a', index=False, header=False)
            print(f"Mean values appended to: {summary_csv_path}")
        else:
            summary_df.to_csv(summary_csv_path, mode='w', index=False, header=True)
            print(f"Mean values saved to: {summary_csv_path}")
    except Exception as e:
        print(f"Error saving evaluation summary CSV: {e}")


def create_composite_image(diff_results, diff_dir, dataset):
    """
    Combine multiple difference map images into a single composite PNG
    (vertical layout).
    """
    if not diff_results:
        print("No difference map results, skipping composite image generation.")
        return

    n = len(diff_results)
    fig, axes = plt.subplots(n, 1, figsize=(5, 5*n))

    if n == 1:
        axes = [axes]

    for i, item in enumerate(diff_results):
        diff_map_path = item["difference_map"]
        diff_img = imread(diff_map_path)
        axes[i].imshow(diff_img)
        axes[i].axis("off")
        axes[i].set_title(os.path.basename(diff_map_path), fontsize=8)

    composite_path = os.path.join(diff_dir, f"{dataset}_composite_diff.png")
    plt.tight_layout()
    plt.savefig(composite_path, dpi=300)
    plt.close(fig)
    print(f"Composite difference map saved to: {composite_path}")


def process_dataset(dataset_info, summary_results, top_percentage=5):
    """
    Process a dataset: read CSV, select top percentage by SSIM,
    generate difference maps, and compute summary statistics.
    """
    dataset = dataset_info["dataset"]
    input_csv = dataset_info["input_csv"]

    print(f"\n===== Processing Dataset: {dataset} =====")
    print(f"Input CSV: {input_csv}")

    required_columns = ['input_path', 'output_path', 'target_path', 'prompt', 'SSIM', 'MSE', 'PSNR', 'FID']
    try:
        df = pd.read_csv(input_csv, usecols=required_columns, skipinitialspace=True)
    except FileNotFoundError:
        print(f"Input CSV file not found: {input_csv}")
        return []
    except ValueError as ve:
        print(f"Input CSV missing required columns: {ve}")
        return []
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []

    if df.empty:
        print(f"No data in CSV: {input_csv}")
        return []

    # Select top N% by SSIM
    top_n = max(1, int(len(df) * top_percentage / 100))
    top_df = df.nlargest(top_n, 'SSIM')
    best_idx = top_df['SSIM'].idxmax()

    print(f"Total image pairs: {len(df)}")
    print(f"Selected top {top_percentage}% ({len(top_df)}) image pairs")

    # Create evaluation directory
    evaluation_root = "./results/evaluation"
    dataset_dir = os.path.join(evaluation_root, dataset)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"Created results directory: {dataset_dir}")

    diff_dir = os.path.join(dataset_dir, "diff_maps")
    if not os.path.exists(diff_dir):
        os.makedirs(diff_dir)

    diff_results = []

    # Generate difference maps
    for idx, row in top_df.iterrows():
        input_path = row['input_path']
        output_path = row['output_path']
        target_path = row['target_path']
        prompt = row.get('prompt', 'No Prompt')

        input_name = os.path.splitext(os.path.basename(input_path))[0]
        target_name = os.path.splitext(os.path.basename(target_path))[0]
        # Mark best result with "best_" prefix
        if idx == best_idx:
            difference_map_path = os.path.join(diff_dir, f"best_{input_name}_{target_name}.png")
        else:
            difference_map_path = os.path.join(diff_dir, f"{input_name}_{target_name}.png")

        generate_difference_map(input_path, output_path, target_path, difference_map_path, prompt=prompt)

        diff_results.append({
            "input_image": input_path,
            "output_image": output_path,
            "target_image": target_path,
            "difference_map": difference_map_path,
            "Dataset": dataset,
            "prompt": prompt
        })

    # Save difference map results CSV
    diff_csv_path = os.path.join(dataset_dir, f"{dataset}_diff_results.csv")
    diff_df = pd.DataFrame(diff_results)
    try:
        if os.path.exists(diff_csv_path):
            diff_df.to_csv(diff_csv_path, mode='a', index=False, header=False)
        else:
            diff_df.to_csv(diff_csv_path, mode='w', index=False, header=True)
        print(f"Difference map results saved to: {diff_csv_path}")
    except Exception as e:
        print(f"Error saving difference map results CSV: {e}")

    # Compute mean values
    mean_values = top_df[['SSIM', 'MSE', 'PSNR']].mean().to_dict()
    mean_values['FID'] = top_df['FID'].mean()
    mean_values['Dataset'] = dataset
    print(f"\nMean values: {mean_values}")

    summary_results.append(mean_values)

    create_composite_image(diff_results, diff_dir, dataset)

    return diff_results


def main():
    """
    Run evaluation: read CSV files, generate difference maps,
    and save results to the evaluation directory.
    """
    # Configuration - set these paths before running
    evaluations = [
        {
            "dataset": "ADtoAD",
            "input_csv": "./results/evaluation/ADtoAD/ADtoAD_best_results.csv",
        },
        {
            "dataset": "MCtoAD",
            "input_csv": "./results/evaluation/MCtoAD/MCtoAD_best_results.csv",
        },
        {
            "dataset": "MCtoMC",
            "input_csv": "./results/evaluation/MCtoMC/MCtoMC_best_results.csv",
        },
        {
            "dataset": "CNtoMC",
            "input_csv": "./results/evaluation/CNtoMC/CNtoMC_best_results.csv",
        },
        {
            "dataset": "CNtoCN",
            "input_csv": "./results/evaluation/CNtoCN/CNtoCN_best_results.csv",
        }
    ]

    evaluation_root = "./results/evaluation"
    summary_csv_path = os.path.join(evaluation_root, "evaluate.csv")

    summary_results = []

    for eval_ in evaluations:
        process_dataset(eval_, summary_results, top_percentage=5)

    generate_evaluation_summary(summary_results, summary_csv_path)


if __name__ == "__main__":
    main()
