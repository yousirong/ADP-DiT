# numactl --interleave=all python metrics/metrics.py
import os
# Limit thread count to prevent excessive thread creation under numactl
os.environ["OMP_NUM_THREADS"] = "128"
os.environ["MKL_NUM_THREADS"] = "128"

import pandas as pd
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.io import imread
from skimage.transform import resize
from skimage.color import gray2rgb
from scipy.linalg import sqrtm
from torchvision import models
import torch
from torchvision.transforms import ToTensor, Compose, Normalize
import concurrent.futures
import threading

# Thread-local storage for per-thread FIDCalculator instances
thread_local = threading.local()

# FID Calculator
class FIDCalculator:
    def __init__(self):
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        self.model = self.model.float()

    def calculate_features(self, images):
        images = images.float()
        with torch.no_grad():
            features = self.model(images).cpu().numpy()
        return features

    def calculate_fid(self, real_features, generated_features):
        mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)
        diff = mu1 - mu2
        cov_mean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
        if np.iscomplexobj(cov_mean):
            cov_mean = cov_mean.real
        return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * cov_mean)

def calculate_metrics_ssim_mse(output_img, target_img, resize_dim=(256, 256)):
    """Calculate SSIM, MSE, and PSNR between output and target images."""
    # Convert grayscale to RGB
    if output_img.ndim == 2:
        output_img = gray2rgb(output_img)
    if target_img.ndim == 2:
        target_img = gray2rgb(target_img)
    output_img = output_img.astype(np.float32)
    target_img = target_img.astype(np.float32)
    channel_axis = -1 if output_img.ndim == 3 else None
    # Adjust win_size for small images
    min_dim = min(output_img.shape[0], output_img.shape[1])
    if min_dim < 7:
        win_size = min_dim
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            win_size = 3
    else:
        win_size = 7
    try:
        ssim_val = ssim(
            target_img,
            output_img,
            data_range=output_img.max() - output_img.min(),
            channel_axis=channel_axis,
            win_size=win_size
        )
    except TypeError:
        ssim_val = ssim(
            target_img,
            output_img,
            data_range=output_img.max() - output_img.min(),
            multichannel=(channel_axis is not None)
        )
    mse_val = np.mean((output_img - target_img) ** 2)
    psnr_val = psnr(target_img, output_img, data_range=output_img.max() - output_img.min())
    return ssim_val, mse_val, psnr_val

def preprocess_image_fid(img_path, img_size=299):
    """Preprocess image for FID calculation with normalization."""
    from skimage.io import imread
    from skimage.transform import resize
    from skimage.color import gray2rgb
    img = imread(img_path)
    if img.ndim == 2:
        img = gray2rgb(img)
    img = resize(img, (img_size, img_size), anti_aliasing=True)
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor

def process_row(idx, row, original_base_path, new_base_path):
    """Worker function to process each row."""
    try:
        input_path_original = row['input_image_path']
        output_path_original = row['output_image_path']
        target_path_original = row['target_image_path']
        prompt = row['prompt']

        if os.path.isabs(input_path_original):
            input_path = input_path_original.replace(original_base_path, new_base_path)
        else:
            input_path = os.path.join(new_base_path, input_path_original)
        if os.path.isabs(output_path_original):
            output_path = output_path_original.replace(original_base_path, new_base_path)
        else:
            output_path = os.path.join(new_base_path, output_path_original)
        if os.path.isabs(target_path_original):
            target_path = target_path_original.replace(original_base_path, new_base_path)
        else:
            target_path = os.path.join(new_base_path, target_path_original)
        input_path = os.path.normpath(input_path)
        output_path = os.path.normpath(output_path)
        target_path = os.path.normpath(target_path)
        print(f"\nProcessing row {idx}:")
        print(f"Input path: {input_path}")
        print(f"Output path: {output_path}")
        print(f"Target path: {target_path}")
        print(f"Prompt: {prompt}")
        if (not os.path.exists(input_path) or not os.path.exists(output_path) or not os.path.exists(target_path)):
            print(f"Row {idx}: Missing image files, skipping.")
            return None
        output_img = imread(output_path)
        target_img = imread(target_path)
        ssim_val, mse_val, psnr_val = calculate_metrics_ssim_mse(output_img, target_img)
        target_tensor = preprocess_image_fid(target_path)
        output_tensor = preprocess_image_fid(output_path)
        # Create per-thread FIDCalculator instance (reused once created)
        if not hasattr(thread_local, 'fid_calculator'):
            thread_local.fid_calculator = FIDCalculator()
        fid_calc = thread_local.fid_calculator
        target_feature = fid_calc.calculate_features(target_tensor)
        output_feature = fid_calc.calculate_features(output_tensor)
        return {
            "input_path": input_path,
            "output_path": output_path,
            "target_path": target_path,
            "prompt": prompt,
            "SSIM": ssim_val,
            "MSE": mse_val,
            "PSNR": psnr_val,
            "target_feature": target_feature,
            "output_feature": output_feature
        }
    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        return None

def evaluate_images():
    """
    Read data from results CSV files, select best SSIM results per input,
    compute FID and metrics. Save results to evaluation directory.
    """
    # Configuration - set these paths before running
    evaluations = [
        {"dataset": "ADtoAD", "input_csv": "./results/ADtoAD/results.csv"},
        {"dataset": "MCtoAD", "input_csv": "./results/MCtoAD/results.csv"},
        {"dataset": "MCtoMC", "input_csv": "./results/MCtoMC/results.csv"},
        {"dataset": "CNtoMC", "input_csv": "./results/CNtoMC/results.csv"},
        {"dataset": "CNtoCN", "input_csv": "./results/CNtoCN/results.csv"}
    ]
    evaluation_root = "./results/evaluation"
    summary_csv_path = os.path.join(evaluation_root, "evaluate.csv")
    original_base_path = '/workspace'
    new_base_path = '.'
    for eval_ in evaluations:
        dataset = eval_["dataset"]
        input_csv = eval_["input_csv"]
        print(f"\n===== Processing Dataset: {dataset} =====")
        print(f"Input CSV: {input_csv}")
        required_columns = ['input_image_path', 'output_image_path', 'target_image_path', 'prompt']
        try:
            df = pd.read_csv(input_csv, usecols=required_columns, skipinitialspace=True)
        except Exception as e:
            print(f"Error reading CSV ({input_csv}): {e}")
            continue
        df.columns = df.columns.str.strip()
        print("CSV columns:", df.columns.tolist())
        print("CSV data sample:")
        print(df.head())
        all_results = []
        # Parallel processing: submit each row to thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for idx, row in df.iterrows():
                futures.append(executor.submit(process_row, idx, row, original_base_path, new_base_path))
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res is not None:
                    all_results.append(res)
        if not all_results:
            print(f"{dataset}: No valid results, skipping FID calculation.")
            continue
        all_results_df = pd.DataFrame(all_results)
        best_results_df = (
            all_results_df.sort_values(by="SSIM", ascending=False)
                          .groupby("input_path", as_index=False)
                          .head(1)
        )
        best_target_features = np.vstack(best_results_df["target_feature"])
        best_output_features = np.vstack(best_results_df["output_feature"])
        # Compute FID as single value per dataset
        fid_calc_main = FIDCalculator()
        fid_val = fid_calc_main.calculate_fid(best_target_features, best_output_features)
        best_results_df["FID"] = fid_val
        mean_values = {
            "SSIM": best_results_df["SSIM"].mean(),
            "MSE":  best_results_df["MSE"].mean(),
            "PSNR": best_results_df["PSNR"].mean(),
            "FID":  fid_val,
            "Dataset": dataset
        }
        print(f"\nBest results mean values: {mean_values}")
        dataset_dir = os.path.join(evaluation_root, dataset)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            print(f"Created results directory: {dataset_dir}")
        best_csv_path = os.path.join(dataset_dir, f"{dataset}_best_results.csv")
        columns_to_save = [
            "input_path", "output_path", "target_path", "prompt",
            "SSIM", "MSE", "PSNR", "FID"
        ]
        best_save_df = best_results_df[columns_to_save].copy()
        if os.path.exists(best_csv_path):
            best_save_df.to_csv(best_csv_path, mode='a', index=False, header=False)
            print(f"Best results appended to: {best_csv_path}")
        else:
            best_save_df.to_csv(best_csv_path, mode='w', index=False, header=True)
            print(f"Best results saved to: {best_csv_path}")
        summary_df = pd.DataFrame([{
            "Dataset": mean_values["Dataset"],
            "SSIM": mean_values["SSIM"],
            "MSE": mean_values["MSE"],
            "PSNR": mean_values["PSNR"],
            "FID": mean_values["FID"]
        }])
        if not os.path.exists(evaluation_root):
            os.makedirs(evaluation_root)
        if os.path.exists(summary_csv_path):
            cols = ["Dataset"] + [c for c in summary_df.columns if c != "Dataset"]
            summary_df = summary_df[cols]
            summary_df.to_csv(summary_csv_path, mode='a', index=False, header=False)
            print(f"Mean values appended to: {summary_csv_path}")
        else:
            cols = ["Dataset"] + [c for c in summary_df.columns if c != "Dataset"]
            summary_df = summary_df[cols]
            summary_df.to_csv(summary_csv_path, mode='w', index=False, header=True)
            print(f"Mean values saved to: {summary_csv_path}")

if __name__ == "__main__":
    evaluate_images()
