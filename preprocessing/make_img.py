import os
import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image, ImageOps


# Parameters
nifti_dir = 'z_adjusted'  # Root directory for NIfTI files
output_dir = 'dataset/images'  # Path to save PNG images
heights = [170] #list(range(70, 161, 10))  # Heights to extract (e.g., 100~200, step 10)
csv_path = 'Final_A_with_Descriptive_Text.csv'  # Path to the original CSV
temp_csv_path = 'dataset/csvfile/image_text_temp.csv'  # Path for the generated temp CSV
final_csv_path = 'dataset/csvfile/image_text.csv'  # Path for the final generated CSV


def save_semi_stretched_mirrored_image(slice_2d, output_path, size=(256, 256)):
    """
    Save a PNG image after vertically stretching, adding horizontal padding,
    and horizontally flipping the 2D slice.

    Args:
        slice_2d (ndarray): 2D slice data.
        output_path (str): Output file path.
        size (tuple): Final image size (default: 256x256).
    """
    # Normalize data to 0-255
    slice_2d = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d)) * 255
    slice_2d = slice_2d.astype(np.uint8)

    # Convert to PIL Image
    pil_image = Image.fromarray(slice_2d)

    # Horizontal flip
    mirrored_image = ImageOps.mirror(pil_image)

    # Vertical stretch
    stretched_height = size[1]
    aspect_ratio = slice_2d.shape[1] / slice_2d.shape[0]
    stretched_width = int(stretched_height * aspect_ratio)
    stretched_image = mirrored_image.resize((stretched_width, stretched_height), Image.Resampling.LANCZOS)

    # Add horizontal padding
    padded_image = ImageOps.pad(stretched_image, size, method=Image.Resampling.LANCZOS, color=(0))

    # Save
    padded_image.save(output_path)
    print(f"Saved image: {output_path}")


def nifti_to_png_and_csv(nifti_dir, output_dir, heights, csv_path, temp_csv_path):
    """
    Convert NIfTI data to PNG images by extracting 2D slices at specified heights.

    Args:
        nifti_dir (str): Directory containing NIfTI files.
        output_dir (str): Directory to save PNG images.
        heights (list): Slice heights to extract (Z-coordinates).
        csv_path (str): Path to the CSV file for text mapping.
        temp_csv_path (str): Path for the generated temporary CSV file.
    """
    # Step 1: Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Step 2: Read final CSV
    final_csv = pd.read_csv(csv_path)
    final_csv['Acq Date'] = pd.to_datetime(final_csv['Acq Date']).dt.strftime('%Y-%m-%d')

    image_data = []

    # Step 3: Process NIfTI files
    for subject_folder in os.listdir(nifti_dir):
        subject_path = os.path.join(nifti_dir, subject_folder)
        if not os.path.isdir(subject_path):
            continue

        for nifti_file in os.listdir(subject_path):
            if nifti_file.endswith('.nii.gz'):
                # Extract Subject and Timestamp from filename
                parts = nifti_file.replace('.nii.gz', '').split('_')
                subject_id = '_'.join(parts[:3])  # e.g., 002_S_1018
                timestamp = '_'.join(parts[3:])  # e.g., 2006-11-29_10_00_05.0
                nifti_path = os.path.join(subject_path, nifti_file)

                # Convert NIfTI data to NumPy array and fix orientation
                img = nib.load(nifti_path).get_fdata()
                oriented_data = np.flip(np.transpose(img, (1, 0, 2)), axis=0)  # Fix default orientation

                # Step 4: Extract and process 2D slices at the specified heights
                for z in heights:
                    if z >= oriented_data.shape[2]:
                        print(f"Warning: Height {z} out of bounds for {nifti_file}")
                        continue

                    slice_2d = oriented_data[:, :, z]
                    image_name = f"{subject_id}_{timestamp}_{z}.png"
                    image_path = os.path.join(output_dir, image_name)

                    # Save image (vertical stretch + horizontal padding + horizontal flip)
                    save_semi_stretched_mirrored_image(slice_2d, image_path, size=(256, 256))

                    # Text mapping
                    text_row = final_csv[(final_csv['Subject'] == subject_id) &
                                         (final_csv['Acq Date'] == timestamp.split('_')[0])]
                    if not text_row.empty:
                        text = text_row.iloc[0]['Text']# + f", Z-coordinate {z}"
                        # Force path to start with "./dataset"
                        relative_path = os.path.join("./dataset", os.path.relpath(image_path, start=output_dir))
                        image_data.append({'Image Path': relative_path, 'Subject': subject_id, 
                                           'Timestamp': timestamp, 'Z Index': z, 'Text': text})

    # Save temporary CSV
    temp_df = pd.DataFrame(image_data)
    temp_df.to_csv(temp_csv_path, index=False)
    print(f"Temporary dataset CSV saved at {temp_csv_path}")


def create_and_sort_csv(image_dir, temp_csv_path, final_csv_path):
    """
    Map PNG image paths with text data to create and sort a CSV file.

    Args:
        image_dir (str): Root directory containing PNG images.
        temp_csv_path (str): Path to the temporary CSV file.
        final_csv_path (str): Path for the final generated CSV file.
    """
    # Read final CSV
    temp_csv = pd.read_csv(temp_csv_path)

    # Sort by Z Index and additionally by Image Name
    temp_csv['Image Name'] = temp_csv['Image Path'].apply(lambda x: os.path.basename(x).split('_')[0])
    sorted_df = temp_csv.sort_values(by=['Image Name']).reset_index(drop=True)

    # Save final CSV
    sorted_df[['Image Path', 'Text']].to_csv(final_csv_path, index=False)
    print(f"Final sorted dataset CSV saved at {final_csv_path}")


if __name__ == "__main__":
    # Step 1: NIfTI to PNG and Temporary CSV
    nifti_to_png_and_csv(nifti_dir, output_dir, heights, csv_path, temp_csv_path)

    # Step 2: Create and Sort Final CSV
    create_and_sort_csv(output_dir, temp_csv_path, final_csv_path)