import os
from glob import glob
import ants
import nibabel as nib
from multiprocessing import Pool

# NIfTI data paths
nifti_dir = ""
output_base_dir = ""
aligned_dir = os.path.join(output_base_dir, "aligned")
z_aligned_dir = os.path.join(output_base_dir, "z_adjusted")
error_log_file = os.path.join(output_base_dir, "error.txt")
os.makedirs(aligned_dir, exist_ok=True)
os.makedirs(z_aligned_dir, exist_ok=True)

# Initialize error log
with open(error_log_file, "w") as f:
    f.write("")

# Set reference image A
reference_a = ""
fixed_image = ants.image_read(reference_a)

# Function to process each subject in parallel
def process_subject(subject_dir):
    subject = os.path.basename(subject_dir)
    registered_subject_dir = os.path.join(aligned_dir, subject)
    z_adjusted_subject_dir = os.path.join(z_aligned_dir, subject)

    # Skip subjects that have already been processed
    if os.path.exists(z_adjusted_subject_dir):
        print(f"Subject {subject} already processed. Skipping...")
        return

    os.makedirs(registered_subject_dir, exist_ok=True)
    os.makedirs(z_adjusted_subject_dir, exist_ok=True)

    print(f"Processing subject: {subject}")

    try:
        # Group files by timestamp
        timestamps = {}
        for filepath in glob(os.path.join(subject_dir, "*.nii.gz")):
            filename = os.path.basename(filepath)
            timestamp = filename.split("_")[1]  # Extract timestamp from filename
            if timestamp not in timestamps:
                timestamps[timestamp] = []
            timestamps[timestamp].append(filepath)

        # Process each timestamp group
        for timestamp, files in timestamps.items():
            print(f"Processing timestamp: {timestamp}")

            for moving_path in files:
                moving_image = ants.image_read(moving_path)
                output_aligned_path = os.path.join(registered_subject_dir, os.path.basename(moving_path))
                output_z_aligned_path = os.path.join(z_adjusted_subject_dir, os.path.basename(moving_path))

                print(f"Registering {os.path.basename(moving_path)} to A...")
                registration = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform="SyN")
                aligned_image = registration["warpedmovout"]

                # Save aligned image (B')
                ants.image_write(aligned_image, output_aligned_path)
                print(f"Aligned image saved to: {output_aligned_path}")

                # Adjust z-translation using nibabel
                print(f"Adjusting z-axis for {os.path.basename(moving_path)}...")
                nib_image = nib.load(output_aligned_path)
                affine = nib_image.affine.copy()

                # Calculate z-translation adjustment
                z_translation = fixed_image.origin[2] - aligned_image.origin[2]
                affine[2, 3] += z_translation

                # Save z-adjusted image
                adjusted_image = nib.Nifti1Image(aligned_image.numpy(), affine)
                nib.save(adjusted_image, output_z_aligned_path)
                print(f"Z-adjusted image saved to: {output_z_aligned_path}")

    except Exception as e:
        # Log errors to error.txt
        print(f"Error processing subject {subject}: {e}")
        with open(error_log_file, "a") as f:
            f.write(f"{subject}: {e}\n")
        return

# Parallel processing setup
if __name__ == "__main__":
    nifti_dirs = sorted(glob(os.path.join(nifti_dir, "*")))
    nifti_dirs = [d for d in nifti_dirs if os.path.isdir(d)]  # Filter directories only

    # Parallel processing using Pool
    with Pool(processes=10) as pool:  # Adjust number of processes based on your system
        pool.map(process_subject, nifti_dirs)

    print("All alignments and z-adjustments completed!")