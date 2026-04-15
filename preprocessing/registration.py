import os
from glob import glob
import ants
from multiprocessing import Pool

# NIfTI data paths
nifti_dir = "i"
output_base_dir = ""
error_log_file = "error.txt"
os.makedirs(output_base_dir, exist_ok=True)

# Initialize error log
with open(error_log_file, "w") as f:
    f.write("")

# Function to be executed in parallel
def process_subject(subject_dir):
    subject = os.path.basename(subject_dir)
    registered_subject_dir = os.path.join(output_base_dir, subject)

    # Skip subjects that have already been processed
    if os.path.exists(registered_subject_dir):
        print(f"Subject {subject} already processed. Skipping...")
        return

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
            print(f"  Processing timestamp: {timestamp}")

            # Directory for registered output files
            os.makedirs(registered_subject_dir, exist_ok=True)

            # Copy the reference file (use the first file found as reference)
            reference = sorted(files)[0]
            reference_output_path = os.path.join(registered_subject_dir, os.path.basename(reference))
            print(f"    Copying reference file: {os.path.basename(reference)}")
            ants.image_write(ants.image_read(reference), reference_output_path)

            # Register remaining files
            for moving in files:
                if moving != reference:
                    output_file = os.path.join(registered_subject_dir, os.path.basename(moving))

                    # Perform registration
                    fixed = ants.image_read(reference)
                    moving_image = ants.image_read(moving)

                    print(f"    Registering {os.path.basename(moving)} to {os.path.basename(reference)}...")
                    registration = ants.registration(fixed=fixed, moving=moving_image, type_of_transform="SyN")
                    aligned_image = registration["warpedmovout"]

                    # Save registered file
                    ants.image_write(aligned_image, output_file)
                    print(f"    Final aligned image saved to: {output_file}")

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
    with Pool(processes=8) as pool:  # Adjust number of processes based on your system
        pool.map(process_subject, nifti_dirs)

    print("All alignments completed!")