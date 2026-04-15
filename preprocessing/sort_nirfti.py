import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

def align_single_nifti(input_file, reference_path, output_file):
    """
    Align a single .nii.gz file to the reference image using FSL flirt.

    Parameters:
        input_file (str): Path to the input .nii.gz file.
        reference_path (str): Path to the reference .nii.gz file.
        output_file (str): Path to save the aligned .nii.gz file.
    """
    flirt_command = [
        "flirt",
        "-in", input_file,
        "-ref", reference_path,
        "-out", output_file,
        "-applyxfm"
    ]

    try:
        subprocess.run(flirt_command, check=True)
        print(f"Aligned {input_file} to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error aligning {input_file}: {e}")

def align_all_nifti_parallel(base_dir, reference_path, output_dir, max_workers=4):
    """
    Align all .nii.gz files in a nested directory structure to a reference image using parallel processing.

    Parameters:
        base_dir (str): Path to the root directory containing subject subdirectories.
        reference_path (str): Path to the reference .nii.gz file.
        output_dir (str): Path to the output directory where aligned files will be saved.
        max_workers (int): Maximum number of worker processes.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tasks = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for subject_id in os.listdir(base_dir):
            subject_path = os.path.join(base_dir, subject_id)

            if os.path.isdir(subject_path):
                subject_output_dir = os.path.join(output_dir, subject_id)
                os.makedirs(subject_output_dir, exist_ok=True)

                for file_name in os.listdir(subject_path):
                    if file_name.endswith(".nii.gz"):
                        input_file = os.path.join(subject_path, file_name)
                        output_file = os.path.join(subject_output_dir, file_name)
                        tasks.append(executor.submit(align_single_nifti, input_file, reference_path, output_file))

        # Wait for all tasks to complete
        for task in tasks:
            task.result()

# Example usage
base_directory = ""
reference_nifti = ""
output_directory = ""

# Run with 4 parallel workers
align_all_nifti_parallel(base_directory, reference_nifti, output_directory, max_workers=10)