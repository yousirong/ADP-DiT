#!/bin/bash

input_dir=""      # Path to DICOM files
output_dir=""    # Path to save converted NIfTI files

# Create output directory
mkdir -p "$output_dir"

# Traverse DICOM directories and build a list of directories to process
find "$input_dir" -mindepth 4 -type d | while read -r dir; do
    # Add only directories that contain DICOM files
    if ls "$dir"/*.dcm >/dev/null 2>&1; then
        echo "$dir"
    fi
done > dir_list.txt

# Define parallel processing function
process_dir() {
    dir="$1"
    # Parse the path to extract Subject ID and Timestamp
    rel_path=${dir#"$input_dir/"}  # Relative path from input directory
    subject=$(echo "$rel_path" | cut -d'/' -f1)   # First directory level as Subject ID
    timestamp=$(echo "$rel_path" | cut -d'/' -f3) # Third directory level as Timestamp

    # Set output directory and filename
    subject_output_dir="$output_dir/$subject"
    mkdir -p "$subject_output_dir"
    output_file="$subject_output_dir/${subject}_${timestamp}.nii.gz"

    # Run dcm2niix
    echo "Processing Subject: $subject, Timestamp: $timestamp"
    dcm2niix -z y -o "$subject_output_dir" -f "${subject}_${timestamp}" "$dir"
}

export -f process_dir
export input_dir output_dir

# Run in parallel using GNU parallel
parallel --joblog joblog.txt -j 20 process_dir :::: dir_list.txt

echo "All conversions completed!"