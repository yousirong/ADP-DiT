import pandas as pd

# Parameters
csv_path = 'Final_A_with_Descriptive_Text.csv'  # Path to the original CSV
temp_csv_path = 'image_text_temp.csv'  # Path for the generated CSV file
subject_counts = {"CN": 25, "MCI": 25, "AD": 50}  # Number of subjects to select per group

def select_subjects_by_timestamps(csv_path, subject_counts):
    """
    Select subjects per group from the CSV based on the number of timestamps.

    Args:
        csv_path (str): Path to the original CSV file.
        subject_counts (dict): Number of subjects to select per group.

    Returns:
        pd.DataFrame: DataFrame of selected subjects.
    """
    # Read original CSV
    csv_data = pd.read_csv(csv_path)

    # Convert Acq Date format
    csv_data['Acq Date'] = pd.to_datetime(csv_data['Acq Date'], format='%Y/%m/%d').dt.strftime('%Y-%m-%d')

    # Select subjects per group
    selected_subjects = []
    for group, count in subject_counts.items():
        group_data = csv_data[csv_data['Group'] == group]
        if group_data.empty:
            print(f"Warning: No subjects found for group {group}")
            continue

        # Sort subjects by timestamp count
        group_data['Timestamp Count'] = group_data.groupby('Subject')['Acq Date'].transform('count')
        group_sorted = group_data.sort_values(by=['Subject', 'Timestamp Count', 'Acq Date'], ascending=[True, False, True])
        selected_subjects.append(group_sorted.head(count * len(group_data['Acq Date'].unique())))

    return pd.concat(selected_subjects)


if __name__ == "__main__":
    # Step 1: Select subjects
    selected_subjects_df = select_subjects_by_timestamps(csv_path, subject_counts)

    # Step 2: Save temporary CSV
    selected_subjects_df.to_csv(temp_csv_path, index=False)
    print(f"Temporary dataset CSV saved at {temp_csv_path}")