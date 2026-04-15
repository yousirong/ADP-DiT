import pandas as pd
import re

# Paths to existing CSV files
existing_csv_path = "Final_A_with_Descriptive_Text.csv"  # Path to the original CSV
image_csv_path = "All_Subjects_FHQ.csv"  # CSV containing PTID
output_csv_path = "joined_output.csv"  # Path to save the merged result

# Step 1: Read CSV files
existing_df = pd.read_csv(existing_csv_path)
image_df = pd.read_csv(image_csv_path)

# Step 2: Standardize merge key column names
# Rename 'Subject' to 'PTID' to prepare for merging
existing_df.rename(columns={'Subject': 'PTID'}, inplace=True)

# Step 3: Merge the two dataframes
merged_df = pd.merge(existing_df, image_df, on='PTID', how='inner')  # 'inner' merge (common values only)

# Step 4: Save merge result
merged_df.to_csv(output_csv_path, index=False)

print(f"Merged CSV saved to {output_csv_path}")

# Step 1: Read A and image_data.csv
a_df = pd.read_csv(existing_csv_path)
image_data_df = pd.read_csv(image_csv_path)
image_data_df.rename(columns={'PTID': 'Subject'}, inplace=True)

# Step 2: Extract Subject from Image Path in A
def extract_subject(image_path):
    # Use regex to extract strings in n+_S_n+ format
    match = re.search(r'\d+_S_\d+', image_path)
    return match.group(0) if match else None

image_data_df = image_data_df.drop_duplicates(subset=["Subject"], keep="first")

# Apply function to create Subject column
a_df['Subject'] = a_df['Image Path'].apply(extract_subject)

# Step 3: Merge A with image_data.csv (left join on A)
merged_df = pd.merge(a_df, image_data_df, on="Subject", how="left")  # Left join on A

# Step 4: Drop Text column
merged_df.drop(columns=["Text"], inplace=True)

# Step 5: Save final result
merged_df.to_csv(output_csv_path, index=False)


image_data_csv_path = "All_Subjects_FHQ.csv"  # Path to image_data.csv
b_csv_path = "All_Subjects_PTDEMOG.csv"  # Path to B file
output_csv_path = "final_output_.csv"  # Final output file path

# Step 1: Read A and image_data.csv
a_df = pd.read_csv(output_csv_path)
image_data_df = pd.read_csv(image_data_csv_path)
b_df = pd.read_csv(b_csv_path)

# Step 2: Extract Subject from Image Path in A
def extract_subject(image_path):
    # Use regex to extract strings in n+_S_n+ format
    match = re.search(r'\d+_S_\d+', image_path)
    return match.group(0) if match else None

# Apply function to create Subject column
a_df['Subject'] = a_df['Image Path'].apply(extract_subject)
image_data_df.rename(columns={"PTID": "Subject"}, inplace=True)

# Step 3: Remove duplicates from image_data.csv (by Subject)
image_data_df = image_data_df.drop_duplicates(subset=["Subject"], keep="first")

# Step 4: Merge A with image_data.csv (inner join on A)
merged_df = pd.merge(a_df, image_data_df, on="Subject", how="inner")

# Step 5: Select columns up to PTRACCAT from B.csv
ptraccat_index = b_df.columns.get_loc("PTRACCAT")  # Find index of PTRACCAT column
b_df = b_df.iloc[:, :ptraccat_index + 1]  # Keep columns up to and including PTRACCAT
b_df.rename(columns={"PTID": "Subject"}, inplace=True)

# Step 6: Remove duplicates from B.csv (by Subject)
b_df = b_df.drop_duplicates(subset=["Subject"], keep="first")

# Step 7: Merge A with B
merged_df = pd.merge(merged_df, b_df, on="Subject", how="inner")

# Step 8: Handle duplicate columns (_x, _y suffixes)
# Identify and resolve duplicate columns
for col in merged_df.columns:
    if "_x" in col and col.replace("_x", "_y") in merged_df.columns:
        merged_df[col.replace("_x", "")] = merged_df[col]  # Use `_x` column and rename
        merged_df.drop(columns=[col, col.replace("_x", "_y")], inplace=True)  # Drop both `_x` and `_y`

# Step 9: Drop Text column
merged_df.drop(columns=["Text"], inplace=True)

# Step 10: Save final result
merged_df.to_csv(output_csv_path, index=False)

print(f"Merged CSV saved to {output_csv_path}")

output_csv_path = "Filtered_Data_Dict.csv"

required_columns = [
    "FHQSOURCE", "FHQPROV", "FHQMOM", "FHQMOMAD", "FHQDAD", "FHQDADAD",
    "FHQSIB", "ID", "SITEID", "USERDATE", "USERDATE2", "update_stamp",
    "PTSOURCE", "PTGENDER", "PTDOB", "PTDOBYY", "PTHAND", "PTMARRY", "PTEDUCAT",
    "PTWORKHS", "PTWORK", "PTNOTRT", "PTRTYR", "PTHOME", "PTTLANG", "PTPLANG",
    "PTADBEG", "PTCOGBEG", "PTADDX", "PTETHCAT", "PTRACCAT", "PHASE", "RID",
    "VISCODE", "VISCODE2", "VISDATE"
]

# Read CSV file
data_dict = a_df

# Filter: keep only rows where FLDNAME is in the required columns list
filtered_data = data_dict[data_dict['FLDNAME'].isin(required_columns)]

# Remove duplicates: keep only the first row per FLDNAME
filtered_data_unique = filtered_data.drop_duplicates(subset="FLDNAME", keep="first")

# Save filtered data
filtered_data_unique.to_csv(output_csv_path, index=False)

print(f"Filtered CSV saved at {output_csv_path}")


df = pd.read_csv('image_text_temp.csv')

# Read CSV
a_df = df
b_df = merged_df
output_csv_path = 'ImgText.csv'
# Extract Subject and Date from B
def extract_subject_and_date(image_path):
    match = re.search(r'(\d+_S_\d+)_(\d{4}-\d{2}-\d{2})', image_path)
    if match:
        subject = match.group(1)
        date = match.group(2)
        return pd.Series([subject, date])
    return pd.Series([None, None])

b_df[['Subject', 'Date']] = b_df['Image Path'].apply(extract_subject_and_date)

# Join A and B
merged_df = pd.merge(
    a_df, b_df,
    left_on=['Subject', 'Acq Date'],  # Subject, Acq Date from A
    right_on=['Subject', 'Date'],    # Subject, Date extracted from B
    how='inner'  # inner join
)

# Save result
merged_df.to_csv(output_csv_path, index=False)
print(f"Joined CSV saved to {output_csv_path}")

df = pd.read_csv('ImgText.csv')

columns_to_keep = [
    "Image Path", "Subject", "Month_to_Visit", "Group", "Sex", "Age", "PTGENDER", "PTETHCAT", "PTRACCAT", "PTMARRY", "FHQMOM", "FHQDAD", "FHQSIB",
    "PTHAND", "PTWORK", "PTHOME", "PTEDUCAT", "PTCOGBEG", "PTDOBYY", "PTADBEG", "PTRTYR"
]

filtered_df = df[columns_to_keep].copy()

filtered_df['CognitiveDecline_Age'] = filtered_df['PTCOGBEG'] - filtered_df['PTDOBYY']  # Age at onset of cognitive decline
filtered_df['Alzheimer_Age'] = filtered_df['PTADBEG'] - filtered_df['PTDOBYY']          # Age at Alzheimer's diagnosis
filtered_df['Cognitive_to_Alzheimer'] = filtered_df['PTADBEG'] - filtered_df['PTCOGBEG']  # Years from cognitive decline to Alzheimer's diagnosis
filtered_df['Retirement_to_Cognitive'] = filtered_df['PTCOGBEG'] - filtered_df['PTRTYR']  # Years from retirement to cognitive decline

# Define descriptions for categorical data values
category_descriptions = {
    "Group": {
    "AD": "Alzheimer's Disease",
    "MCI": "Mild Cognitive Impairment",
    "CN": "Cognitive Normal"},
    "Sex": {"M":"Male", "F":"Female"},
    "PTETHCAT": {1: "Hispanic", 2: "Non-Hispanic"},
    "PTRACCAT": {1: "White", 2: "African American", 3: "Asian", 4: "Other"},
    "PTMARRY": {1: "Married", 2: "Single", 3: "Divorced", 4: "Widowed"},
    "FHQMOM": {0: "Mother: No dementia", 1: "Mother: Dementia"},
    "FHQDAD": {0: "Father: No dementia", 1: "Father: Dementia"},
    "FHQSIB": {0: "Sibling: No dementia", 1: "Sibling: Dementia"},
    "PTHAND": {1: "Right-handed", 2: "Left-handed", 3: "Ambidextrous"},
    "PTWORK": {0: "Unemployed", 1: "Employed"},
    "PTHOME": {1: "Living independently", 2: "Living with family", 3: "Living in care facility"}
}


# Convert categorical data to text and concatenate
def generate_text(row):
    text_parts = []
    for col, mapping in category_descriptions.items():
        if col in row and not pd.isna(row[col]):
            value = row[col]
            if value in mapping:
                text_parts.append(mapping[value])
    return ", ".join(text_parts)

# Create Text column
filtered_df['Text'] = filtered_df.apply(generate_text, axis=1)
output_csv_path = 'CATARI_df.csv'
# Save result
filtered_df[['Image Path', 'Text']].to_csv(output_csv_path, index=False)
print(f"Final CSV with Text column saved to {output_csv_path}")

# Input file path
input_csv_path = "ImgText.csv"  # CSV file containing existing data
output_metadata_path = "metadata.csv"  # File containing only Image Path and numeric data

# Read
df = pd.read_csv(input_csv_path)\

# 2. Create dataframe with only Image Path and numeric columns
numeric_columns = [
    'Age', 'Month_to_Visit', 'PTEDUCAT', 'PTCOGBEG', 'PTADBEG', 'PTRTYR',
    'CognitiveDecline_Age', 'Alzheimer_Age', 'Cognitive_to_Alzheimer', 'Retirement_to_Cognitive'
]

filtered_numeric_columns = ['Image Path']

# Validate each numeric column
for col in numeric_columns:
    if col in df.columns:  # Check column exists
        # Verify all values are non-negative and no missing values
        if (df[col] >= 0).all() and not df[col].isnull().any():
            filtered_numeric_columns.append(col)

# Create filtered dataframe
metadata_df = df[filtered_numeric_columns]

# Save
metadata_df.to_csv(output_metadata_path, index=False)
print(f"Metadata CSV saved to {output_metadata_path}")

import re

df = pd.read_csv('image_Text.csv')
output_csv_path = "processed_image_Text.csv"  # Output CSV file path

# Read CSV
df = pd.read_csv(input_csv_path)

# Function to extract age and visit duration
def process_text(row):
    text_parts = row['Text'].split(', ')  # Split on ', '

    # Extract age from 3rd element
    age_part = text_parts[2]  # e.g., "90 years old"
    age_match = re.search(r'(\d+)\s*years\s*old', age_part)
    age = int(age_match.group(1)) if age_match else None

    # Extract months from first visit from 4th element
    visit_part = text_parts[3]  # e.g., "11 months from first visit"
    months_match = re.search(r'(\d+)\s*months', visit_part)
    months_from_first_visit = int(months_match.group(1)) if months_match else 0  # 0 for first visit

    # Rejoin remaining text (excluding 3rd and 4th elements)
    new_text_parts = text_parts[:2] + text_parts[4:]
    new_text = ', '.join(new_text_parts)

    return pd.Series([new_text, age, months_from_first_visit])

# Process Text and create new columns
df[['Text', 'Age', 'MonthsFromFirstVisit']] = df.apply(process_text, axis=1)

# Save result
df.to_csv(output_csv_path, index=False)
df[['Image Path', 'Text']].to_csv('IMAGE_TEXT.csv', index=False)
print(f"Processed CSV saved to {output_csv_path}")
