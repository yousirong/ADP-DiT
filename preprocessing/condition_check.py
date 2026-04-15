# -*- coding: utf-8 -*-
import csv
import os
import re
from datetime import datetime
from collections import defaultdict

# Constants
CSV_FILE = "dataset/AD_meta/csvfile/test.csv"
IMAGE_ROOT = "dataset/AD_meta/images"

# Allowed conversion pairs
ALLOWED_PAIRS = {
    ("Cognitive Normal", "Cognitive Normal"),
    ("Cognitive Normal", "Mild Cognitive Impairment"),
    ("Mild Cognitive Impairment", "Mild Cognitive Impairment"),
    ("Mild Cognitive Impairment", "Alzheimer Disease"),
    ("Alzheimer Disease", "Alzheimer Disease")
}

def get_condition(text):
    if "Alzheimer Disease" in text:
        return "Alzheimer Disease"
    elif "Mild Cognitive Impairment" in text:
        return "Mild Cognitive Impairment"
    elif "Cognitive Normal" in text:
        return "Cognitive Normal"
    else:
        return None

def extract_images_and_prompts(csv_file, condition):
    cond2 = "first visit"
    all_image_prompts = []
    first_visit_image_prompts = []

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = row['image_path'].strip()
            text_en = row['text_en'].strip()
            abs_path = os.path.normpath(os.path.join(IMAGE_ROOT, image_path))
            all_image_prompts.append((abs_path, text_en))
            if condition in text_en and cond2 in text_en:
                first_visit_image_prompts.append((abs_path, text_en))
    return all_image_prompts, first_visit_image_prompts

def extract_latest_image_and_prompt(subject_id, all_image_prompts):
    latest_date = None
    latest_image_path = None
    latest_prompt = None

    for path, prompt in all_image_prompts:
        if f'_S_{subject_id}_' in path:
            date_match = re.search(r'_(\d{4}-\d{2}-\d{2})_', path)
            if date_match:
                try:
                    date_val = datetime.strptime(date_match.group(1), '%Y-%m-%d')
                    if latest_date is None or date_val > latest_date:
                        latest_date = date_val
                        latest_image_path = os.path.abspath(path)
                        latest_prompt = prompt
                except:
                    continue
    return latest_image_path, latest_prompt

def extract_target_path(image_path, all_image_prompts):
    match = re.search(r'\d{3}_S_(\d{4})', image_path)
    if match:
        subject_id = match.group(1)
        return subject_id, extract_latest_image_and_prompt(subject_id, all_image_prompts)
    else:
        return None, (None, None)

def count_conversion_pairs(csv_file):
    cond1_list = [
        "Cognitive Normal",
        "Mild Cognitive Impairment",
        "Alzheimer Disease"
    ]

    # set of subject IDs per conversion pair
    conversion_subject_ids = defaultdict(set)

    for condition in cond1_list:
        all_image_prompts, first_visit_image_prompts = extract_images_and_prompts(csv_file, condition)
        if not first_visit_image_prompts:
            continue

        for image_path, prompt in first_visit_image_prompts:
            subject_id, (target_path, target_prompt) = extract_target_path(image_path, all_image_prompts)
            if subject_id is None or image_path == target_path or target_prompt is None:
                continue

            input_condition = condition
            target_condition = get_condition(target_prompt)
            if target_condition is None:
                continue

            conversion_pair = (input_condition, target_condition)
            if conversion_pair in ALLOWED_PAIRS:
                conversion_subject_ids[conversion_pair].add(subject_id)

    return conversion_subject_ids

def main():
    print("Counting unique subject IDs per conversion pair...\n")
    conversion_subject_ids = count_conversion_pairs(CSV_FILE)

    print("Conversion pair unique subject counts:")
    for pair in sorted(conversion_subject_ids.keys()):
        count = len(conversion_subject_ids[pair])
        print(f"{pair[0]} → {pair[1]}: {count} subjects")

if __name__ == "__main__":
    main()
