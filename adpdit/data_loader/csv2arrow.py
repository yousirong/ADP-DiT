# -*- coding: utf-8 -*-
import datetime
import gc
import os
import time
from multiprocessing import Pool
import subprocess
import pandas as pd
import pyarrow as pa
from tqdm import tqdm
import hashlib
from PIL import Image
import sys


def parse_data(data):
    """
    Parse data for text-guided image-to-image.
    data[0]: input_image path
    data[1]: edited_image path
    data[2]: edit_prompt
    """
    try:
        input_img_path = data[0]
        edited_img_path = data[1]
        edit_prompt = data[2]

        # Read input image
        with open(input_img_path, "rb") as fp:
            input_image = fp.read()
            input_md5 = hashlib.md5(input_image).hexdigest()

        # Read edited image
        with open(edited_img_path, "rb") as fp:
            edited_image = fp.read()
            edited_md5 = hashlib.md5(edited_image).hexdigest()

        # Get image dimensions (assuming both images have same size)
        with Image.open(input_img_path) as f:
            width, height = f.size

        return [edit_prompt, input_md5, edited_md5, width, height, input_image, edited_image]

    except Exception as e:
        print(f'error: {e}, input_path: {data[0]}, edited_path: {data[1]}')
        return

def make_arrow(csv_root, dataset_root, start_id=0, end_id=-1):
    arrow_dir = dataset_root

    if not os.path.exists(arrow_dir):
        os.makedirs(arrow_dir)

    data = pd.read_csv(csv_root)
    # Text-guided image-to-image: input_image, edited_image, edit_prompt
    data = data[["input_image", "edited_image", "edit_prompt"]]
    columns_list = data.columns.tolist()

    if end_id < 0:
        end_id = len(data)
    print(f'start_id:{start_id}  end_id:{end_id}')
    data = data[start_id:end_id]
    num_slice = 5000
    start_sub = int(start_id / num_slice)
    sub_len = int(len(data) // num_slice)
    subs = list(range(sub_len + 1))
    for sub in tqdm(subs):
        arrow_path = os.path.join(arrow_dir, '{}.arrow'.format(str(sub + start_sub).zfill(5)))
        if os.path.exists(arrow_path):
            continue
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} start {sub + start_sub}")

        sub_data = data[sub * num_slice: (sub + 1) * num_slice].values

        bs = pool.map(parse_data, sub_data)
        bs = [b for b in bs if b]
        print(f'length of this arrow:{len(bs)}')

        # Updated columns for text-guided img2img
        columns_list = ["edit_prompt", "input_md5", "edited_md5", "width", "height", "input_image", "edited_image"]
        dataframe = pd.DataFrame(bs, columns=columns_list)
        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(arrow_path, "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
        del dataframe
        del table
        del bs
        gc.collect()


if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Usage: python hydit/data_loader/csv2arrow.py ${csv_root} ${output_arrow_data_path} ${pool_num}")
        print("csv_root: The path to your created CSV file. For more details, see https://github.com/Deep-Generative-Models-research/ADP-DiT/tree/main")
        print("output_arrow_data_path: The path for storing the created Arrow file")
        print("pool_num: The number of processes, used for multiprocessing. If you encounter memory issues, you can set pool_num to 1")
        sys.exit(1)
    csv_root = sys.argv[1]
    output_arrow_data_path = sys.argv[2]
    pool_num = int(sys.argv[3])
    pool = Pool(pool_num)
    make_arrow(csv_root, output_arrow_data_path)
