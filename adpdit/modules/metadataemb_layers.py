import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch

def process_metadata(metadata_csv, metadata_fields):
    """
    Process metadata CSV to prepare it for embedding.

    Parameters:
    - metadata_csv: str, path to the metadata CSV file.
    - metadata_fields: list of str, columns to use from the CSV.

    Returns:
    - metadata_tensor: torch.Tensor, processed metadata tensor for the embedder.
    """
    # Load the CSV
    df = pd.read_csv(metadata_csv)

    # Ensure the required fields exist
    missing_fields = [field for field in metadata_fields if field not in df.columns]
    if missing_fields:
        raise ValueError(f"Missing fields in metadata CSV: {missing_fields}")

    # Select and process the required fields
    processed_metadata = []
    for field in metadata_fields:
        if df[field].dtype in [np.float64, np.int64]:  # Numeric fields
            scaler = StandardScaler()
            normalized_field = scaler.fit_transform(df[[field]])
            processed_metadata.append(normalized_field)
        elif df[field].dtype == object:  # Categorical fields
            encoder = OneHotEncoder(sparse=False)
            encoded_field = encoder.fit_transform(df[[field]])
            processed_metadata.append(encoded_field)
        else:
            raise ValueError(f"Unsupported data type for field '{field}': {df[field].dtype}")

    # Concatenate all processed fields into a single array
    processed_metadata = np.hstack(processed_metadata)

    # Convert to PyTorch tensor
    metadata_tensor = torch.tensor(processed_metadata, dtype=torch.float32)

    return metadata_tensor
