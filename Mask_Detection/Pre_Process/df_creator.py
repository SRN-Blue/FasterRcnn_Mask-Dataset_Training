import os
import pandas as pd


def get_dataframe():
    """
    Create a DataFrame containing file names from a directory.

    Returns:
    - df (pd.DataFrame): DataFrame containing file names.
    """
    # Path to the directory containing image files
    image_dir = r'.\archive_3\images'

    # List all files in the directory
    image_files = os.listdir(image_dir)

    # Extract file names without extensions
    file_names = [os.path.splitext(file)[0] for file in image_files]

    # Create a DataFrame from the list of file names
    df = pd.DataFrame({'filename': file_names})
    return df
