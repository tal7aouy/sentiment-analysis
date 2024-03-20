import logging
import os
import zipfile
import pandas as pd
from kaggle import KaggleApi
from sklearn.model_selection import train_test_split

from dl.utils.constants import RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


def make_dataset(split: float = 0.2, frac: float = 0.05):
    # https://www.kaggle.com/datasets/kazanova/sentiment140
    api = KaggleApi()
    api.authenticate()
    # Download the dataset from Kaggle
    logger.info("Downloading dataset from Kaggle")
    api.dataset_download_files('kazanova/sentiment140', path=RAW_DATA_DIR, unzip=False)

    # Check if the dataset has been unzipped
    raw_files = os.listdir(RAW_DATA_DIR)
    zip_files = [f for f in raw_files if f.endswith('.zip')]
    
    # If there is a zip file, unzip it
    if zip_files:
        logger.info(f"Unzipping dataset files in {RAW_DATA_DIR}")
        with zipfile.ZipFile(os.path.join(RAW_DATA_DIR, zip_files[0]), 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        raw_files = os.listdir(RAW_DATA_DIR)  # Update the file list after unzipping
    
    # Check for multiple raw data files
    if len(raw_files) > 1:
        raise ValueError(f"More than one raw data file in {RAW_DATA_DIR}")

    # Assuming there's only one CSV file after extraction
    raw_data_file = os.path.join(RAW_DATA_DIR, raw_files[0])

    if not os.path.isfile(raw_data_file):
        raise ValueError(f"{raw_data_file} is not a file")

    data = pd.read_csv(
        os.path.join(RAW_DATA_DIR, raw_data_file),
        names=["target", "ids", "date", "flag", "user", "text"],
        encoding="ISO-8859-1",
    )

    logger.info(f"Initial data size: {len(data)}")
    # keep only a portion of the data for faster training
    data = data.sample(frac=frac, random_state=0)

    logger.info(f"Fraction data size: {len(data)}")

    data = data[["text", "target"]]
    data["target"] = data["target"].replace(4, 1)

    train, test = train_test_split(data, test_size=split, random_state=0)

    logger.info(f"Train size: {len(train)}")
    logger.info(f"Test size: {len(test)}")

    if not os.path.exists(PROCESSED_DATA_DIR):
        os.mkdir(PROCESSED_DATA_DIR)

    train.to_csv(os.path.join(PROCESSED_DATA_DIR, "train.csv"), index=False, encoding="ISO-8859-1")
    test.to_csv(os.path.join(PROCESSED_DATA_DIR, "test.csv"), index=False, encoding="ISO-8859-1")

    logger.info("Successfully created train and test sets")
