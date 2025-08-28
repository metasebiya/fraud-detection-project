import pandas as pd
from src.config.config import DATA_PATHS
import os
import logging

class DataLoader:
    """
    A class to load the required datasets for the fraud detection project.
    """
    def __init__(self):
        """
        Initializes the DataLoader.
        """

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        print("DataLoader initialized.")

    def load_data(self, dataset_paths_dict):
        """
        Loads multiple datasets from CSV files based on a dictionary of paths.

        Args:
            dataset_paths_dict (dict): A dictionary where keys are descriptive names
                                       for the datasets (e.g., 'fraud_data', 'ip_to_country')
                                       and values are their respective file paths.
                                       Example:
                                       {
                                           'fraud_data': 'Fraud_Data.csv',
                                           'ip_to_country': 'IpAddress_to_Country.csv',
                                           'creditcard_data': 'creditcard.csv'
                                       }

        Returns:
            dict: A dictionary where keys are the dataset names and values are
                  the loaded pandas DataFrames. If a file is not found, its
                  corresponding value in the dictionary will be None.
        """
        datasets = {}
        for name, path in DATA_PATHS.items():
            print(f"name:{name}, path:{path}")
            try:
                if os.path.exists(path):
                    datasets[name] = pd.read_csv(path)
                    logger.info(f"✅ Loaded {name} from {path}")
                else:
                    logger.warning(f"⚠️ File not found for '{name}' at '{path}'")
                    datasets[name] = None
            except FileNotFoundError:
                logger.error(f"❌ File not found for '{name}' at '{path}'. Setting to None.")
                datasets[name] = None
            except Exception as e:
                print(f"An unexpected error occurred while loading '{name}' from '{path}': {e}. Setting to None.")
                datasets[name] = None
        return datasets

