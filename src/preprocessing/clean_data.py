import pandas as pd
import numpy as np
import socket
import struct
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DataCleaner:
    def __init__(self):
        logger.info("🧼 DataCleaner initialized.")

    def handle_missing_values(self, df, column_name, strategy='drop'):
        if column_name not in df.columns:
            logger.warning(f"Column '{column_name}' not found.")
            return df

        missing_count = df[column_name].isnull().sum()
        if missing_count > 0:
            logger.info(f"Handling {missing_count} missing values in '{column_name}' using '{strategy}' strategy.")
            if strategy == 'drop':
                df = df.dropna(subset=[column_name])
            elif strategy == 'impute':
                logger.warning(f"Imputation not implemented for '{column_name}'.")
        return df

    def convert_time_columns(self, df, time_columns):
        for col in time_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                logger.info(f"Converted '{col}' to datetime.")
            else:
                logger.warning(f"Time column '{col}' not found.")
        return df

    def remove_duplicates(self, df, df_name="DataFrame"):
        initial_rows = df.shape[0]
        df = df.drop_duplicates()
        dropped = initial_rows - df.shape[0]
        logger.info(f"Removed {dropped} duplicates from {df_name}.")
        return df

    def ip_to_int(self, ip_str):
        if pd.isna(ip_str):
            return np.nan
        try:
            return struct.unpack('>I', socket.inet_aton(ip_str))[0]
        except OSError:
            try:
                return int(float(ip_str))
            except Exception:
                return np.nan

    def int_to_ip(self, ip_int):
        if pd.isna(ip_int):
            return None
        try:
            return socket.inet_ntoa(struct.pack('>I', int(ip_int)))
        except Exception:
            return None

    def convert_ip_to_int_columns(self, df, ip_column, new_int_column):
        if ip_column not in df.columns:
            logger.warning(f"IP column '{ip_column}' not found.")
            return df
        df[new_int_column] = df[ip_column].astype(str).apply(self.ip_to_int)
        logger.info(f"Converted IPs in '{ip_column}' to '{new_int_column}'.")
        return df

    def clean_all_datasets(self, datasets_dict, config=None):
        logger.info("🚿 Starting dataset cleaning...")
        cleaned = {}

        # Fraud data
        fraud = datasets_dict.get('fraud_data')
        if fraud is not None:
            fraud = self.handle_missing_values(fraud, 'ip_address')
            fraud = self.convert_time_columns(fraud, ['signup_time', 'purchase_time'])
            fraud = self.remove_duplicates(fraud, "Fraud_Data.csv")
            fraud = self.convert_ip_to_int_columns(fraud, 'ip_address', 'ip_address_int')
            cleaned['fraud_data'] = fraud
        else:
            logger.warning("Fraud_Data.csv missing or empty.")

        # IP mapping
        ip_map = datasets_dict.get('ip_to_country')
        if ip_map is not None:
            ip_map = self.remove_duplicates(ip_map, "IpAddress_to_Country.csv")
            ip_map = self.convert_ip_to_int_columns(ip_map, 'lower_bound_ip_address', 'lower_bound_ip_address_int')
            ip_map = self.convert_ip_to_int_columns(ip_map, 'upper_bound_ip_address', 'upper_bound_ip_address_int')
            cleaned['ip_to_country'] = ip_map
        else:
            logger.warning("IpAddress_to_Country.csv missing or empty.")

        # Credit card data
        credit = datasets_dict.get('creditcard_data')
        if credit is not None:
            credit = self.remove_duplicates(credit, "creditcard.csv")
            cleaned['creditcard_data'] = credit
        else:
            logger.warning("creditcard.csv missing or empty.")

        logger.info("✅ All datasets cleaned.")
        return cleaned
