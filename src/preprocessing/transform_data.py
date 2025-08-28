import pandas as pd
import numpy as np
import os
import logging
import socket
import struct
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from src.utils.config_loader import load_yaml_config

fraud_config = load_yaml_config("configs/fraud_feature_config.yaml")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DataTransformer:
    def __init__(self):
        logger.info("🔄 DataTransformer initialized.")
        self.smote = SMOTE(random_state=42)

    def merge_geolocation_data(self, fraud_df, ip_df):
        logger.info("🌍 Merging geolocation data...")
        ip_df_sorted = ip_df.sort_values(by='lower_bound_ip_address_int')
        fraud_df_sorted = fraud_df.sort_values('ip_address_int')

        merged = pd.merge_asof(
            fraud_df_sorted,
            ip_df_sorted[['lower_bound_ip_address_int', 'upper_bound_ip_address_int', 'country']],
            left_on='ip_address_int',
            right_on='lower_bound_ip_address_int',
            direction='backward'
        )

        merged['country'] = merged.apply(
            lambda row: row['country'] if pd.notna(row['ip_address_int']) and
                                          row['ip_address_int'] <= row['upper_bound_ip_address_int']
                        else 'Unknown',
            axis=1
        )

        merged.drop(columns=['lower_bound_ip_address_int', 'upper_bound_ip_address_int'], inplace=True)
        logger.info(f"✅ Geolocation merged. Unique countries: {merged['country'].nunique()}")
        return merged

    def engineer_fraud_features(self, df):
        logger.info("🧪 Engineering fraud features...")
        df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')
        df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')

        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        df['time_since_signup_seconds'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
        df['time_since_signup_hours'] = df['time_since_signup_seconds'] / 3600

        df.sort_values(by='purchase_time', inplace=True)
        for col, group in [('user_transactions_24h', 'user_id'),
                           ('device_transactions_24h', 'device_id'),
                           ('ip_transactions_24h', 'ip_address')]:
            df[col] = df.groupby(group)['purchase_time'].transform(
                lambda x: x.set_axis(x).rolling('24h', closed='right').count() - 1
            ).fillna(0)

        return df

    def prepare_fraud_data_for_modeling(self, df, feature_config):
        logger.info("⚙️ Preparing fraud data for modeling...")
        num_features = feature_config['numerical']
        cat_features = feature_config['categorical']

        for col in cat_features:
            if col in df.columns:
                df[col] = df[col].astype('category')
            else:
                logger.warning(f"Missing categorical feature: {col}")

        X = df.drop(columns=['user_id', 'signup_time', 'purchase_time', 'device_id',
                             'ip_address', 'ip_address_int', 'class'])
        y = df['class']

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])

        return X, y, preprocessor, num_features + cat_features

    def prepare_creditcard_data_for_modeling(self, df):
        logger.info("⚙️ Preparing credit card data for modeling...")
        num_features = df.drop(columns=['Class']).columns.tolist()
        X = df.drop(columns=['Class'])
        y = df['Class']

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_features)
        ])

        return X, y, preprocessor, num_features

    def handle_class_imbalance(self, X, y, strategy="smote", dataset_name=""):
        logger.info(f"📉 Handling class imbalance for {dataset_name} using {strategy.upper()}")
        logger.info(f"Original distribution:\n{y.value_counts()}")

        sampler = SMOTE(random_state=42) if strategy == "smote" else RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        logger.info(f"Resampled distribution:\n{y_resampled.value_counts()}")
        return X_resampled, y_resampled

    def transform_data_for_ml(self, fraud_df, ip_df, credit_df, fraud_config):
        logger.info("🚀 Starting full transformation pipeline...")

        # Fraud pipeline
        fraud_merged = self.merge_geolocation_data(fraud_df.copy(), ip_df.copy())
        fraud_engineered = self.engineer_fraud_features(fraud_merged.copy())
        X_fraud, y_fraud, pre_fraud, fraud_features = self.prepare_fraud_data_for_modeling(fraud_engineered, fraud_config)

        X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(
            X_fraud, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud
        )

        pipe_fraud = Pipeline([('preprocessor', pre_fraud)])
        X_train_fraud_proc = pipe_fraud.fit_transform(X_train_fraud)
        X_test_fraud_proc = pipe_fraud.transform(X_test_fraud)
        fraud_features_out = pipe_fraud.named_steps['preprocessor'].get_feature_names_out().tolist()

        X_train_fraud_res, y_train_fraud_res = self.handle_class_imbalance(
            X_train_fraud_proc, y_train_fraud, "Fraud_Data.csv"
        )

        # Credit card pipeline
        X_cc, y_cc, pre_cc, cc_features = self.prepare_creditcard_data_for_modeling(credit_df.copy())
        X_train_cc, X_test_cc, y_train_cc, y_test_cc = train_test_split(
            X_cc, y_cc, test_size=0.2, random_state=42, stratify=y_cc
        )

        pipe_cc = Pipeline([('preprocessor', pre_cc)])
        X_train_cc_proc = pipe_cc.fit_transform(X_train_cc)
        X_test_cc_proc = pipe_cc.transform(X_test_cc)
        cc_features_out = pipe_cc.named_steps['preprocessor'].get_feature_names_out().tolist()

        X_train_cc_res, y_train_cc_res = self.handle_class_imbalance(
            X_train_cc_proc, y_train_cc, "creditcard.csv"
        )

        logger.info("✅ Transformation complete.")
        return {
            'X_train_fraud_resampled': X_train_fraud_res,
            'y_train_fraud_resampled': y_train_fraud_res,
            'X_test_fraud_processed': X_test_fraud_proc,
            'y_test_fraud': y_test_fraud,
            'fraud_feature_names_out': fraud_features_out,
            'X_train_creditcard_resampled': X_train_cc_res,
            'y_train_creditcard_resampled': y_train_cc_res,
            'X_test_creditcard_processed': X_test_cc_proc,
            'y_test_creditcard': y_test_cc,
            'cc_feature_names_out': cc_features_out
        }
