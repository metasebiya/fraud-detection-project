import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner
from src.preprocessing.transform_data import DataTransformer
from src.utils.config_loader import load_yaml_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Processor:
    def __init__(self):
        self.loader = DataLoader()
        self.cleaner = DataCleaner()
        self.transformer = DataTransformer()

    def perform_eda(self, df, df_name):
        logger.info(f"📊 EDA for {df_name}")
        logger.info(df.describe())

        target_col = 'class' if 'class' in df.columns else 'Class' if 'Class' in df.columns else None
        if not target_col:
            logger.warning(f"No target column found in {df_name}")
            return

        logger.info(f"Class distribution:\n{df[target_col].value_counts()}")
        plt.figure(figsize=(6, 4))
        sns.countplot(x=target_col, data=df)
        plt.title(f'Class Distribution in {df_name}')
        plt.show()

        if df_name == 'Fraud_Data.csv':
            for col in ['source', 'browser', 'sex']:
                if col in df.columns:
                    fraud_rate = df.groupby(col)[target_col].mean().sort_values(ascending=False)
                    logger.info(f"Fraud rate by {col}:\n{fraud_rate}")
                    sns.barplot(x=col, y=target_col, data=df)
                    plt.title(f'Fraud Rate by {col}')
                    plt.show()

            for col in ['purchase_value', 'age']:
                if col in df.columns:
                    sns.histplot(data=df, x=col, hue=target_col, kde=True, bins=50)
                    plt.title(f'{col} Distribution by Class')
                    plt.show()

        elif df_name == 'creditcard.csv' and 'Amount' in df.columns:
            sns.histplot(data=df, x='Amount', hue=target_col, kde=True, bins=50)
            plt.title('Transaction Amount Distribution by Class')
            plt.show()

    def preprocess_dataset(self, X, y, preprocessor, config, name):
        logger.info(f"🔧 Preprocessing {name} dataset")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.get("test_size", 0.2),
            random_state=config.get("random_state", 42),
            stratify=y
        )

        pipeline = Pipeline([('preprocessor', preprocessor)])
        X_train_proc = pipeline.fit_transform(X_train)
        X_test_proc = pipeline.transform(X_test)

        X_train_res, y_train_res = self.transformer.handle_class_imbalance(
            X_train_proc, y_train,
            dataset_name=name,
            strategy=config.get("imbalance_strategy", "smote")
        )

        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out().tolist()
        feature_names = [str(f) for f in feature_names]

        return {
            f"X_train_{name}_resampled": X_train_res,
            f"y_train_{name}_resampled": y_train_res,
            f"X_test_{name}": X_test_proc,
            f"y_test_{name}": y_test,
            f"feature_names_{name}": feature_names
        }

    def run_pipeline(self, config):
        logger.info("🚀 Running full pipeline")

        # Load feature config
        fraud_feature_config = load_yaml_config(config["feature_config_path"])

        # Load data
        datasets = self.loader.load_data(config["data_paths"])
        fraud_df = datasets.get("fraud_data")
        ip_df = datasets.get("ip_to_country")
        credit_df = datasets.get("creditcard_data")

        if not all([fraud_df is not None, ip_df is not None, credit_df is not None]):
            raise ValueError("❌ One or more datasets failed to load.")

        # Clean data
        fraud_df = self.cleaner.clean_all_datasets({"fraud_data": fraud_df})["fraud_data"]
        ip_df = self.cleaner.clean_all_datasets({"ip_to_country": ip_df})["ip_to_country"]
        credit_df = self.cleaner.clean_all_datasets({"creditcard_data": credit_df})["creditcard_data"]

        # Transform data
        transformed = self.transformer.transform_data_for_ml(
            fraud_df, ip_df, credit_df, fraud_feature_config
        )

        logger.info("✅ Pipeline complete")
        return transformed
