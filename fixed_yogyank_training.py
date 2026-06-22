
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import ( mean_absolute_error,mean_squared_error, r2_score,)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# CONFIG


RANDOM_STATE = 42
DATA_PATH = "farmer_scoring_sample_yogyank_round1.csv"

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

TARGET = "target_entitlement_score"


# DATA LOADING



def load_data(path: str) -> pd.DataFrame:
    """
    Load source dataset.
    """

    df = pd.read_csv(path)

    print("=" * 60)
    print("Dataset Loaded")
    print("=" * 60)
    print(f"Rows : {len(df):,}")
    print(f"Cols : {df.shape[1]}")
    print()

    return df

# AUDIT CHECKS

def run_data_audit(df: pd.DataFrame):

    print("=" * 60)
    print("DATA AUDIT")
    print("=" * 60)

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nDuplicate Rows")
    print(df.duplicated().sum())

    print("\nTarget Summary")
    print(df[TARGET].describe())

# FEATURE AVAILABILITY REVIEW


def get_model_features(df):

    leakage_features = [
        "defaulted_in_next_12_months","farmer_id",]

    features = [ c
        for c in df.columns
        if c not in leakage_features + [TARGET]]

    return features, leakage_features


# TEMPORAL VALIDATION

def temporal_split(df):

    latest_year = df["application_year"].max()

    train_df = df[
        df["application_year"] < latest_year
    ].copy()

    valid_df = df[
        df["application_year"] == latest_year
    ].copy()

    print("\nTemporal Split")
    print(f"Train Years : {sorted(train_df.application_year.unique())}")
    print(f"Valid Year  : {latest_year}")

    print(f"Train Rows  : {len(train_df):,}")
    print(f"Valid Rows  : {len(valid_df):,}")

    return train_df, valid_df

# PREPROCESSING

def build_preprocessor(X):

    numeric_features = (
        X.select_dtypes(include=["int64", "float64"])
        .columns
        .tolist()
    )

    categorical_features = (
        X.select_dtypes(include=["object"])
        .columns
        .tolist()
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    return (
        preprocessor,
        numeric_features,
        categorical_features,
    )


# MODEL

def build_model():

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return model

# TRAINING

def train_pipeline(df):

    features, leakage = get_model_features(df)

    train_df, valid_df = temporal_split(df)

    X_train = train_df[features]
    y_train = train_df[TARGET]

    X_valid = valid_df[features]
    y_valid = valid_df[TARGET]

    (
        preprocessor,
        num_cols,
        cat_cols,
    ) = build_preprocessor(X_train)

    model = build_model()

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    print("\nTraining Model...")
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_valid)

    metrics = {
        "r2": round(
            r2_score(y_valid, preds),
            4,
        ),
        "mae": round(
            mean_absolute_error(
                y_valid,
                preds,
            ),
            4,
        ),
        "rmse": round(
            np.sqrt(
                mean_squared_error(
                    y_valid,
                    preds,
                )
            ),
            4,
        ),
    }

    print("\nValidation Metrics")
    print(metrics)

    metadata = {
        "model_name":
            "Yogyank Safe Baseline",
        "target":
            TARGET,
        "random_state":
            RANDOM_STATE,
        "features":
            features,
        "removed_leakage_features":
            leakage,
        "numeric_features":
            num_cols,
        "categorical_features":
            cat_cols,
        "validation_strategy":
            "Temporal Split",
    }

    save_artifacts(
        pipeline,
        metrics,
        metadata,
    )

    return pipeline, metrics



# ARTIFACTS

def save_artifacts(
    pipeline,
    metrics,
    metadata,
):

    model_path = (
        ARTIFACT_DIR
        / "yogyank_pipeline.pkl"
    )

    metrics_path = (
        ARTIFACT_DIR
        / "validation_metrics.json"
    )

    metadata_path = (
        ARTIFACT_DIR
        / "model_metadata.json"
    )

    joblib.dump(
        pipeline,
        model_path,
    )

    with open(
        metrics_path,
        "w",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=4,
        )

    with open(
        metadata_path,
        "w",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4,
        )

    print("\nArtifacts Saved")
    print(model_path)
    print(metrics_path)
    print(metadata_path)

#MAIN
def main():

    print(
        "\nYOGYANK ENTITLEMENT SCORE "
        "SAFE BASELINE\n"
    )

    df = load_data(DATA_PATH)

    run_data_audit(df)

    train_pipeline(df)

    print("\nTraining Completed Successfully")


if __name__ == "__main__":
    main()