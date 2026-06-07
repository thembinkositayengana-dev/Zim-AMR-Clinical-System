
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    f1_score,
    precision_score
)
import os
import streamlit as st
import joblib

MODEL_PATH = "zim_amr_v3_model.pkl"
COLS_PATH = "model_columns_v3.pkl"
METRICS_PATH = "model_metrics_v3.pkl"


@st.cache_data
def clean_hospital_data(df):
    """
    Enhanced ETL Pipeline for Zimbabwe AMR Clinical System
    """

    cleaning_report = {}

    cleaning_report["records_uploaded"] = len(df)

    required_columns = [
        "Organism",
        "Specimen",
        "Age",
        "Gender",
        "Ward",
        "Admission_Status",
        "Prev_Resistance",
        "Antibiotic",
        "Result"
    ]

    # Create missing columns if absent
    for col in [
        "Admission_Status",
        "Prev_Resistance",
        "Gender",
        "Ward",
        "Specimen"
    ]:
        if col not in df.columns:
            df[col] = "Unknown"

    # -------------------------
    # AGE IMPUTATION
    # -------------------------
    if "Age" in df.columns:
        age_missing_before = df["Age"].isna().sum()

        median_age = int(df["Age"].median())
        df["Age"] = df["Age"].fillna(median_age)
        cleaning_report["missing_age_corrected"] = age_missing_before
        cleaning_report["median_age_used"] = median_age
    else:
        df["Age"] = 39
        cleaning_report["missing_age_corrected"] = 0
        cleaning_report["median_age_used"] = 39
    cleaning_report["missing_gender_corrected"] = (
        df["Gender"].isna().sum()
    )

    cleaning_report["missing_ward_corrected"] = (
        df["Ward"].isna().sum()
    )

    cleaning_report["missing_specimen_corrected"] = (
        df["Specimen"].isna().sum()
    )

    df["Gender"] = df["Gender"].fillna("Unknown")
    df["Ward"] = df["Ward"].fillna("Unknown")
    df["Specimen"] = df["Specimen"].fillna("Unknown")
    df["Admission_Status"] = df["Admission_Status"].fillna("Unknown")
    df["Prev_Resistance"] = df["Prev_Resistance"].fillna("Unknown")

    # -------------------------
    # REMOVE CRITICAL MISSING
    # -------------------------
    before_drop = len(df)

    df = df.dropna(
        subset=[
            "Organism",
            "Antibiotic",
            "Result"
        ]
    )

    after_drop = len(df)

    cleaning_report["records_removed"] = (
        before_drop - after_drop
    )

    # -------------------------
    # DATE HANDLING
    # -------------------------
    if "Date" in df.columns:

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df = df[
            (df["Date"].dt.year >= 2023) &
            (df["Date"].dt.year <= 2025)
        ]

        df["Month"] = df["Date"].dt.month
        df["Year"] = df["Date"].dt.year

    else:

        df["Month"] = 1
        df["Year"] = 2024

    # -------------------------
    # STANDARDIZATION
    # -------------------------
    text_cols = [
        "Organism",
        "Specimen",
        "Ward",
        "Antibiotic",
        "Result",
        "Admission_Status",
        "Prev_Resistance",
        "Gender"
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.capitalize()
            )

    # -------------------------
    # TARGET CREATION
    # -------------------------
    df["Target"] = df["Result"].apply(
        lambda x: 1
        if str(x).lower().startswith("s")
        else 0
    )

    cleaning_report["records_final"] = len(df)

    cleaning_report["completeness"] = round(
        (
            len(df)
            / cleaning_report["records_uploaded"]
        ) * 100,
        2
    )

    return df, cleaning_report


@st.cache_resource
def train_inference_engine(force_retrain=False):

    if (
        os.path.exists(MODEL_PATH)
        and not force_retrain
    ):
        try:
            return (
                joblib.load(MODEL_PATH),
                joblib.load(COLS_PATH),
                joblib.load(METRICS_PATH)
            )
        except:
            pass

    if not os.path.exists("data.csv"):
        return None, None, None

    raw_df = pd.read_csv("data.csv")

    df, cleaning_report = clean_hospital_data(raw_df)

    features = [
        "Organism",
        "Specimen",
        "Age",
        "Gender",
        "Ward",
        "Admission_Status",
        "Prev_Resistance",
        "Antibiotic",
        "Month",
        "Year"
    ]

    X = pd.get_dummies(df[features])

    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42
    )

    cv_f1 = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="f1"
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    probs = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        preds
    ).ravel()

    metrics = {

        "accuracy":
            accuracy_score(y_test, preds) * 100,

        "precision":
            precision_score(
                y_test,
                preds,
                zero_division=0
            ) * 100,

        "sensitivity":
            recall_score(y_test, preds) * 100,

        "specificity":
            (
                tn / (tn + fp)
            ) * 100 if (tn + fp) > 0 else 0,

        "f1":
            f1_score(y_test, preds) * 100,

        "auroc":
            roc_auc_score(y_test, probs),

        "cv_f1":
            np.mean(cv_f1) * 100,

        "total_n":
            len(df),

        "cm":
            [tn, fp, fn, tp],

        "cleaning_report":
            cleaning_report
    }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(X.columns, COLS_PATH)
    joblib.dump(metrics, METRICS_PATH)

    return model, X.columns, metrics

    
