import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix, f1_score
import os
import streamlit as st
import joblib
import datetime

MODEL_PATH = "zim_amr_v2_5_model.pkl"
COLS_PATH = "model_columns_v2_5.pkl"
METRICS_PATH = "model_metrics_v2_5.pkl"

@st.cache_data
def clean_hospital_data(df):
    """ETL Pipeline: Sanitizes records for the 2023-2025 study cohort."""
    df = df.dropna(subset=['Organism', 'Antibiotic', 'Result'])
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[(df['Date'].dt.year >= 2023) & (df['Date'].dt.year <= 2025)]
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
    else:
        df['Month'], df['Year'] = 1, 2024

    text_cols = ['Organism', 'Specimen', 'Ward', 'Antibiotic', 'Result', 'Admission_Status', 'Prev_Resistance', 'Gender']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.capitalize()
    
    for col in ['Prev_Resistance', 'Admission_Status', 'Gender']:
        df[col] = df[col].fillna('Unknown') if col in df.columns else 'Unknown'

    df['Target'] = df['Result'].apply(lambda x: 1 if str(x).startswith('S') else 0)
    return df

@st.cache_resource
def train_inference_engine(force_retrain=False):
    """Scientific Pipeline with 5-Fold Cross-Validation and Model Persistence."""
    if os.path.exists(MODEL_PATH) and not force_retrain:
        try:
            return joblib.load(MODEL_PATH), joblib.load(COLS_PATH), joblib.load(METRICS_PATH)
        except: pass 

    if not os.path.exists("data.csv"): return None, None, None

    df = clean_hospital_data(pd.read_csv("data.csv"))
    features = ['Organism', 'Specimen', 'Age', 'Gender', 'Ward', 'Admission_Status', 'Prev_Resistance', 'Antibiotic', 'Month', 'Year']
    X = pd.get_dummies(df[features])
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    cv_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    
    metrics = {
        "accuracy": accuracy_score(y_test, preds)*100, 
        "cv_f1": np.mean(cv_f1)*100, 
        "sensitivity": recall_score(y_test, preds)*100, 
        "specificity": (tn/(tn+fp))*100 if (tn+fp) > 0 else 0, 
        "f1": f1_score(y_test, preds)*100, 
        "auroc": roc_auc_score(y_test, probs), 
        "total_n": len(df), # RESTORED
        "cm": [tn, fp, fn, tp]
    }

    joblib.dump(model, MODEL_PATH); joblib.dump(X.columns, COLS_PATH); joblib.dump(metrics, METRICS_PATH)
    return model, X.columns, metrics