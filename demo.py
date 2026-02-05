import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from agus_method import AgusModel

def run_demo():
    print("Loading data...")
    dm = pd.read_csv("dm.csv")
    ae = pd.read_csv("ae.csv")
    
    # Feature Engineering:
    # Target: Did the patient experience a SEVERE Adverse Event?
    # We aggregate AE data to subject level.
    
    severe_ae_subjects = ae[ae['AESEV'] == 'SEVERE']['USUBJID'].unique()
    dm['HadSevereAE'] = dm['USUBJID'].isin(severe_ae_subjects).astype(int)
    
    print(f"Target Distribution (HadSevereAE):\n{dm['HadSevereAE'].value_counts(normalize=True)}")
    
    # Features
    # We need to encode categoricals for XGBoost
    # XGBoost supports categorical data natively with `enable_categorical=True` 
    # but we need to set the dtype to 'category'.
    
    feature_cols = ["AGE", "SEX", "RACE", "ETHNIC", "ARM", "COUNTRY"]
    X = dm[feature_cols].copy()
    y = dm['HadSevereAE']
    
    for col in ["SEX", "RACE", "ETHNIC", "ARM", "COUNTRY"]:
        X[col] = X[col].astype("category")
        
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    # Initialize Agus Model
    # 200 trees, depth 2 => 200 input features to Transformer
    # Transformer d_model=64
    # RoRA rank=16
    model = AgusModel(
        n_estimators=200, 
        max_depth=2, 
        d_model=64, 
        rora_rank=16, 
        epochs=15, 
        lr=0.005
    )
    
    # Fit
    print("\n--- Training Agus Model (XGB + RandomTransformer + RoRA) ---")
    model.fit(X_train, y_train)
    
    # Predict
    print("\n--- Evaluating ---")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nDemo Completed Successfully!")

if __name__ == "__main__":
    run_demo()
