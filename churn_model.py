# Customer Churn Prediction Model
# Vantara Retail Solutions

import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import joblib

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "online_retail_II.xlsx"
)

MODEL_DIR = PROJECT_DIR / "models"

# Create the models folder if it does not exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Display setup information
print("Churn prediction model setup started.")

print(f"Dataset path: {DATA_PATH}")

print(f"Model folder: {MODEL_DIR}")

print("Setup completed successfully!")
# ==========================================
# STEP 1: LOAD AND PREPARE THE DATASET
# ==========================================

print("\nLoading retail dataset...")

# Load both sheets from the Excel file
sheet_1 = pd.read_excel(
    DATA_PATH,
    sheet_name="Year 2009-2010"
)

sheet_2 = pd.read_excel(
    DATA_PATH,
    sheet_name="Year 2010-2011"
)

# Combine both years
df = pd.concat(
    [sheet_1, sheet_2],
    ignore_index=True
)

print(f"Original records: {len(df):,}")

# Remove duplicate records
df = df.drop_duplicates().copy()

# Convert invoice dates to datetime
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    errors="coerce"
)

# Remove records without essential information
df = df.dropna(
    subset=["Customer ID", "InvoiceDate", "Quantity", "Price"]
).copy()

# Keep positive sales transactions only
# Returns and cancellations are excluded from this first model
df = df[
    (df["Quantity"] > 0) &
    (df["Price"] > 0)
].copy()

# Calculate transaction value
df["TotalPrice"] = df["Quantity"] * df["Price"]

# Ensure customer IDs are consistent
df["Customer ID"] = df["Customer ID"].astype(str)

print(f"Valid sales records: {len(df):,}")
print(f"Unique customers: {df['Customer ID'].nunique():,}")

# ==========================================
# STEP 2: DEFINE OBSERVATION AND CHURN PERIODS
# ==========================================

# Use the last 90 days of available data
# as the future period for identifying churn.
max_date = df["InvoiceDate"].max()

prediction_end = max_date.normalize()
snapshot_date = prediction_end - pd.Timedelta(days=90)
observation_start = snapshot_date - pd.Timedelta(days=180)

print("\nChurn period settings:")
print(f"Observation starts: {observation_start.date()}")
print(f"Snapshot date: {snapshot_date.date()}")
print(f"Prediction period ends: {prediction_end.date()}")

# Purchases during the 180-day observation period
observation_df = df[
    (df["InvoiceDate"] >= observation_start) &
    (df["InvoiceDate"] < snapshot_date)
].copy()

# Purchases during the following 90-day period
future_df = df[
    (df["InvoiceDate"] >= snapshot_date) &
    (df["InvoiceDate"] <= prediction_end)
].copy()

if observation_df.empty:
    raise ValueError(
        "No purchases found in the observation period. "
        "Please check the dataset dates."
    )

# ==========================================
# STEP 3: CREATE CUSTOMER FEATURES
# ==========================================

customer_features = observation_df.groupby(
    "Customer ID"
).agg(
    LastPurchase=("InvoiceDate", "max"),
    Frequency=("Invoice", "nunique"),
    Monetary=("TotalPrice", "sum"),
    TotalItems=("Quantity", "sum")
).reset_index()

# Days since the customer's last purchase
customer_features["Recency"] = (
    snapshot_date - customer_features["LastPurchase"]
).dt.days

# Average spending per order
customer_features["AverageOrderValue"] = (
    customer_features["Monetary"] /
    customer_features["Frequency"]
)

# ==========================================
# STEP 4: CREATE CHURN LABELS
# ==========================================

# Find customers who purchased in the future period
future_customers = set(future_df["Customer ID"].unique())

# Churned = 1 if no purchase occurs in the next 90 days
customer_features["Churned"] = (
    ~customer_features["Customer ID"].isin(future_customers)
).astype(int)

# Remove the date column because it is not a model feature
customer_features = customer_features.drop(
    columns=["LastPurchase"]
)

# Replace infinite values and remove invalid rows
customer_features = customer_features.replace(
    [np.inf, -np.inf],
    np.nan
).dropna()

print("\nCustomer dataset prepared successfully!")
print(f"Customers available for modelling: {len(customer_features):,}")
print(f"Churned customers: {customer_features['Churned'].sum():,}")
print(
    "Customers who purchased again: "
    f"{(customer_features['Churned'] == 0).sum():,}"
)

print("\nCustomer dataset preview:")
print(customer_features.head())

print("\nChurn label distribution:")
# ==========================================
# STEP 5: TRAIN THE CHURN PREDICTION MODEL
# ==========================================

print("\nPreparing model training...")

# Select the input features
feature_columns = [
    "Recency",
    "Frequency",
    "Monetary",
    "TotalItems",
    "AverageOrderValue"
]

X = customer_features[feature_columns]
y = customer_features["Churned"]

# Check that both classes exist
if y.nunique() < 2:
    raise ValueError(
        "The dataset must contain both churned and "
        "non-churned customers to train this model."
    )

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale the input features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("Model training completed successfully!")

# ==========================================
# STEP 6: EVALUATE THE MODEL
# ==========================================

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, zero_division=0
)
recall = recall_score(
    y_test, y_pred, zero_division=0
)
f1 = f1_score(
    y_test, y_pred, zero_division=0
)

print("\n========== MODEL EVALUATION ==========")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

if y_test.nunique() == 2:
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC:   {auc:.4f}")
else:
    print("ROC-AUC: unavailable because the test set has one class.")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

# ==========================================
# STEP 7: SAVE THE MODEL
# ==========================================

model_path = MODEL_DIR / "churn_model.joblib"
scaler_path = MODEL_DIR / "churn_scaler.joblib"

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print("\nModel and scaler saved successfully!")
print(f"Model location: {model_path}")
# ==========================================
# STEP 8: TRAIN A RANDOM FOREST MODEL
# ==========================================

from sklearn.ensemble import RandomForestClassifier

print("\nTraining Random Forest model...")

# Create the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Train using the same training data
rf_model.fit(X_train, y_train)

# Predict on the same test data
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
rf_accuracy = accuracy_score(y_test, rf_pred)

rf_precision = precision_score(
    y_test, rf_pred, zero_division=0
)

rf_recall = recall_score(
    y_test, rf_pred, zero_division=0
)

rf_f1 = f1_score(
    y_test, rf_pred, zero_division=0
)

rf_auc = roc_auc_score(y_test, rf_prob)

print("\n========== RANDOM FOREST RESULTS ==========")
print(f"Accuracy:  {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall:    {rf_recall:.4f}")
print(f"F1-score:  {rf_f1:.4f}")
print(f"ROC-AUC:   {rf_auc:.4f}")

# ==========================================
# STEP 9: COMPARE BOTH MODELS
# ==========================================

comparison = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC"
    ],
    "Logistic Regression": [
        accuracy,
        precision,
        recall,
        f1,
        auc
    ],
    "Random Forest": [
        rf_accuracy,
        rf_precision,
        rf_recall,
        rf_f1,
        rf_auc
    ]
})

print("\n========== MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

# ==========================================
# STEP 10: SAVE RANDOM FOREST MODEL
# ==========================================

rf_model_path = MODEL_DIR / "random_forest_churn.joblib"

joblib.dump(rf_model, rf_model_path)

print("\nRandom Forest model saved successfully!")
print(f"Model location: {rf_model_path}")
# ==========================================
# STEP 11: CONFUSION MATRIX VISUALIZATION
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# Create two charts for comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Logistic Regression confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Purchased Again", "Churned"],
    cmap="Blues",
    ax=axes[0],
    colorbar=False
)

axes[0].set_title("Logistic Regression")

# Random Forest confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    rf_pred,
    display_labels=["Purchased Again", "Churned"],
    cmap="Greens",
    ax=axes[1],
    colorbar=False
)

axes[1].set_title("Random Forest")

plt.tight_layout()

# Save the chart
chart_path = MODEL_DIR / "confusion_matrix_comparison.png"

plt.savefig(chart_path, dpi=300, bbox_inches="tight")

print("\nConfusion matrix chart saved successfully!")
print(f"Chart location: {chart_path}")

plt.show()
# ==========================================
# STEP 12: ROC CURVE COMPARISON
# ==========================================

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Calculate ROC curve values
lr_fpr, lr_tpr, _ = roc_curve(y_test, y_prob)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)

# Calculate AUC scores
lr_auc = roc_auc_score(y_test, y_prob)
rf_auc = roc_auc_score(y_test, rf_prob)

# Create the chart
plt.figure(figsize=(9, 6))

plt.plot(
    lr_fpr,
    lr_tpr,
    label=f"Logistic Regression (AUC = {lr_auc:.3f})"
)

plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest (AUC = {rf_auc:.3f})"
)

# Reference line for random guessing
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True, alpha=0.3)

# Save the chart
roc_path = MODEL_DIR / "roc_curve_comparison.png"

plt.savefig(roc_path, dpi=300, bbox_inches="tight")

print("\nROC curve saved successfully!")
print(f"Chart location: {roc_path}")
# ==========================================
# STEP 13: RANDOM FOREST FEATURE IMPORTANCE
# ==========================================

import matplotlib.pyplot as plt

# Get feature importance values
importance_values = rf_model.feature_importances_

# Create a table
importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": importance_values
})

# Sort from highest to lowest
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=True
)

print("\n========== FEATURE IMPORTANCE ==========")
print(
    importance_df.sort_values(
        by="Importance",
        ascending=False
    ).to_string(index=False)
)

# Create a horizontal bar chart
plt.figure(figsize=(9, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Customer Feature")
plt.title("Random Forest Feature Importance")
plt.tight_layout()

# Save the chart
importance_path = MODEL_DIR / "feature_importance.png"

plt.savefig(
    importance_path,
    dpi=300,
    bbox_inches="tight"
)

print("\nFeature importance chart saved successfully!")
print(f"Chart location: {importance_path}")

plt.show()

