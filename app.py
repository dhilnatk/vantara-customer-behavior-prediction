from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Vantara | Customer Behavior Prediction",
    page_icon="🛍️",
    layout="wide",
)

# --------------------------------------------------
# 2. Project title
# --------------------------------------------------
st.title("Vantara Retail Solutions")
st.subheader("Customer Behavior Prediction Platform")

st.write(
    "Explore retail transactions, customer purchasing patterns, "
    "and customer value."
)

# --------------------------------------------------
# 3. Locate the Excel dataset
# --------------------------------------------------
DATA_PATH = (
    Path(__file__).parent
    / "data"
    / "raw"
    / "online_retail_II.xlsx"
)

st.divider()
st.header("Dataset Status")

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

st.success("Online Retail II dataset found!")

# --------------------------------------------------
# 4. Load both Excel sheets
# --------------------------------------------------
@st.cache_data
def load_data(file_path):
    sheet_1 = pd.read_excel(
        file_path,
        sheet_name="Year 2009-2010",
    )

    sheet_2 = pd.read_excel(
        file_path,
        sheet_name="Year 2010-2011",
    )

    data = pd.concat(
        [sheet_1, sheet_2],
        ignore_index=True,
    )

    return data


try:
    with st.spinner("Loading retail transactions..."):
        df = load_data(DATA_PATH)

except Exception as error:
    st.error("The Excel workbook could not be loaded.")
    st.exception(error)
    st.stop()

# --------------------------------------------------
# 5. Display basic dataset information
# --------------------------------------------------
st.divider()
st.header("Retail Data Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Transactions", f"{len(df):,}")

with col2:
    st.metric("Columns", len(df.columns))

with col3:
    customer_column = "Customer ID"

    if customer_column in df.columns:
        customer_count = df[customer_column].nunique()
        st.metric("Identified Customers", f"{customer_count:,}")
    else:
        st.metric("Identified Customers", "Unavailable")

# --------------------------------------------------
# 6. Preview the data
# --------------------------------------------------
st.subheader("Transaction Data Preview")
st.dataframe(df.head(10), use_container_width=True)

# --------------------------------------------------
# 7. Show missing values
# --------------------------------------------------
st.subheader("Missing Values by Column")

missing_values = df.isnull().sum()

missing_summary = (
    missing_values[missing_values > 0]
    .sort_values(ascending=False)
    .rename("Missing Values")
    .to_frame()
)

if missing_summary.empty:
    st.success("No missing values found in the loaded dataset.")
else:
    st.dataframe(missing_summary, use_container_width=True)

st.caption(
    "Initial data inspection only. Cleaning, feature engineering, "
    "and predictive modelling will follow."
)
# -------------------------------
# STEP 2: DATA CLEANING
# -------------------------------

st.divider()
st.header("Data Cleaning")

# Keep the original dataset unchanged
clean_df = df.copy()

# Count missing values before cleaning
st.subheader("Missing Values Before Cleaning")

missing_before = clean_df.isnull().sum()
missing_before = missing_before[missing_before > 0]

if missing_before.empty:
    st.success("No missing values found.")
else:
    st.dataframe(
        missing_before.rename("Missing Count").to_frame(),
        use_container_width=True
    )

# Remove duplicate transaction records
original_records = len(clean_df)

duplicates_count = clean_df.duplicated().sum()
clean_df = clean_df.drop_duplicates().copy()

# Record returns and cancellations before filtering
return_mask = clean_df["Quantity"] < 0
zero_quantity_mask = clean_df["Quantity"] == 0

returns_count = int(return_mask.sum())
zero_quantity_count = int(zero_quantity_mask.sum())

# Keep a separate record of return/cancellation transactions
returns_df = clean_df.loc[return_mask].copy()

# Identify records with invalid prices
invalid_price_mask = (
    clean_df["Price"].isna() |
    (clean_df["Price"] <= 0)
)

invalid_price_count = int(invalid_price_mask.sum())

# Prepare sales data:
# Positive quantities and positive prices represent sales here.
sales_df_clean = clean_df.loc[
    (clean_df["Quantity"] > 0) &
    (clean_df["Price"] > 0) &
    clean_df["Price"].notna()
].copy()

# Calculate transaction value for sales
sales_df_clean["TotalPrice"] = (
    sales_df_clean["Quantity"] *
    sales_df_clean["Price"]
)

# Display cleaning results
st.subheader("Cleaning Summary")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Original Records",
    f"{original_records:,}"
)

col2.metric(
    "Duplicate Records Removed",
    f"{duplicates_count:,}"
)

col3.metric(
    "Return/Cancellation Records",
    f"{returns_count:,}"
)

col4, col5 = st.columns(2)

col4.metric(
    "Zero-Quantity Records",
    f"{zero_quantity_count:,}"
)

col5.metric(
    "Invalid-Price Records",
    f"{invalid_price_count:,}"
)

st.metric(
    "Valid Sales Records",
    f"{len(sales_df_clean):,}"
)

st.subheader("Cleaned Sales Data Preview")

st.dataframe(
    sales_df_clean.head(10),
    use_container_width=True
)

st.subheader("Return and Cancellation Data Preview")

if returns_df.empty:
    st.info("No negative-quantity records were found.")
else:
    st.dataframe(
        returns_df.head(10),
        use_container_width=True
    )

st.success(
    "Initial data cleaning completed. "
    "Return/cancellation records have been preserved separately."
)

# Use sales_df_clean for subsequent sales and customer analyses.
# -------------------------------
# STEP 3: EXPLORATORY DATA ANALYSIS
# -------------------------------

st.divider()
st.header("Sales Analysis Dashboard")
sales_df = sales_df_clean.copy()

sales_df["InvoiceDate"] = pd.to_datetime(
    sales_df["InvoiceDate"],
    errors="coerce"
)

sales_df = sales_df.dropna(
    subset=["InvoiceDate"]
).copy()

sales_df["Month"] = (
    sales_df["InvoiceDate"]
    .dt.to_period("M")
    .astype(str)
)

monthly_sales = (
    sales_df.groupby("Month")["TotalPrice"]
    .sum()
    .reset_index()
)
st.subheader("Monthly Sales Trend")
st.line_chart(
    monthly_sales.set_index("Month")["TotalPrice"]
)

# Find the top 10 products by sales value
st.subheader("Top 10 Products by Sales")

product_sales = (
    sales_df.groupby("Description")["TotalPrice"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

st.bar_chart(product_sales)

# Find sales by country
st.subheader("Sales by Country")

country_sales = (
    sales_df.groupby("Country")["TotalPrice"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

st.bar_chart(country_sales)

st.success("Sales analysis completed!")

# -------------------------------
# STEP 4: CUSTOMER BEHAVIOR ANALYSIS
# -------------------------------

st.divider()
st.header("Customer Behavior Analysis")

# Exclude records without a customer ID
customer_df = sales_df.dropna(
    subset=["Customer ID"]
).copy()

# Calculate the reference date for recency
reference_date = (
    customer_df["InvoiceDate"].max()
    + pd.Timedelta(days=1)
)

# Calculate customer-level purchasing information
customer_summary = customer_df.groupby(
    "Customer ID"
).agg(
    Total_Spending=("TotalPrice", "sum"),
    Number_of_Invoices=("Invoice", "nunique"),
    Total_Items=("Quantity", "sum"),
    Last_Purchase=("InvoiceDate", "max")
).reset_index()

# Calculate recency in days
customer_summary["Recency_Days"] = (
    reference_date - customer_summary["Last_Purchase"]
).dt.days

# Calculate average order value safely
customer_summary["Average_Order_Value"] = (
    customer_summary["Total_Spending"]
    / customer_summary["Number_of_Invoices"].replace(0, 1)
)

# Display summary metrics
col1, col2, col3 = st.columns(3)

col1.metric(
    "Customers Analysed",
    f"{len(customer_summary):,}"
)

col2.metric(
    "Average Customer Spending",
    f"{customer_summary['Total_Spending'].mean():,.2f}"
)

col3.metric(
    "Average Invoices per Customer",
    f"{customer_summary['Number_of_Invoices'].mean():.2f}"
)

# Display customer information
st.subheader("Customer Summary Table")

st.dataframe(
    customer_summary.sort_values(
        "Total_Spending",
        ascending=False
    ).head(20),
    use_container_width=True
)

# Display top customers by spending
st.subheader("Top 10 Customers by Total Spending")

top_customers = customer_summary.nlargest(
    10, "Total_Spending"
)

st.bar_chart(
    top_customers.set_index("Customer ID")["Total_Spending"]
)

# Display recency information
st.subheader("Customer Recency")

st.write(
    "Recency measures the number of days since each "
    "customer's most recent purchase."
)

st.dataframe(
    customer_summary[
        ["Customer ID", "Last_Purchase", "Recency_Days"]
    ].sort_values("Recency_Days").head(20),
    use_container_width=True
)

st.success("Customer behavior analysis completed!")

# ------------------------------------
# STEP 5: CUSTOMER SEGMENTATION
# ------------------------------------

st.divider()
st.header("Customer Segmentation Using K-Means")

st.write(
    "K-Means groups customers with similar purchasing patterns. "
    "The Elbow Method and Silhouette Score help evaluate "
    "the number of clusters."
)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Select numerical customer features
feature_columns = [
    "Total_Spending",
    "Number_of_Invoices",
    "Total_Items"
]

segment_features = customer_summary[feature_columns].copy()

# Replace infinite values and identify valid rows
segment_features = segment_features.replace(
    [float("inf"), float("-inf")], float("nan")
)

valid_rows = segment_features.notna().all(axis=1)

segment_features = segment_features.loc[valid_rows]
segment_customers = customer_summary.loc[valid_rows].copy()

# Evaluate possible cluster counts
if len(segment_features) < 3:
    st.warning(
        "Not enough valid customer records for segmentation."
    )

else:
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(segment_features)

    # Elbow Method and Silhouette Score
    max_clusters = min(6, len(segment_features) - 1)

    inertia_values = []
    silhouette_values = []
    tested_clusters = []

    for k in range(2, max_clusters + 1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(scaled_features)

        # Silhouette score needs at least two distinct clusters
        if len(set(labels)) > 1:
            tested_clusters.append(k)
            inertia_values.append(model.inertia_)
            silhouette_values.append(
                silhouette_score(scaled_features, labels)
            )

    if not tested_clusters:
        st.warning(
            "Unable to evaluate clusters for this dataset."
        )

    else:
        # Display Elbow Method chart
        st.subheader("Elbow Method")

        fig, ax = plt.subplots()

        ax.plot(
            tested_clusters,
            inertia_values,
            marker="o"
        )

        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Inertia")
        ax.set_title("Elbow Method for K-Means")

        st.pyplot(fig)
        plt.close(fig)

        # Display Silhouette Scores
        st.subheader("Silhouette Score Comparison")

        score_df = pd.DataFrame({
            "Number of Clusters": tested_clusters,
            "Silhouette Score": silhouette_values
        })

        st.dataframe(
            score_df.round(4),
            use_container_width=True
        )

        # Select the cluster count with the highest score
        best_index = silhouette_values.index(
            max(silhouette_values)
        )

        best_k = tested_clusters[best_index]

        st.info(
            f"The highest silhouette score among the tested "
            f"options is for {best_k} clusters."
        )

        # Train final K-Means model
        kmeans = KMeans(
            n_clusters=best_k,
            random_state=42,
            n_init=10
        )

        segment_customers["Customer_Segment"] = (
            kmeans.fit_predict(scaled_features) + 1
        )

        # Summarize customer segments
        st.subheader("Customer Segment Summary")

        segment_summary = segment_customers.groupby(
            "Customer_Segment"
        ).agg(
            Number_of_Customers=("Customer ID", "count"),
            Average_Spending=("Total_Spending", "mean"),
            Average_Invoices=("Number_of_Invoices", "mean"),
            Average_Items=("Total_Items", "mean")
        ).reset_index()

        st.dataframe(
            segment_summary.round(2),
            use_container_width=True
        )

        # Show the size of each segment
        st.subheader("Customers in Each Segment")

        segment_counts = (
            segment_customers["Customer_Segment"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(segment_counts)

        # Explain segment characteristics
        st.subheader("Customer Segment Profiles")

        for _, row in segment_summary.iterrows():
            st.markdown(
                f"*Segment {int(row['Customer_Segment'])}* — "
                f"{int(row['Number_of_Customers']):,} customers; "
                f"average spending: "
                f"{row['Average_Spending']:,.2f}; "
                f"average invoices: "
                f"{row['Average_Invoices']:.2f}; "
                f"average items: "
                f"{row['Average_Items']:.2f}."
            )

        # Display individual customer assignments
        st.subheader("Customer Segment Details")

        st.dataframe(
            segment_customers[
                [
                    "Customer ID",
                    "Total_Spending",
                    "Number_of_Invoices",
                    "Total_Items",
                    "Customer_Segment"
                ]
            ].sort_values("Customer_Segment"),
            use_container_width=True
        )

        st.success(
            "K-Means segmentation and cluster evaluation completed!"
        )

# ------------------------------------===
# ------------------------------------
# STEP 6: CUSTOMER CHURN PREDICTION
# ------------------------------------

st.divider()
st.header("Customer Churn Prediction")

st.write(
    "Predict which customers may not make another purchase "
    "within the next 90 days using trained machine learning models."
)
# Load the trained churn prediction models
import joblib

MODEL_DIR = Path(__file__).parent / "models"

logistic_model_path = MODEL_DIR / "churn_model.joblib"
scaler_path = MODEL_DIR / "churn_scaler.joblib"
random_forest_path = MODEL_DIR / "random_forest_churn.joblib"

if (
    logistic_model_path.exists()
    and scaler_path.exists()
    and random_forest_path.exists()
):
    logistic_model = joblib.load(logistic_model_path)
    churn_scaler = joblib.load(scaler_path)
    random_forest_model = joblib.load(random_forest_path)

    st.success("Both churn prediction models loaded successfully!")
   # Prepare customer data using the same time windows as training
    st.subheader("Preparing Customer Data")

    churn_customers = customer_df.copy()

    churn_customers["Customer ID"] = (
        churn_customers["Customer ID"].astype(str)
    )

    # Define the same observation and prediction periods
    max_date = churn_customers["InvoiceDate"].max()

    prediction_end = max_date.normalize()
    snapshot_date = prediction_end - pd.Timedelta(days=90)
    observation_start = snapshot_date - pd.Timedelta(days=180)

    # Select transactions from the 180-day observation period
    observation_df = churn_customers[
        (churn_customers["InvoiceDate"] >= observation_start) &
        (churn_customers["InvoiceDate"] < snapshot_date)
    ].copy()

    if observation_df.empty:
        st.warning("No customer purchases found in the observation period.")

    else:
        # Create customer features using the training definitions
        churn_features = observation_df.groupby(
            "Customer ID"
        ).agg(
            LastPurchase=("InvoiceDate", "max"),
            Frequency=("Invoice", "nunique"),
            Monetary=("TotalPrice", "sum"),
            TotalItems=("Quantity", "sum")
        ).reset_index()

        churn_features["Recency"] = (
            snapshot_date - churn_features["LastPurchase"]
        ).dt.days

        churn_features["AverageOrderValue"] = (
            churn_features["Monetary"] /
            churn_features["Frequency"]
        )

        feature_columns = [
            "Recency",
            "Frequency",
            "Monetary",
            "TotalItems",
            "AverageOrderValue"
        ]

        X_churn = churn_features[feature_columns].copy()

        st.success(
            f"Prepared {len(X_churn):,} customers "
            "using the training observation period."
        )

        st.write(
            f"Observation period: {observation_start.date()} "
            f"to {snapshot_date.date()}"
        )

        st.dataframe(
            X_churn.head(10),
            use_container_width=True
        )
        # Generate predictions using both trained models

        st.divider()
        st.subheader("Churn Prediction Results")

        # Logistic Regression predictions
        X_scaled = churn_scaler.transform(X_churn)

        logistic_predictions = logistic_model.predict(X_scaled)
        logistic_probabilities = logistic_model.predict_proba(X_scaled)[:, 1]

        # Random Forest predictions
        rf_predictions = random_forest_model.predict(X_churn)
        rf_probabilities = random_forest_model.predict_proba(X_churn)[:, 1]

        # Store predictions
        churn_results = churn_features[["Customer ID"]].copy()

        churn_results["Logistic Regression Prediction"] = logistic_predictions
        churn_results["Logistic Regression Churn Probability"] = logistic_probabilities

        churn_results["Random Forest Prediction"] = rf_predictions
        churn_results["Random Forest Churn Probability"] = rf_probabilities

        # Convert predictions into readable labels
        churn_results["Churn Status"] = churn_results[
            "Logistic Regression Prediction"
        ].map({
            0: "Likely to Return",
            1: "Potential Churn"
        })

        st.success("Customer churn predictions generated successfully!")
        # Customer churn summary metrics
        total_customers = len(churn_results)

        potential_churn = (
            churn_results["Logistic Regression Prediction"] == 1
        ).sum()

        likely_to_return = total_customers - potential_churn

        churn_percentage = (
            potential_churn / total_customers * 100
            if total_customers > 0 else 0
        )

        st.subheader("Customer Churn Overview")

        col1, col2, col3 = st.columns(3)

        col1.metric("Customers Analysed", f"{total_customers:,}")
        col2.metric("Potential Churn", f"{potential_churn:,}")
        col3.metric("Likely to Return", f"{likely_to_return:,}")

        st.metric("Predicted Churn Rate", f"{churn_percentage:.2f}%")

        st.dataframe(
            churn_results.head(20),
            use_container_width=True
        )
# Customer Risk Ranking

        st.divider()
        st.subheader("High-Risk Customers")

        high_risk_customers = churn_results[
            churn_results["Logistic Regression Prediction"] == 1
        ].copy()

        high_risk_customers = high_risk_customers.sort_values(
            by="Logistic Regression Churn Probability",
            ascending=False
        )

        st.write(
            "Customers are ranked from highest to lowest "
            "predicted churn probability."
        )

        st.dataframe(
            high_risk_customers.head(20),
            use_container_width=True
        )
        # Download customer churn prediction report

        st.subheader("Download Prediction Report")

        csv_data = churn_results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Customer Churn Report",
            data=csv_data,
            file_name="vantara_customer_churn_report.csv",
            mime="text/csv"
        )
        # Model Performance Comparison

        st.divider()
        st.subheader("Model Performance Comparison")

        performance_data = pd.DataFrame({
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score",
                "ROC-AUC"
            ],
            "Logistic Regression": [
                0.6362,
                0.5162,
                0.7465,
                0.6104,
                0.7042
            ],
            "Random Forest": [
                0.6165,
                0.4982,
                0.6573,
                0.5668,
                0.6704
            ]
        })

        st.dataframe(
            performance_data,
            use_container_width=True
        )

        st.bar_chart(
            performance_data.set_index("Metric")
        )
else:
    st.error(
        "One or more trained model files are missing. "
        "Please check your models folder."
    )
    st.stop()