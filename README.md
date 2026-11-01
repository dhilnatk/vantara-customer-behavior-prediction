Vantara Retail Solutions

Customer Behavior Prediction Platform

Project Overview

Vantara Customer Behavior Prediction Platform is a machine learning-based retail analytics application designed to analyse customer purchasing patterns, segment customers, and predict potential customer churn.

The platform helps businesses understand customer behaviour, identify customers who may stop purchasing, and support data-driven customer retention strategies.

Project Objectives

- Analyse retail transaction data.
- Clean and prepare customer sales records.
- Identify sales trends and purchasing patterns.
- Segment customers using K-Means clustering.
- Predict potential customer churn using machine learning.
- Compare the performance of different prediction models.
- Identify high-risk customers.
- Generate downloadable customer prediction reports.

Dataset

Dataset: Online Retail II

The dataset contains retail transaction records from two periods:

- Year 2009–2010
- Year 2010–2011

The dataset includes information such as invoice numbers, product descriptions, quantities, prices, invoice dates, customer IDs, and countries.

### Dataset Availability

The Online Retail II dataset is required to run this project. The original dataset is not included in this GitHub repository because it exceeds GitHub's file upload size limit.

To run the project, obtain the dataset separately and place it inside the data/raw/ directory with the filename online_retail_II.xlsx.

Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Joblib
- OpenPyXL

Project Modules

1. Dataset Overview

- Loads and displays retail transaction data.
- Shows dataset dimensions and missing values.

2. Data Cleaning

- Removes duplicate records.
- Separates return and cancellation transactions.
- Filters invalid sales records.
- Calculates total transaction value.

3. Sales Analysis

- Monthly sales trends.
- Top-selling products by sales value.
- Sales distribution by country.

4. Customer Behavior Analysis

- Total customer spending.
- Number of invoices.
- Total items purchased.
- Customer recency.
- Average order value.

5. Customer Segmentation

- Uses K-Means clustering.
- Applies StandardScaler for feature scaling.
- Evaluates clusters using the Elbow Method and Silhouette Score.
- Displays customer segment profiles.

6. Customer Churn Prediction

Two machine learning algorithms are implemented:

- Logistic Regression
- Random Forest Classifier

Customer features include:

- Recency
- Frequency
- Monetary
- Total Items
- Average Order Value

For this project, churn is defined as a customer making no recorded purchase during the following 90-day period.

7. Customer Risk Analysis

- Displays predicted churn status.
- Calculates predicted churn probabilities.
- Ranks high-risk customers.
- Displays customer churn summary metrics.

8. Model Evaluation

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- ROC Curve
- Random Forest Feature Importance

9. Report Generation

Users can download customer churn predictions as a CSV report.

Model Performance

The following results were obtained during model evaluation:

Evaluation Metric| Logistic Regression| Random Forest
Accuracy| 63.62%| 61.65%
Precision| 51.62%| 49.82%
Recall| 74.65%| 65.73%
F1-Score| 61.04%| 56.68%
ROC-AUC| 70.42%| 67.04%

Logistic Regression achieved better overall results than Random Forest on the reported test split.

Project Structure

customer-behavior-prediction/
│
├── data/
│   └── raw/
│       └── online_retail_II.xlsx
│
├── models/
│   ├── churn_model.joblib
│   ├── churn_scaler.joblib
│   ├── random_forest_churn.joblib
│   ├── confusion_matrix_comparison.png
│   ├── roc_curve_comparison.png
│   └── feature_importance.png
│
├── app.py
├── churn_model.py
└── README.md

Installation and Setup

1. Clone the repository

git clone YOUR_GITHUB_REPOSITORY_LINK

2. Navigate to the project directory

cd customer-behavior-prediction

3. Create a virtual environment

python -m venv .venv

4. Activate the virtual environment

Windows PowerShell:

.\.venv\Scripts\Activate.ps1

5. Install the required packages

pip install streamlit pandas numpy scikit-learn matplotlib seaborn plotly openpyxl joblib

6. Train the churn prediction models

python churn_model.py

Ensure the dataset is available at:

data/raw/online_retail_II.xlsx

7. Run the Streamlit application

streamlit run app.py

Limitations

- Predictions depend on historical transaction data.
- Churn predictions are estimates, not confirmed future customer behaviour.
- The churn definition uses a specific 90-day prediction period.
- Model performance may change when applied to different datasets or time periods.
- Customer retention recommendations require further business analysis.

Future Enhancements

- Advanced machine learning algorithms.
- Explainable AI using SHAP or LIME.
- Additional customer segmentation techniques.
- Interactive customer retention recommendations.
- API integration.
- Database integration.
- Automated model monitoring.

Conclusion

Vantara Customer Behavior Prediction Platform combines retail analytics, customer segmentation, and machine learning-based churn prediction in an interactive dashboard.

The project demonstrates how historical retail data can be transformed into useful business insights to support customer understanding and retention planning.

---

Developed as an AI and Data Science project.
