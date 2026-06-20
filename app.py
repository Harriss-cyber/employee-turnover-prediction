import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score


st.set_page_config(
    page_title="Employee Attrition Dashboard",
    layout="wide"
)

st.title("Employee Attrition Prediction Dashboard")
st.write(
    "This dashboard helps HR stakeholders analyse employee attrition patterns "
    "and identify employees with potential attrition risk."
)

DATA_PATH = "IBM-HR-Analytics-Employee-Attrition-and-Performance-Revised.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()


# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_features(data):
    data = data.copy()

    label_maps = {
        "BusinessTravel": {
            "Non-Travel": 0,
            "Travel_Rarely": 1,
            "Travel_Frequently": 2
        },
        "Education": {
            "Below College": 1,
            "College": 2,
            "Bachelor": 3,
            "Master": 4,
            "Doctor": 5
        },
        "EnvironmentSatisfaction": {
            "Low": 1,
            "Medium": 2,
            "High": 3,
            "Very High": 4
        },
        "JobInvolvement": {
            "Low": 1,
            "Medium": 2,
            "High": 3,
            "Very High": 4
        },
        "JobSatisfaction": {
            "Low": 1,
            "Medium": 2,
            "High": 3,
            "Very High": 4
        },
        "RelationshipSatisfaction": {
            "Low": 1,
            "Medium": 2,
            "High": 3,
            "Very High": 4
        },
        "WorkLifeBalance": {
            "Bad": 1,
            "Good": 2,
            "Better": 3,
            "Best": 4
        },
        "PerformanceRating": {
            "Excellent": 3,
            "Outstanding": 4
        },
        "JobLevel": {
            "Entry Level": 1,
            "Junior Level": 2,
            "Mid Level": 3,
            "Senior Level": 4,
            "Executive Level": 5
        },
        "OverTime": {
            "No": 0,
            "Yes": 1
        },
        "Gender": {
            "Female": 0,
            "Male": 1
        }
    }

    for col, mapping in label_maps.items():
        data[col] = data[col].map(mapping)

    data["IncomePerYearAtCompany"] = data["MonthlyIncome"] / (data["YearsAtCompany"] + 1)
    data["TenureRatio"] = data["YearsAtCompany"] / (data["TotalWorkingYears"] + 1)
    data["PromotionGapRatio"] = data["YearsSinceLastPromotion"] / (data["YearsAtCompany"] + 1)

    nominal_cols = ["Department", "EducationField", "JobRole", "MaritalStatus"]

    data_encoded = pd.get_dummies(
        data,
        columns=nominal_cols,
        drop_first=True,
        dtype=int
    )

    return data_encoded


# -----------------------------
# Train improved model
# -----------------------------
df_model = df.copy()
df_model["Attrition"] = df_model["Attrition"].map({"No": 0, "Yes": 1})

X_raw = df_model.drop("Attrition", axis=1)
y = df_model["Attrition"]

X = preprocess_features(X_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=2,
    random_state=42
)

model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]
threshold = 0.23
y_pred = (y_proba >= threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)


# -----------------------------
# Sidebar interactive filters
# -----------------------------
st.sidebar.header("Interactive Filters")

department_filter = st.sidebar.selectbox(
    "Select Department",
    ["All"] + sorted(df["Department"].unique().tolist())
)

overtime_filter = st.sidebar.selectbox(
    "Select Overtime Status",
    ["All"] + sorted(df["OverTime"].unique().tolist())
)

age_range = st.sidebar.slider(
    "Select Age Range",
    int(df["Age"].min()),
    int(df["Age"].max()),
    (int(df["Age"].min()), int(df["Age"].max()))
)

job_role_filter = st.sidebar.multiselect(
    "Select Job Role",
    sorted(df["JobRole"].unique().tolist())
)


filtered_df = df.copy()

if department_filter != "All":
    filtered_df = filtered_df[filtered_df["Department"] == department_filter]

if overtime_filter != "All":
    filtered_df = filtered_df[filtered_df["OverTime"] == overtime_filter]

filtered_df = filtered_df[
    (filtered_df["Age"] >= age_range[0]) &
    (filtered_df["Age"] <= age_range[1])
]

if job_role_filter:
    filtered_df = filtered_df[filtered_df["JobRole"].isin(job_role_filter)]

if filtered_df.empty:
    st.warning("No employee records match the selected filters.")
    st.stop()


# -----------------------------
# Dashboard KPIs
# -----------------------------
st.subheader("Dashboard Summary")

col1, col2, col3, col4 = st.columns(4)

total_employees = len(filtered_df)
attrition_count = (filtered_df["Attrition"] == "Yes").sum()
attrition_rate = attrition_count / total_employees * 100
avg_income = filtered_df["MonthlyIncome"].mean()

col1.metric("Total Employees", total_employees)
col2.metric("Attrition Cases", attrition_count)
col3.metric("Attrition Rate", f"{attrition_rate:.2f}%")
col4.metric("Average Monthly Income", f"{avg_income:,.0f}")

st.write("Filtered Dataset Preview")
st.dataframe(filtered_df.head())


# -----------------------------
# Visualization 1
# -----------------------------
st.subheader("Visualization 1: Attrition Distribution")

attrition_counts = filtered_df["Attrition"].value_counts()

fig1, ax1 = plt.subplots()
ax1.bar(attrition_counts.index, attrition_counts.values)
ax1.set_xlabel("Attrition")
ax1.set_ylabel("Number of Employees")
ax1.set_title("Employee Attrition Distribution")
st.pyplot(fig1)

st.write(
    "Observation: The distribution shows that most employees did not leave the organisation. "
    "This confirms that the attrition dataset is imbalanced."
)


# -----------------------------
# Visualization 2
# -----------------------------
st.subheader("Visualization 2: Attrition Rate by Department")

dept_rate = (
    filtered_df.assign(AttritionFlag=(filtered_df["Attrition"] == "Yes").astype(int))
    .groupby("Department")["AttritionFlag"]
    .mean()
    .sort_values(ascending=False) * 100
)

fig2, ax2 = plt.subplots()
ax2.bar(dept_rate.index, dept_rate.values)
ax2.set_xlabel("Department")
ax2.set_ylabel("Attrition Rate (%)")
ax2.set_title("Attrition Rate by Department")
plt.xticks(rotation=20)
st.pyplot(fig2)

st.write(
    "Observation: This chart helps HR compare attrition risk across departments "
    "and identify areas that may require retention strategies."
)


# -----------------------------
# Visualization 3
# -----------------------------
st.subheader("Visualization 3: Monthly Income by Attrition Status")

fig3, ax3 = plt.subplots()
filtered_df.boxplot(column="MonthlyIncome", by="Attrition", ax=ax3)
ax3.set_xlabel("Attrition")
ax3.set_ylabel("Monthly Income")
ax3.set_title("Monthly Income by Attrition Status")
plt.suptitle("")
st.pyplot(fig3)

st.write(
    "Observation: This visualization helps stakeholders compare the income distribution "
    "between employees who stayed and employees who left."
)


# -----------------------------
# Model performance output
# -----------------------------
st.subheader("Predictive Output from Improved Model")

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("Accuracy", f"{accuracy:.2%}")
m2.metric("Precision", f"{precision:.2%}")
m3.metric("Recall", f"{recall:.2%}")
m4.metric("F1-score", f"{f1:.2%}")
m5.metric("ROC-AUC", f"{roc_auc:.2%}")

st.write(
    "The improved Gradient Boosting model is used to support HR decision-making "
    "by estimating employee attrition risk."
)


# -----------------------------
# High-risk employee table
# -----------------------------
st.subheader("Analytical Output: Top Employees by Predicted Attrition Risk")

filtered_features = filtered_df.drop("Attrition", axis=1)
filtered_encoded = preprocess_features(filtered_features)
filtered_encoded = filtered_encoded.reindex(columns=X.columns, fill_value=0)

filtered_risk = model.predict_proba(filtered_encoded)[:, 1]

risk_table = filtered_df[
    ["Age", "Department", "JobRole", "OverTime", "MonthlyIncome", "YearsAtCompany", "Attrition"]
].copy()

risk_table["Predicted Attrition Risk"] = filtered_risk
risk_table["Predicted Attrition Risk"] = risk_table["Predicted Attrition Risk"].round(3)

risk_table = risk_table.sort_values(
    by="Predicted Attrition Risk",
    ascending=False
).head(10)

st.dataframe(risk_table)

st.write(
    "This table helps HR identify employee groups with higher predicted attrition risk "
    "for early retention actions."
)


# -----------------------------
# Individual employee prediction
# -----------------------------
st.subheader("Individual Employee Attrition Risk Prediction")

with st.form("prediction_form"):
    input_age = st.slider("Age", int(df["Age"].min()), int(df["Age"].max()), 30)
    input_income = st.slider(
        "Monthly Income",
        int(df["MonthlyIncome"].min()),
        int(df["MonthlyIncome"].max()),
        int(df["MonthlyIncome"].median())
    )
    input_distance = st.slider(
        "Distance From Home",
        int(df["DistanceFromHome"].min()),
        int(df["DistanceFromHome"].max()),
        int(df["DistanceFromHome"].median())
    )
    input_overtime = st.selectbox("OverTime", sorted(df["OverTime"].unique()))
    input_department = st.selectbox("Department", sorted(df["Department"].unique()))
    input_jobrole = st.selectbox("Job Role", sorted(df["JobRole"].unique()))
    input_business_travel = st.selectbox("Business Travel", sorted(df["BusinessTravel"].unique()))
    input_job_satisfaction = st.selectbox("Job Satisfaction", sorted(df["JobSatisfaction"].unique()))
    input_worklife = st.selectbox("Work-Life Balance", sorted(df["WorkLifeBalance"].unique()))
    input_years_company = st.slider(
        "Years at Company",
        int(df["YearsAtCompany"].min()),
        int(df["YearsAtCompany"].max()),
        int(df["YearsAtCompany"].median())
    )

    submitted = st.form_submit_button("Predict Attrition Risk")

if submitted:
    base_row = {}

    for col in df.drop("Attrition", axis=1).columns:
        if df[col].dtype == "object":
            base_row[col] = df[col].mode()[0]
        else:
            base_row[col] = df[col].median()

    base_row["Age"] = input_age
    base_row["MonthlyIncome"] = input_income
    base_row["DistanceFromHome"] = input_distance
    base_row["OverTime"] = input_overtime
    base_row["Department"] = input_department
    base_row["JobRole"] = input_jobrole
    base_row["BusinessTravel"] = input_business_travel
    base_row["JobSatisfaction"] = input_job_satisfaction
    base_row["WorkLifeBalance"] = input_worklife
    base_row["YearsAtCompany"] = input_years_company

    input_df = pd.DataFrame([base_row])
    input_encoded = preprocess_features(input_df)
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

    risk_probability = model.predict_proba(input_encoded)[:, 1][0]

    st.metric("Predicted Attrition Risk", f"{risk_probability:.2%}")

    if risk_probability >= threshold:
        st.warning("This employee is predicted as HIGH attrition risk.")
    else:
        st.success("This employee is predicted as LOW attrition risk.")
# -----------------------------
# Q5: Monitoring Metrics
# -----------------------------
st.header("Q5: Dashboard Monitoring")

st.subheader("Monitoring Metrics")

# Data quality monitoring
missing_values = int(df.isnull().sum().sum())
duplicate_rows = int(df.duplicated().sum())
invalid_age_count = int(((df["Age"] < 18) | (df["Age"] > 65)).sum())

# Business monitoring
high_risk_percentage = float((model.predict_proba(X)[:, 1] >= threshold).mean() * 100)
overall_attrition_rate = float((df["Attrition"] == "Yes").mean() * 100)

monitor_col1, monitor_col2, monitor_col3, monitor_col4 = st.columns(4)

monitor_col1.metric("Missing Values", missing_values)
monitor_col2.metric("Duplicate Rows", duplicate_rows)
monitor_col3.metric("Invalid Age Records", invalid_age_count)
monitor_col4.metric("High Attrition Risk %", f"{high_risk_percentage:.2f}%")

monitor_col5, monitor_col6, monitor_col7, monitor_col8 = st.columns(4)

monitor_col5.metric("Model Accuracy", f"{accuracy:.2%}")
monitor_col6.metric("Model Recall", f"{recall:.2%}")
monitor_col7.metric("Model F1-score", f"{f1:.2%}")
monitor_col8.metric("Actual Attrition Rate", f"{overall_attrition_rate:.2f}%")

st.write(
    "The monitoring metrics track data quality, model performance, and business risk. "
    "Missing values, duplicate rows, and invalid age records monitor data reliability. "
    "Accuracy, recall, and F1-score monitor model performance. "
    "The high attrition risk percentage helps HR understand the proportion of employees "
    "predicted as high risk."
)


# -----------------------------
# Q5: Data Drift Analysis using PSI
# -----------------------------
st.subheader("Data Drift Analysis")

def calculate_psi(expected, actual, buckets=10):
    expected = np.array(expected)
    actual = np.array(actual)

    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 2:
        return 0.0

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)

    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)

    return float(np.sum(psi_values))


drift_features = ["Age", "MonthlyIncome", "DistanceFromHome", "YearsAtCompany"]

drift_results = []

for feature in drift_features:
    psi_value = calculate_psi(X_train[feature], X_test[feature])

    if psi_value < 0.10:
        drift_status = "No significant drift"
    elif psi_value < 0.25:
        drift_status = "Moderate drift"
    else:
        drift_status = "Significant drift"

    drift_results.append({
        "Feature": feature,
        "PSI Value": round(psi_value, 4),
        "Drift Status": drift_status
    })

drift_df = pd.DataFrame(drift_results)

st.dataframe(drift_df)

st.write(
    "Population Stability Index was used to compare selected feature distributions "
    "between training and testing data. PSI values below 0.10 indicate no significant drift, "
    "values between 0.10 and 0.25 indicate moderate drift, and values above 0.25 indicate "
    "significant drift."
)

if (drift_df["PSI Value"] >= 0.25).any():
    st.warning(
        "Significant drift was detected in at least one feature. "
        "The model may need retraining in the next sprint."
    )
elif (drift_df["PSI Value"] >= 0.10).any():
    st.info(
        "Moderate drift was detected. The feature should be monitored in future iterations."
    )
else:
    st.success(
        "No significant drift was detected. The current model appears stable for the monitored features."
    )


# -----------------------------
# Q5: Sprint 4 Backlog Display
# -----------------------------
st.subheader("Sprint 4 Backlog")

sprint4_backlog = pd.DataFrame({
    "Priority": ["High", "High", "Medium", "Medium"],
    "Backlog Item": [
        "Improve imbalance handling using SMOTE or class-weighted modelling",
        "Add high-risk employee ranking table with downloadable output",
        "Add automated alert when model recall drops below target",
        "Add department-level drift monitoring"
    ],
    "Expected Benefit": [
        "Improves detection of employees likely to leave",
        "Helps HR plan early retention actions",
        "Allows the team to respond quickly to model performance degradation",
        "Helps identify changes in employee patterns across departments"
    ]
})

st.dataframe(sprint4_backlog)
