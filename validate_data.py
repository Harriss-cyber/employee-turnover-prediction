import pandas as pd

DATA_PATH = "IBM-HR-Analytics-Employee-Attrition-and-Performance-Revised.csv"

def validate_dataset(file_path):
    df = pd.read_csv(file_path)

    results = {
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "invalid_age_count": int(((df["Age"] < 18) | (df["Age"] > 65)).sum()),
        "invalid_income_count": int((df["MonthlyIncome"] <= 0).sum()),
        "invalid_attrition_count": int((~df["Attrition"].isin(["Yes", "No"])).sum())
    }

    return results

if __name__ == "__main__":
    results = validate_dataset(DATA_PATH)

    print("Automated Data Validation Results")
    for key, value in results.items():
        print(f"{key}: {value}")
