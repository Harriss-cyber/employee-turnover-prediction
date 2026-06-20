import pandas as pd

DATA_PATH = "IBM-HR-Analytics-Employee-Attrition-and-Performance-Revised.csv"

def test_no_missing_values():
    df = pd.read_csv(DATA_PATH)
    assert df.isnull().sum().sum() == 0

def test_no_duplicate_rows():
    df = pd.read_csv(DATA_PATH)
    assert df.duplicated().sum() == 0

def test_valid_attrition_values():
    df = pd.read_csv(DATA_PATH)
    assert df["Attrition"].isin(["Yes", "No"]).all()

def test_valid_age_range():
    df = pd.read_csv(DATA_PATH)
    assert ((df["Age"] >= 18) & (df["Age"] <= 65)).all()
