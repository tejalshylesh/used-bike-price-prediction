import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILE = "Used_Bikes.csv"
FEATURES = ["bike_name", "city", "kms_driven", "owner", "age", "power", "brand"]
TARGET = "price"
NUMERIC = ["kms_driven", "age", "power"]
CATEGORICAL = ["bike_name", "city", "owner", "brand"]

def main():
    df = pd.read_csv(DATA_FILE)
    df.columns = [str(c).strip() for c in df.columns]
    print(f"Dataset shape: {df.shape}")
    print("Columns:", list(df.columns))
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Expected columns missing: {missing}. Found: {list(df.columns)}")

    X = df[FEATURES].copy()
    y = pd.to_numeric(df[TARGET], errors="coerce")
    valid = y.notna() & (y > 0)
    X, y = X.loc[valid], y.loc[valid]
    for col in NUMERIC:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL)
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    algorithms = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regression": RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1)
    }
    results, trained = {}, {}
    for name, algorithm in algorithms.items():
        pipe = Pipeline([("preprocessing", preprocessor), ("regressor", algorithm)])
        pipe.fit(X_train, y_train)
        predictions = np.maximum(pipe.predict(X_test), 0)
        results[name] = {
            "MAE": float(mean_absolute_error(y_test, predictions)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "R2": float(r2_score(y_test, predictions))
        }
        trained[name] = pipe
        print(f"\n{name}")
        print(f"  MAE: ₹{results[name]['MAE']:,.2f}")
        print(f"  RMSE: ₹{results[name]['RMSE']:,.2f}")
        print(f"  R²: {results[name]['R2']:.4f}")
    best_name = min(results, key=lambda name: results[name]["MAE"])
    bundle = {
        "model": trained[best_name], "model_name": best_name,
        "features": FEATURES, "numeric": NUMERIC, "categorical": CATEGORICAL,
        "results": results, "dataset_rows": int(len(X)),
        "training_data": X.copy()
    }
    joblib.dump(bundle, "bike_price_model.pkl")
    print("\nComparison table:")
    print(pd.DataFrame(results).T)
    print(f"\nSelected by lowest test MAE: {best_name}")
    print("Saved model to bike_price_model.pkl")
    print("Now run: python -m streamlit run app.py")

if __name__ == "__main__":
    main()
