#Imports
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from lightgbm import LGBMRegressor

#Path
artifact_dir = "./artifacts"
os.makedirs(artifact_dir, exist_ok=True)

#Load synthetic dataset
print("\nLoading synthetic dataset")
df = pd.read_parquet("./data/raw/synthetic_prompt.parquet")

target = "ce_score"
print("Dataset shape:", df.shape)

#Droping prompt_type
if "prompt_type" in df.columns:
    df = df.drop(columns=["prompt_type"])

#Feature matrix
X = df.drop(columns=[target])
y = df[target]

print("\nInitial features:")
print(X.columns.tolist())

#Train / Val / Test split
X_train_pool, X_test, y_train_pool, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_pool, y_train_pool, test_size=0.20, random_state=42
)

#Mutual Information
print("\nRunning Mutual Information feature selection")

mi_scores = mutual_info_regression(X_train, y_train, random_state=42)

mi_series = pd.Series(mi_scores, index=X_train.columns)
mi_series = mi_series.sort_values(ascending=False)

print("\nMI Scores:")
print(mi_series)

#Select top 7 features
selected_features = mi_series.head(7).index.tolist()

print("\nSelected Features (Top 8 MI):")
print(selected_features)

#Save feature list
np.save(f"{artifact_dir}/cepm_features.npy", np.array(selected_features))

#Apply selection
X_train = X_train[selected_features]
X_val   = X_val[selected_features]
X_test  = X_test[selected_features]

#Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

#Train CEPM
print("\nTraining CEPM model")

cepm_model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42,
    n_jobs=4
)

cepm_model.fit(X_train, y_train)

#Evaluation
print("\nTrain Performance")
print("MAE:", mean_absolute_error(y_train, cepm_model.predict(X_train)))
print("MSE:", mean_squared_error(y_train, cepm_model.predict(X_train)))
print("R2 :", r2_score(y_train, cepm_model.predict(X_train)))

print("\nValidation Performance")
print("MAE:", mean_absolute_error(y_val, cepm_model.predict(X_val)))
print("MSE:", mean_squared_error(y_val, cepm_model.predict(X_val)))
print("R2 :", r2_score(y_val, cepm_model.predict(X_val)))

print("\nTest Performance")
print("MAE:", mean_absolute_error(y_test, cepm_model.predict(X_test)))
print("MSE:", mean_squared_error(y_test, cepm_model.predict(X_test)))
print("R2 :", r2_score(y_test, cepm_model.predict(X_test)))

#Save artifacts
joblib.dump(cepm_model, f"{artifact_dir}/cepm_lightgbm.pkl")
joblib.dump(scaler, f"{artifact_dir}/cepm_scaler.pkl")

print("\nCEPM training complete. Artifacts saved.")