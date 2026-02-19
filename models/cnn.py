#Imports
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Reshape, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import shap
import matplotlib.pyplot as plt
import joblib

#Paths
graph_dir = "./graphs"
artifact_dir = "./artifacts"
os.makedirs(graph_dir, exist_ok=True)
os.makedirs(artifact_dir, exist_ok=True)

#Load Synthetic Prompt Dataset
df = pd.read_parquet("./data/raw/synthetic_prompt.parquet")
df = df[df["ce_score"].notna()].reset_index(drop=True)

target = "ce_score"

print("Dataset loaded:", df.shape)

#Feature matrix
X_full = df.drop(columns=[target])

#Ensure numeric-only features
non_numeric_cols = X_full.select_dtypes(exclude=["number"]).columns.tolist()

if len(non_numeric_cols) > 0:
    print("Dropping non-numeric columns:", non_numeric_cols)
    X_full = X_full.drop(columns=non_numeric_cols)

y = df[target]

#Recursive Feature Elimination
print("\nRunning RFE for feature selection")

base_estimator = ExtraTreesRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=4
)

rfe = RFE(
    estimator=base_estimator,
    n_features_to_select=6,
    step=1
)

rfe.fit(X_full, y)

selected_features = X_full.columns[rfe.support_]

print("Final CNN features:", selected_features.tolist())

np.save(f"{artifact_dir}/cnn_features.npy", selected_features.to_numpy())

#CNN Data Preparation
X = df[selected_features].values

X_train_pool, X_test, y_train_pool, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_pool, y_train_pool, test_size=0.20, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

joblib.dump(scaler, f"{artifact_dir}/cnn_scaler.pkl")

#CNN Model
n_features = X_train.shape[1]

model = Sequential([
    Reshape((n_features, 1), input_shape=(n_features,)),
    Conv1D(32, 3, activation="relu", padding="same"),
    MaxPooling1D(2),
    Conv1D(64, 3, activation="relu", padding="same"),
    MaxPooling1D(2),
    Flatten(),
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1)
])

model.compile(
    optimizer=Adam(0.001),
    loss="mse",
    metrics=["mae"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

#Evaluation
def evaluate(name, Xs, ys):
    preds = model.predict(Xs)
    print(f"\n{name}")
    print("MAE:", mean_absolute_error(ys, preds))
    print("R2 :", r2_score(ys, preds))
    return preds

y_train_pred = evaluate("Train", X_train, y_train)
y_val_pred   = evaluate("Validation", X_val, y_val)
y_test_pred  = evaluate("Test", X_test, y_test)

#Save model
model.save(os.path.join(artifact_dir, "cnn_ce_model.keras"))

#SHAP + Intention Modeling
def cnn_predict(x):
    return model.predict(x).reshape(-1)

background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
explainer = shap.KernelExplainer(cnn_predict, background)
shap_values = explainer.shap_values(X_test[:50], nsamples=200)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test[:50], feature_names=selected_features.tolist(), show=False)
plt.savefig(f"{graph_dir}/shap_summary.png", bbox_inches="tight")
plt.close()

embedding_layer = model.layers[-5].output
embedding_model = Model(inputs=model.layers[0].input, outputs=embedding_layer)

embeddings = embedding_model.predict(X_test[:50])
clusters = KMeans(n_clusters=4, random_state=42).fit_predict(embeddings)

pd.DataFrame({
    "cluster": clusters,
    "ce_score": y_test.iloc[:50].values
}).groupby("cluster").mean().to_csv(
    f"{graph_dir}/intention_cluster_means.csv"
)

cluster_mean = pd.read_csv(
    f"{graph_dir}/intention_cluster_means.csv",
    index_col=0
)

plt.figure(figsize=(8, 5))

colors = ["#4CAF50", "#FF9800", "#03A9F4", "#9C27B0"]

cluster_mean["ce_score"].plot(
    kind="bar",
    color=colors[:len(cluster_mean)],
    edgecolor="black"
)

plt.title("Intention Clusters vs Cognitive Efficiency", fontsize=14)
plt.xlabel("Intention Cluster", fontsize=12)
plt.ylabel("Mean Cognitive Efficiency", fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.6)

for i, v in enumerate(cluster_mean["ce_score"]):
    plt.text(i, v + 0.02 * cluster_mean["ce_score"].max(),
             f"{v:.2f}", ha="center", fontsize=11)

plt.tight_layout()
plt.savefig(f"{graph_dir}/intention_cluster_means.png", dpi=300)
plt.close()

print("\nCNN training complete. Artifacts + SHAP + Intention modeling saved.")