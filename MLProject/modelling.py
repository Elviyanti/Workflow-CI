import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset hasil preprocessing
df = pd.read_csv("heartdataset_preprocessing.csv")

# Pisahkan fitur dan target
X = df.drop("target", axis=1)
y = df["target"]

# Split data (stratified supaya hasil lebih stabil)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Hyperparameter grid
n_estimators_options = [50, 100, 200]
max_depth_options = [5, 10, None]
min_samples_split_options = [2, 5]
min_samples_leaf_options = [1, 2]

# Loop semua kombinasi hyperparameter
for n in n_estimators_options:
    for depth in max_depth_options:
        for split in min_samples_split_options:
            for leaf in min_samples_leaf_options:

                with mlflow.start_run():

                    # Build model
                    model = RandomForestClassifier(
                        n_estimators=n,
                        max_depth=depth,
                        min_samples_split=split,
                        min_samples_leaf=leaf,
                        random_state=42,
                    )

                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    acc = accuracy_score(y_test, preds)

                    print(
                        f"n_estimators={n}, max_depth={depth}, "
                        f"min_samples_split={split}, min_samples_leaf={leaf}, "
                        f"ACC={acc}"
                    )

                    # Log parameters
                    mlflow.log_param("n_estimators", n)
                    mlflow.log_param("max_depth", depth)
                    mlflow.log_param("min_samples_split", split)
                    mlflow.log_param("min_samples_leaf", leaf)

                    # Log metrics
                    mlflow.log_metric("accuracy", acc)

                    # Log model
                    mlflow.sklearn.log_model(model, "model")
