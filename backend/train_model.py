import pandas as pd
import numpy as np
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Load your dataset (replace with actual source)
df = pd.read_csv("training_data.csv")

# Convert 'last_attempt' datetime to numeric timestamp
df["last_attempt_ts"] = pd.to_datetime(df["last_attempt"]).astype(int) / 10**9


# Drop rows with missing values
required_columns = ["latest_score", "avg_score", "attempts_count", "last_attempt_ts", "target_days"]
df = df.dropna(subset=required_columns)

# Define features and target
X = df[["latest_score", "avg_score", "attempts_count", "last_attempt_ts"]]
y = df["target_days"]

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline: scale numeric features
numeric_features = X.columns.tolist()
numeric_transformer = Pipeline([
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features)
])

# Model pipeline with XGBoost
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(objective="reg:squarederror", random_state=42))
])

# Hyperparameter tuning with GridSearchCV
param_grid = {
    "regressor__n_estimators": [100, 200],
    "regressor__max_depth": [3, 5, 7],
    "regressor__learning_rate": [0.05, 0.1, 0.2],
    "regressor__subsample": [0.8, 1.0]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="neg_mean_absolute_error", verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Evaluate model
y_pred = grid_search.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("--- Model Evaluation ---")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2: {r2:.4f}")
print("Best Params:", grid_search.best_params_)

# Log results
with open("training_log.txt", "a") as log:
    log.write(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, Params: {grid_search.best_params_}\\n")

# Cross-validation score
cv_scores = cross_val_score(grid_search.best_estimator_, X, y, cv=5, scoring="neg_mean_absolute_error")
print(f"Cross-validated MAE: {-cv_scores.mean():.4f}")

# Plot feature importance
booster = grid_search.best_estimator_.named_steps["regressor"]
plot_importance(booster)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# Save final model
joblib.dump(grid_search.best_estimator_, "model1.pkl")
print("\\n✅ Model saved as model1.pkl")
