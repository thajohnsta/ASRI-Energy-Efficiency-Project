# energy_efficiency_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


# Load dataset
data = pd.read_csv("recs2009_public.csv")

# Select variables
target = "KWH"
features = [
    "POVERTY100",
    "POVERTY150",
    "YEARMADE",
    "Climate_Region_Pub"
]

df = data[[target] + features].dropna()

# One-hot encode categorical climate region
df = pd.get_dummies(df, columns=["Climate_Region_Pub"], drop_first=True)

X = df.drop(columns=[target])
y = df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("Model Performance")
print("-----------------")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.3f}")

# Feature importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance")
print(importance)

# Plot actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Energy Usage")
plt.ylabel("Predicted Energy Usage")
plt.title("Actual vs Predicted Energy Usage")
plt.grid(True)
plt.show()

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=importance, x="Importance", y="Feature")
plt.title("Feature Importance for Energy Usage Prediction")
plt.tight_layout()
plt.show()
