import pandas as pd

# Step 1: Load and clean the data
df = pd.read_csv("sleeptime_prediction_dataset.csv")
df = df[(df["SleepTime"] >= 1.0) & (df["SleepTime"] <= 12.0)]

print("Data loaded and filtered")
print(df.shape)

# Step 2: Split into features (X) and target (y)
X = df.drop(columns=["SleepTime"])
y = df["SleepTime"]

print("X shape:", X.shape)
print("y shape:", y.shape)

# Step 3: Split into training and testing sets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Step 4: Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model training complete!")

# Step 5: Evaluate the model
from sklearn.metrics import mean_absolute_error, r2_score

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f} hours")
print(f"R² Score: {r2:.2f}")


# Step 6: Save the model
import joblib

joblib.dump(model, "model.pkl")
print("Model saved to model.pkl")
