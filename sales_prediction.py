import pandas as pd

# Load dataset
df = pd.read_csv("Advertising.csv")

# Show basic information
print("Dataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:", df.duplicated().sum())
# Remove unnecessary index column
df = df.drop("Unnamed: 0", axis=1)

print("\nAfter Cleaning:")
print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
# Select features and target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
from sklearn.linear_model import LinearRegression

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("\nModel Training Completed!")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)
import matplotlib.pyplot as plt

# Plot actual vs predicted sales
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

# Perfect prediction reference line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.tight_layout()

# Save the graph
plt.savefig("actual_vs_predicted.png")

plt.show()

print("\nGraph saved as actual_vs_predicted.png")
# Analyze correlation between advertising and sales
correlation = df[["TV", "Radio", "Newspaper", "Sales"]].corr()

print("\nCorrelation with Sales:")
print(correlation["Sales"].sort_values(ascending=False))
# Plot correlation with sales
sales_correlation = correlation["Sales"].drop("Sales").sort_values(ascending=False)

plt.figure(figsize=(8, 5))

sales_correlation.plot(kind="bar")

plt.xlabel("Advertising Platform")
plt.ylabel("Correlation with Sales")
plt.title("Advertising Impact on Sales")

plt.tight_layout()

# Save the graph
plt.savefig("advertising_impact.png")

plt.show()

print("\nGraph saved as advertising_impact.png")