
"""Linear Regression_modelling
"""

#  Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#  Load the dataset
df = pd.read_csv('/content/Housing.csv')

#  Preprocess the dataset
# Convert categorical variables to numerical using one-hot encoding
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                    'airconditioning', 'prefarea', 'furnishingstatus']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

#  Define features and target variable
# For Simple Linear Regression, we'll use 'area' as the sole feature
X_simple = df_encoded[['area']]
y = df_encoded['price']

# For Multiple Linear Regression, we'll use all features except 'price'
X_multiple = df_encoded.drop('price', axis=1)

#  Split data into training and testing sets
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
    X_simple, y, test_size=0.2, random_state=42)

X_train_multiple, X_test_multiple, y_train_multiple, y_test_multiple = train_test_split(
    X_multiple, y, test_size=0.2, random_state=42)

#  Fit Linear Regression models
# Simple Linear Regression
model_simple = LinearRegression()
model_simple.fit(X_train_simple, y_train_simple)

# Multiple Linear Regression
model_multiple = LinearRegression()
model_multiple.fit(X_train_multiple, y_train_multiple)

#  Make predictions
y_pred_simple = model_simple.predict(X_test_simple)
y_pred_multiple = model_multiple.predict(X_test_multiple)

# Evaluate models
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name} Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R² Score: {r2:.4f}")

evaluate_model(y_test_simple, y_pred_simple, "Simple Linear Regression")
evaluate_model(y_test_multiple, y_pred_multiple, "Multiple Linear Regression")

# Plot regression line for Simple Linear Regression
plt.figure(figsize=(8, 6))
plt.scatter(X_test_simple, y_test_simple, color='blue', label='Actual')
plt.plot(X_test_simple, y_pred_simple, color='red', linewidth=2, label='Predicted')
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Simple Linear Regression: Area vs Price")
plt.legend()
plt.grid(True)
plt.show()

# Interpret coefficients
print("\nSimple Linear Regression Coefficients:")
print(f"Intercept: {model_simple.intercept_:.2f}")
print(f"Coefficient for 'area': {model_simple.coef_[0]:.2f}")

print("\nMultiple Linear Regression Coefficients:")
coefficients = pd.Series(model_multiple.coef_, index=X_multiple.columns)
print(coefficients)

# Plot coefficients for Multiple Linear Regression
plt.figure(figsize=(10, 8))
coefficients.sort_values().plot(kind='barh')
plt.title("Multiple Linear Regression Coefficients")
plt.xlabel("Coefficient Value")
plt.ylabel("Features")
plt.grid(True)
plt.tight_layout()
plt.show()

# Predict the price of a new house with custom input
print("\n Predict price for a new house with the following details:")

# Create a dictionary for new house input
new_house = {
    'area': 2780,
    'bedrooms': 2,
    'bathrooms': 1,
    'stories': 1,
    'parking': 1,
    'mainroad_yes': 1,
    'guestroom_yes': 1,
    'basement_yes': 1,
    'hotwaterheating_yes': 1,
    'airconditioning_yes': 1,
    'prefarea_yes': 0,
    'furnishingstatus_semi-furnished': 1,
    'furnishingstatus_unfurnished': 0
}

# Create a DataFrame with the same columns as X_multiple
new_input = pd.DataFrame([new_house])

# Ensure all columns in training set exist in input
missing_cols = set(X_multiple.columns) - set(new_input.columns)
for col in missing_cols:
    new_input[col] = 0  # fill missing one-hot cols with 0

# Reorder columns to match training data
new_input = new_input[X_multiple.columns]

# Predict using trained multiple regression model
predicted_price = model_multiple.predict(new_input)[0]
print(f" Predicted price for the new house: ₹{predicted_price:,.2f}")
