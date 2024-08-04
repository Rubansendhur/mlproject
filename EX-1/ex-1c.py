import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error , r2_score
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("D:/cit-22smcb0055/SEM-V/Machine Learning/EX-1/USA_Housing.csv")

# --- Simple Linear Regression ---
x_simple = df[['Avg. Area Income']] 
y = df['Price']
X_train_simple, X_test_simple, y_train, y_test = train_test_split(x_simple, y, test_size=0.2, random_state=42)
lr_simple = LinearRegression()
lr_simple.fit(X_train_simple, y_train)

print(df.columns)

# Predict and evaluate
predictions_simple = lr_simple.predict(X_test_simple)

# --- Multiple Linear Regression ---
# Use all independent variables
x_multiple = df.drop(['Price', 'Address'], axis=1)
y = df['Price']
X_train_multiple, X_test_multiple, y_train, y_test = train_test_split(x_multiple, y, test_size=0.2, random_state=42)
lr_multiple = LinearRegression()
lr_multiple.fit(X_train_multiple, y_train)

# Predict and evaluate
predictions_multiple = lr_multiple.predict(X_test_multiple)

# --- Polynomial Regression ---
x_poly = df[['Avg. Area Income']]
y = df['Price']
X_train_poly, X_test_poly, y_train, y_test = train_test_split(x_poly, y, test_size=0.2, random_state=42)
poly_features = PolynomialFeatures(degree=2) 
X_train_poly = poly_features.fit_transform(X_train_poly)
X_test_poly = poly_features.transform(X_test_poly)
lr_poly = LinearRegression()
lr_poly.fit(X_train_poly, y_train)

# Predict and evaluate
predictions_poly = lr_poly.predict(X_test_poly)
 

print("Model Evaluation:")
print("-" * 30)
models = [("Simple Linear Regression", lr_simple, X_test_simple, predictions_simple),
          ("Multiple Linear Regression", lr_multiple, X_test_multiple, predictions_multiple),
          ("Polynomial Regression (degree 2)", lr_poly, X_test_poly, predictions_poly)]

for name, model, X_test, predictions in models:
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)  

    print(f"{name}:")
    print(f"  MAE: {mae:.2f}")
    print(f"  MSE: {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R-squared: {r2:.2f}")     
    print("-" * 30)


#----Visualisation-----

y_test = y_test[:200]

# Line chart for Simple Linear Regression
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test, color='blue', label='Actual Prices')
plt.plot(range(len(y_test)), predictions_simple[:200], color='red', linestyle='dashed', label='Predicted Prices')
plt.xlabel('Index')
plt.ylabel('Price')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()

# Line chart for Multiple Linear Regression
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test, color='blue', label='Actual Prices')
plt.plot(range(len(y_test)), predictions_multiple[:200], color='red', linestyle='dashed', label='Predicted Prices')
plt.xlabel('Index')
plt.ylabel('Price')
plt.title('Multiple Linear Regression')
plt.legend()
plt.show()

# Line chart for Polynomial Regression
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test, color='blue', label='Actual Prices')
plt.plot(range(len(y_test)), predictions_poly[:200], color='red', linestyle='dashed', label='Predicted Prices')
plt.xlabel('Index')
plt.ylabel('Price')
plt.title('Polynomial Regression (degree 2)')
plt.legend()
plt.show()

