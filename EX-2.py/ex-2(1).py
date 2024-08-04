import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("D:/cit-22smcb0055/SEM-V/Machine Learning/EX-1/USA_Housing.csv")

# --- Multiple Linear Regression ---
x_multiple = df.drop(['Price', 'Address'], axis=1)
y = df['Price']
X_train_multiple, X_test_multiple, y_train, y_test = train_test_split(x_multiple, y, test_size=0.2, random_state=42)
lr_multiple = LinearRegression()
lr_multiple.fit(X_train_multiple, y_train)

# Predict and evaluate
predictions_multiple = lr_multiple.predict(X_test_multiple)

print("Model Evaluation:")
print("-" * 30)
models = [("Multiple Linear Regression", lr_multiple, X_test_multiple, predictions_multiple)]

for name, model, X_test, predictions in models:
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)

    print(f"{name}:")
    print(f"  MAE: {mae:.2f}")
    print(f"  MSE: {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R-squared: {r2:.3f}")
    print("-" * 30)

# k-fold cross validation

#find optimal value of k
best_k = 0
best_mse = float('inf') .

best_r2 = 0 

for i in range(2, 11):
    kf = KFold(n_splits=i, shuffle=True, random_state=42)  
    scores = -1 * (cross_val_score(lr_multiple, x_multiple, y, cv=kf, scoring='neg_mean_squared_error'))
    mean_mse = np.mean(scores)
    
    # Calculate R-squared using cross-validation
    r2_scores = cross_val_score(lr_multiple, x_multiple, y, cv=kf, scoring='r2')
    mean_r2 = np.mean(r2_scores)

    print(f"K = {i}, MSE: {mean_mse:.2f}, R-squared: {mean_r2:.3f}")

    # Update best_k based on MSE and R-squared
    if mean_mse < best_mse and mean_r2 > best_r2:
        best_mse = mean_mse
        best_k = i
        best_r2 = mean_r2

print(f"\nBest K: {best_k}")
print(f"Best Mean MSE: {best_mse:.2f}")
print(f"Best Mean R-squared: {best_r2:.2f}")