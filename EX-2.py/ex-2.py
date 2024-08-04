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

# --- Simple Linear Regression ---
x_simple = df[['Avg. Area Income']]
y = df['Price']
X_train_simple, X_test_simple, y_train, y_test = train_test_split(x_simple, y, test_size=0.2, random_state=42)
lr_simple = LinearRegression()
lr_simple.fit(X_train_simple, y_train)

# Predict and evaluate
predictions_simple = lr_simple.predict(X_test_simple)

# --- Multiple Linear Regression ---
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
X_train_poly, X_test_poly, y_train, y_test = train_test_split(x_multiple, y, test_size=0.2, random_state=42)
poly_features = PolynomialFeatures(degree=3)
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

# Visualization
y_test_sample = y_test[:200]


#find kfold
acc = []
max_acc = []
mean_acc = []   
std_acc=[]

for i in range(2,11):
    kf = KFold(n_splits=i , shuffle=True, random_state=42)
    fold_accuracy = []
    for train_index, test_index in kf.split(x_multiple , y):
        X_train, X_test = x_multiple.iloc[train_index], x_multiple.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        lr_multiple.fit(X_train, y_train)
        fold_accuracy.append(lr_multiple.score(X_test, y_test))
    acc.append(fold_accuracy)
    max_acc.append(max(fold_accuracy))
    mean_acc.append(np.mean(fold_accuracy))
    std_acc.append(np.std(fold_accuracy))
    print(f"K={i}, Accuracy: {fold_accuracy}")
    print(f"Max Accuracy: {max(fold_accuracy)}")
    print(f"Mean Accuracy: {np.mean(fold_accuracy)}")
    print(f"Standard Deviation: {np.std(fold_accuracy)}")
    print("\n")

max_acc_index = np.argmax(max_acc)
best_k = max_acc_index + 2
print(f"Max Accuracy in K: {max(max_acc)}, K={best_k}")
print(f"Mean Accuracy in K: {mean_acc[max_acc_index]}")
print("\n")
print(max_acc)

# Visualization of kfold
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), max_acc, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.axvline(x=best_k, color='red', linestyle='--')
plt.title('Accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Max Accuracy')
plt.annotate(f'Best K = {best_k}', xy=(best_k, mean_acc[max_acc_index]), xytext=(best_k + 1, mean_acc[max_acc_index]),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()


