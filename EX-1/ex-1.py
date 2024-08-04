import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("D:/cit-22smcb0055/SEM-V/Machine Learning/EX-1/USA_Housing.csv")

# Print the first few rows of the dataset
df.head()

# Describe the data to get summary statistics
print("\nSummary statistics:")
print(df.describe())

# Check for Null values
print("\nNull values present?")
print(df.isnull().values.any())

# Check for Duplicates
print("\nDuplicates present?")
print(df.duplicated().values.any())

# Print the column names
print("\nColumn names:")
print(df.columns)


# --- Simple Linear Regression ---
x_simple = df[['Avg. Area Income']] 
y = df['Price']

# Split the data
X_train_simple, X_test_simple, y_train, y_test = train_test_split(x_simple, y, test_size=0.2, random_state=42)

# Create and train the model
lr_simple = LinearRegression()
lr_simple.fit(X_train_simple, y_train)

# Predict and evaluate
predictions_simple = lr_simple.predict(X_test_simple)
mae_simple = mean_absolute_error(y_test, predictions_simple)
mse_simple = mean_squared_error(y_test, predictions_simple)

print("\nSimple Linear Regression:")
print("MAE:", mae_simple)
print("MSE:", mse_simple)

# --- Multiple Linear Regression ---
# Use all independent variables
x_multiple = df.drop(['Price', 'Address'], axis=1)
y = df['Price']

# Split the data
X_train_multiple, X_test_multiple, y_train, y_test = train_test_split(x_multiple, y, test_size=0.2, random_state=42)

# Create and train the model
lr_multiple = LinearRegression()
lr_multiple.fit(X_train_multiple, y_train)

# Predict and evaluate
predictions_multiple = lr_multiple.predict(X_test_multiple)
mae_multiple = mean_absolute_error(y_test, predictions_multiple)
mse_multiple = mean_squared_error(y_test, predictions_multiple)

print("\nMultiple Linear Regression:")
print("MAE:", mae_multiple)
print("MSE:", mse_multiple)

# --- Polynomial Regression ---
x_poly = df[['Avg. Area Income']]
y = df['Price']

# Split the data
X_train_poly, X_test_poly, y_train, y_test = train_test_split(x_poly, y, test_size=0.2, random_state=42)

# Create polynomial features (degree 2)
poly_features = PolynomialFeatures(degree=2) 
X_train_poly = poly_features.fit_transform(X_train_poly)
X_test_poly = poly_features.transform(X_test_poly)

# Create and train the model
lr_poly = LinearRegression()
lr_poly.fit(X_train_poly, y_train)

# Predict and evaluate
predictions_poly = lr_poly.predict(X_test_poly)
mae_poly = mean_absolute_error(y_test, predictions_poly)
mse_poly = mean_squared_error(y_test, predictions_poly)

print("\nPolynomial Regression (degree 2):")
print("MAE:", mae_poly)
print("MSE:", mse_poly)

# Determine the coefficients of the independent variables
cdf = pd.DataFrame(lr_multiple.coef_, x_multiple.columns, columns=['Coeff'])
print("\nCoefficients:\n", cdf)

# Drop the 'Address' column for correlation calculation
df_numeric = df.drop(['Address'], axis=1)

# Calculate correlation matrix
correlation_matrix = df_numeric.corr()
print("\nCorrelation matrix:\n", correlation_matrix)

# Correlation with price
correlation_with_price = correlation_matrix["Price"].drop("Price")
print("\nCorrelation of other variables with Price:\n", correlation_with_price)

# Predicting on the unseen dataset (Test dataset)
predict_multiple = lr_multiple.predict(X_test_multiple)  
print("\n\nThe predicted Values are:", predict_multiple) 

# Evaluation
print("\nMAE is:", mean_absolute_error(y_test, predict_multiple))
print("MSE is:", mean_squared_error(y_test, predict_multiple))

# Evaluate and compare models using MAE, MSE, and R-squared
print("Model Evaluation:")
print("-" * 30)
models = [("Simple Linear Regression", lr_simple, X_test_simple, predictions_simple),
          ("Multiple Linear Regression", lr_multiple, X_test_multiple, predictions_multiple),
          ("Polynomial Regression (degree 2)", lr_poly, X_test_poly, predictions_poly)]

for name, model, X_test, predictions in models:
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = model.score(X_test, y_test)  

    print(f"{name}:")
    print(f"  MAE: {mae:.2f}")
    print(f"  MSE: {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R-squared: {r2:.2f}")
    print("-" * 30)

# --- Visualizations ---

# 1. Scatterplots for each independent variable against Price
plt.figure(figsize=(12, 8))
for i, col in enumerate(x_multiple.columns):
    plt.subplot(3, 2, i+1)
    plt.scatter(x_multiple[col], y, alpha=0.5)
    plt.xlabel(col)
    plt.ylabel('Price')
plt.tight_layout()
plt.show()

# 2. Heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# 3. Actual vs Predicted Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions_multiple, alpha=0.5)  # Assuming multiple regression is best
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs. Predicted Prices (Multiple Linear Regression)")
plt.show()

# 4. Residual plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predict_multiple - y_test, alpha=0.5)
plt.xlabel('Actual Price')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# 5. Distribution of the predicted prices
plt.figure(figsize=(8, 6))
sns.histplot(predict_multiple, kde=True)
plt.title('Distribution of Predicted Prices')
plt.xlabel('Predicted Price')
plt.ylabel('Frequency')
plt.show()

# 6. Distribution of the actual prices
plt.figure(figsize=(8, 6))
sns.histplot(y_test, kde=True)
plt.title('Distribution of Actual Prices')
plt.xlabel('Actual Price')
plt.ylabel('Frequency')
plt.show()

def get_user_input_and_predict():
    while True:
        print("\nWelcome to Housing Price Prediction")
        print("Press 1 to continue or 0 to exit")
        choice = input()
        if choice == '0':
            break
        
        print("\nPlease enter the following details to predict house price:")
        avg_area_income = float(input("Average Area Income: "))
        avg_area_house_age = float(input("Average Area House Age: "))
        avg_area_number_of_rooms = float(input("Average Area Number of Rooms: "))
        avg_area_number_of_bedrooms = float(input("Average Area Number of Bedrooms: "))
        area_population = float(input("Area Population: "))
        
        user_data = pd.DataFrame({
            'Avg. Area Income': [avg_area_income],
            'Avg. Area House Age': [avg_area_house_age],
            'Avg. Area Number of Rooms': [avg_area_number_of_rooms],
            'Avg. Area Number of Bedrooms': [avg_area_number_of_bedrooms],
            'Area Population': [area_population]
        })
        
        prediction = lr_multiple.predict(user_data)
        print("\nThe predicted house price is: ${:,.2f}".format(prediction[0]))

        # Plot user input and prediction (assuming multiple regression is best)
        plt.figure(figsize=(10, 6))
        plt.bar(user_data.columns, user_data.values[0])
        plt.xlabel("Features")
        plt.ylabel("Values")
        plt.title("User Input Data for Prediction")
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.bar(["Predicted Price"], [prediction[0]])
        plt.xlabel("Prediction")
        plt.ylabel("Price")
        plt.title("Predicted House Price (Multiple Linear Regression)")
        plt.show()

# Call the function to get user input and predict house price
get_user_input_and_predict()