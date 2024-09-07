# Step 1: Import required libraries for visualization
import matplotlib.pyplot as plt

# Step 2: Define the dataset
X = [1, 2, 3, 4, 5]  # Independent variable (YearsExperience)
y = [45000, 50000, 60000, 80000, 110000]  # Dependent variable (Salary)

# Step 3: Calculate means of X and y
X_mean = sum(X) / len(X)
y_mean = sum(y) / len(y)

# Step 4: Calculate the coefficients (slope and intercept)
numerator = 0
denominator = 0
for i in range(len(X)):
    numerator += (X[i] - X_mean) * (y[i] - y_mean)
    denominator += (X[i] - X_mean) ** 2

slope = numerator / denominator
intercept = y_mean - slope * X_mean

# Step 5: Make predictions based on the calculated coefficients
y_pred = [slope * x + intercept for x in X]

# Step 6: Plotting the data points and regression line
plt.scatter(X, y, color='blue', label='Actual Data')  # Scatter plot of actual data
plt.plot(X, y_pred, color='red', label='Regression Line')  # Plot of regression line
plt.title('Linear Regression: Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()
