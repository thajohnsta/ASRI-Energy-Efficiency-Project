import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('recs2009_public.csv')

# Display the columns to identify relevant variables
print(data.columns)

# Columns for analysis: 'KWH', 'POVERTY100', 'POVERTY150', 'YEARMADE', 'Climate_Region_Pub'
dependent_variable = 'KWH'  # Example: Total energy usage
independent_variables = ['POVERTY100', 'POVERTY150', 'YEARMADE', 'Climate_Region_Pub']

# Selecting data with non-null values for the selected columns
selected_data = data[[dependent_variable] + independent_variables].dropna()

# Multiple Regression Analysis (using statsmodels)
X_statsmodels = sm.add_constant(selected_data[independent_variables])
model_statsmodels = sm.OLS(selected_data[dependent_variable], X_statsmodels).fit()
print(model_statsmodels.summary())

# Visualize the relationships using pairplot
sns.pairplot(selected_data, x_vars=independent_variables, y_vars=dependent_variable, height=5, aspect=0.8, kind='reg')
plt.suptitle('Scatter plots with regression lines')

# Add a correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(selected_data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

# Visualize predicted vs actual values
predicted_values = model_statsmodels.predict(X_statsmodels)

plt.figure(figsize=(8, 6))
plt.scatter(selected_data[dependent_variable], predicted_values, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.grid(True)
plt.show()

# Basic Decision Tree-like Structure
# You can create rules based on the coefficients and relationships observed from the regression model
# Not a true decision tree model but a structured approach based on the data

# Decision rules based on coefficients from the regression model
for i, var in enumerate(independent_variables):
    coef = model_statsmodels.params[var]
    print(f"{var}: Coefficient = {coef}")

# Simple decision rules
def predict_energy_usage(poverty100, poverty150, yearmade, climate_region):
    if poverty100 > 20 and poverty150 > 30:
        return "High Energy Usage"
    elif yearmade < 1970:
        return "Low Energy Usage"
    else:
        return "Medium Energy Usage"

# Predict function
predicted_usage = predict_energy_usage(25, 35, 1990, 'Cold')
print(f"Predicted Energy Usage: {predicted_usage}")
