import pandas as pd
from scipy.stats import kendalltau
from copulas.multivariate import GaussianMultivariate
from copulas.visualization import scatter_2d
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

stock_data = pd.read_csv("data/NSE_20_Share_Historical_Data.csv")

currency_data = pd.read_csv("data/cbk_forex_raph.csv")

stock_data['Date'] = pd.to_datetime(stock_data['Date'], format='%m/%d/%Y')

currency_data['Date'] = pd.to_datetime(currency_data['Date'], format='%d/%m/%Y')

numeric_columns_stock = ['Price', 'Open', 'High', 'Low']
for col in numeric_columns_stock:
    if stock_data[col].dtype == 'object':
        stock_data[col] = stock_data[col].str.replace(',', '').astype(float)
        
# Filter datasets to include only overlapping dates
common_dates = pd.merge(stock_data['Date'], currency_data['Date'], on='Date')['Date']
stock_data_filtered = stock_data[stock_data['Date'].isin(common_dates)]
currency_data_filtered = currency_data[currency_data['Date'].isin(common_dates)]

# Print lengths of filtered datasets
print("Length of stock_data_filtered:", len(stock_data_filtered))
print("Length of currency_data_filtered:", len(currency_data_filtered))

# Print unique dates in both datasets
print("Unique dates in stock_data_filtered:", stock_data_filtered['Date'].unique())
print("Unique dates in currency_data_filtered:", currency_data_filtered['Date'].unique())

# Print duplicated dates in both datasets
print("Duplicated dates in stock_data_filtered:")
print(stock_data_filtered[stock_data_filtered.duplicated(subset='Date', keep=False)])
print("Duplicated dates in currency_data_filtered:")
print(currency_data_filtered[currency_data_filtered.duplicated(subset='Date', keep=False)])

currency_data_aggregated = currency_data_filtered.groupby('Date')['Mean'].mean().reset_index()

# Compute Kendall's tau correlation coefficient
kendall_corr, _ = kendalltau(stock_data_filtered['Price'], currency_data_aggregated['Mean'])
print("Kendall's tau correlation coefficient:", kendall_corr)


# Step 3: Copula Model Selection
# Based on the characteristics of the data, select candidate copula models (e.g., Gaussian copula, t-copula, etc.)
# For simplicity, let's choose Gaussian copula for demonstration
copula = GaussianMultivariate()


# Step 4: Model Fitting and Evaluation
# Fit selected copula model to the data
# For demonstration purposes, we'll fit the copula using only 'Price' and 'Mean' columns
# Clean the data
# Clean the data
# Data Cleaning for Visualization (modified)
stock_data_filtered['Change %'] = stock_data_filtered['Change %'].str.replace('%', '', regex=True).astype(float)

# Only select rows with valid numeric values (optional)
stock_data_filtered = stock_data_filtered[~stock_data_filtered['Change %'].isna()]  # Filter out NaN values (optional)

print(stock_data_filtered.head())

df = pd.DataFrame(stock_data_filtered)

# Visualize the dependence structure
plt.scatter(df['Price'], df['Change %'])
plt.xlabel('Price')
plt.ylabel('Change %')
plt.title("Comparison of Price & Change %")
plt.show()