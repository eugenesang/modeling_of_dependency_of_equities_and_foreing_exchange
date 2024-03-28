import pandas as pd
from scipy.stats import kendalltau
from copulas.multivariate import GaussianMultivariate
from copulas.visualization import scatter_2d
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

# Read data
stock_data = pd.read_csv("data/NSE_20_Share_Historical_Data.csv")
currency_data = pd.read_csv("data/cbk_forex_raph.csv")

# Convert date columns to datetime
stock_data['Date'] = pd.to_datetime(stock_data['Date'], format='%m/%d/%Y')
currency_data['Date'] = pd.to_datetime(currency_data['Date'], format='%d/%m/%Y')

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

# Data Cleaning
# Replace non-numeric values with NaN
stock_data_filtered.replace({'Price': {'368.60M': pd.NA}}, inplace=True)

# Convert 'Price' column to numeric, converting non-numeric values to NaN
stock_data_filtered['Price'] = pd.to_numeric(stock_data_filtered['Price'], errors='coerce')

# Drop rows with NaN values
stock_data_filtered.dropna(subset=['Price'], inplace=True)

# Impute missing values (excluding datetime column)
imputer = KNNImputer()
stock_data_filtered_imputed = pd.DataFrame(imputer.fit_transform(stock_data_filtered.drop(columns=['Date'])))

# Print first few rows of cleaned and imputed data
print(stock_data_filtered_imputed.head())

# Visualize the dependence structure
plt.scatter(stock_data_filtered_imputed[0], stock_data_filtered_imputed[1])
plt.xlabel('Price')
plt.ylabel('Change %')
plt.title("Comparison of Price & Change %")
plt.show()

# Copula Model Fitting
copula = GaussianMultivariate()
copula.fit(stock_data_filtered_imputed[[0, 1]])

# Dependence Measure (alternative)
correlation = stock_data_filtered_imputed[0].corr(stock_data_filtered_imputed[1])
kendall_tau, _ = kendalltau(stock_data_filtered_imputed[0], stock_data_filtered_imputed[1])

print("Correlation between Price and Change %:", correlation)
print("Kendall's Tau Dependence:", kendall_tau)
