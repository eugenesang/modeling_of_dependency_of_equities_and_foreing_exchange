import pandas as pd
from scipy.stats import kendalltau
from copulas.multivariate import GaussianMultivariate
from copulas.visualization import scatter_2d
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

stock_data = pd.read_csv("data/NSE_20_Share_Historical_Data.csv")
currency_data = pd.read_csv("data/cbk_forex_raph.csv")

stock_data.head()
currency_data.head()

# Convert date columns to datetime
stock_data['Date'] = pd.to_datetime(stock_data['Date'], format='%m/%d/%Y')
currency_data['Date'] = pd.to_datetime(currency_data['Date'], format='%d/%m/%Y')


