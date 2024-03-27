import pandas as pd
from scipy.stats import kendalltau
from copulas.multivariate import GaussianMultivariate
from copulas.visualization import scatter_2d

# Step 1: Data Preprocessing
data = pd.read_csv("data/NSE_20_Share_Historical_Data.csv")
# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
# Remove commas and convert numeric columns to float
numeric_columns = ['Price', 'Open', 'High', 'Low']

# Preprocess 'Vol.' column
def preprocess_volume(volume):
    if isinstance(volume, str):
        if volume.endswith('M'):
            return float(volume[:-1]) * 1e6  # Multiply by 1 million
        elif volume.endswith('B'):
            return float(volume[:-1]) * 1e9  # Multiply by 1 billion
        else:
            return float(volume)
    else:
        return volume

data['Vol.'] = data['Vol.'].apply(preprocess_volume)

data['Change %'] = data['Change %'].str.replace("%", '').astype(float)

for col in numeric_columns:
    data[col] = data[col].str.replace(',', '').astype(float)



# Now the data is ready for further analysis

# Step 2: Statistical Analysis
# Calculate Kendall's tau correlation coefficient between equities and foreign exchange
kendall_corr, _ = kendalltau(data['Price'], data['Change %'])
print("Kendall's tau correlation coefficient:", kendall_corr)

# Step 3: Copula Model Selection
# Based on the characteristics of the data, select candidate copula models (e.g., Gaussian copula, t-copula, etc.)

# Step 4: Model Fitting and Evaluation
# Fit selected copula models to the data
copula = GaussianMultivariate()
copula.fit(data[['Price', 'Change %']])

# Step 5: Model Comparison and Selection
# Evaluate the fitted copula model and compare it with other candidate models
# Visualize the dependence structure

scatter_2d(data[['Price', 'Change %']])


# Additional steps as needed:
# - Consider other copula families
# - Tune parameters of selected copula models
# - Cross-validate the models if applicable
# - Use additional statistical tests or metrics for evaluation
