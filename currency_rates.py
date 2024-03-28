import matplotlib.pyplot as plt
import pandas as pd

# Sample data
data = pd.read_csv("data/cbk_forex_raph.csv")

# Convert 'Date' column to datetime with the correct format
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

# Plotting
plt.figure(figsize=(10, 6))

# Group by currency and plot their rates over time
for currency, data_group in data.groupby('Currency'):
    plt.plot(data_group['Date'], data_group['Mean'], label=currency)

plt.xlabel('Date')
plt.ylabel('Mean Rate')
plt.title('Currency Rates Over Time')
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
