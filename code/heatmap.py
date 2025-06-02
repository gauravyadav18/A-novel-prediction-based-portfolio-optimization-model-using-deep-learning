import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load sted close or closing price for each asset
b = pd.read_csv('AGG.csv', parse_dates=['Date'], index_col='Date')[['Close']].rename(columns={'Close': 'B'})
c = pd.read_csv('DBC.csv', parse_dates=['Date'], index_col='Date')[['Close']].rename(columns={'Close': 'C'})
s = pd.read_csv('VTI.csv', parse_dates=['Date'], index_col='Date')[['Close']].rename(columns={'Close': 'S'})
v = pd.read_csv('VIX.csv', parse_dates=['Date'], index_col='Date')[['Close']].rename(columns={'Close': 'V'})

# Combine into one DataFrame
df = pd.concat([b, c, s, v], axis=1).dropna()

# Set rolling window size
window = 60  # 60 trading days ~ 3 months

# Define index pairs
pairs = [('B', 'C'), ('B', 'S'), ('B', 'V'), 
         ('C', 'S'), ('C', 'V'), ('S', 'V')]

# Compute rolling correlations
rolling_corrs = pd.DataFrame(index=df.index)

for i, j in pairs:
    rolling_corrs[f'{i}-{j}'] = df[i].rolling(window).corr(df[j])

# Drop initial NaNs (from rolling)
rolling_corrs = rolling_corrs.dropna()

# Ensure index is datetime
rolling_corrs.index = pd.to_datetime(rolling_corrs.index)

# Plot heatmap with year-wise x-axis labels:
plt.figure(figsize=(14, 6))
sns.heatmap(rolling_corrs.T, cmap='coolwarm', center=0, 
            cbar_kws={'label': 'Rolling Correlation'})

ax = plt.gca()

# Find year boundaries in index
years = rolling_corrs.index.year
unique_years = sorted(set(years))

# Get the position (integer index) of the first occurrence of each year
year_pos = [rolling_corrs.index.get_loc(rolling_corrs.index[years == year][0]) for year in unique_years]

# Set x-ticks at year starting positions
ax.set_xticks(year_pos)
ax.set_xticklabels(unique_years, rotation=45, ha='right')

# Labels and title
plt.title(f"Rolling Correlation Heatmap (Window = {window} days)")
plt.xlabel("Year")
plt.ylabel("Index Pair")
plt.tight_layout()
plt.show()
