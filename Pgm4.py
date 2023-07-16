import pandas as pd
data = pd.read_csv('Housing1.csv')
data

#Data pre processing

#data cleaning
# Check for missing values
print(data.isnull().sum())
# Remove rows with missing values
data = data.dropna()
data
#data integration
# Merge data from two datasets based on a common column
data = pd.read_csv('Housing1.csv')
data1 = pd.read_csv('Housing2.csv')
data_merged = pd.merge(data, data1, on='price')

# Concatenate two datasets vertically
data_concatenated = pd.concat([data, data1], axis=0)
data_concatenated

#Data transformation involves converting data into a suitable format or scale for analysis. Some common techniques include:
#Min-Max Scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data['offer'] = scaler.fit_transform(data['offer'].values.reshape(-1, 1))

# Standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data['offer'] = scaler.fit_transform(data['offer'].values.reshape(-1, 1))
data

#single column elimination
# Create DataFrame
df = pd.DataFrame(data)

# Print original dataset
print("Original dataset:")
print(df)
print()

# Delete columns with a single value
df = df.loc[:, df.nunique() > 1]

# Print preprocessed dataset
print("Preprocessed dataset:")
print(df)

#Eliminating duplicate rows in dataset
duplicate_rows = data[data.duplicated()]
duplicate_rows

data.drop_duplicates(inplace=True)
data.reset_index(drop=True, inplace=True)
data
