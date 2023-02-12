# The LotFrontage column actually does have missing values: 259, to be precise. Additionally, notice how columns such as MSZoning, 
# PavedDrive, and HouseStyle are categorical. These need to be encoded numerically before you can use XGBoost. 
# This is what you'll do in the coming exercises

---------------
# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Fill missing values with 0
df.LotFrontage = df.LotFrontage.fillna(0)

# Create a boolean mask for categorical columns
categorical_mask = (df.dtypes == object)

# Get list of categorical column names
categorical_columns = df.columns[categorical_mask].tolist()

# Print the head of the categorical columns
print(df[categorical_columns].head())

# Create LabelEncoder object: le
le = LabelEncoder()

# Apply LabelEncoder to categorical columns
df[categorical_columns] = df[categorical_columns].apply(lambda x: le.fit_transform(x))

# Print the head of the LabelEncoded categorical columns
print(df[categorical_columns].head())

# Notice how the entries in each categorical column are now encoded numerically. 
# A BldgTpe of 1Fam is encoded as 0, while a HouseStyle of 2Story is encoded as 5
------------
