# Subset the DataFrame
sub_df = so_survey_df[['Age','Gender']]

# Print the number of non-missing values
print(sub_df.info())

---------
# Print the top 10 entries of the DataFrame
print(sub_df.head(10))

# Print the locations of the missing values
print(sub_df.head(10).isnull())

# Print the locations of the non-missing values
print(sub_df.head(10).notnull())

--------------
# Print the number of rows and columns
print(so_survey_df.shape)

-----
# Create a new DataFrame dropping all incomplete rows
no_missing_values_rows = so_survey_df.dropna()

# Print the shape of the new DataFrame
print(no_missing_values_rows.shape)
----------
# Create a new DataFrame dropping all columns with incomplete rows
no_missing_values_cols = so_survey_df.dropna(how="any", axis=1)

# Print the shape of the new DataFrame
print(no_missing_values_cols.shape)

-----------
# Drop all rows where Gender is missing
no_gender = so_survey_df.dropna(subset=['Gender'])

# Print the shape of the new DataFrame
print(no_gender.shape)

------
# Replace missing values
so_survey_df['Gender'].fillna( value='Not Given',inplace=True)

# Print the count of each value
print(so_survey_df['Gender'].value_counts())

-----
# Print the first five rows of StackOverflowJobsRecommend column
print(so_survey_df['StackOverflowJobsRecommend'].head(5))
-------
# Fill missing values with the mean
so_survey_df['StackOverflowJobsRecommend'].fillna(so_survey_df['StackOverflowJobsRecommend'].mean(), inplace=True)

# Print the first five rows of StackOverflowJobsRecommend column
print(so_survey_df['StackOverflowJobsRecommend'].head())

----------
# Fill missing values with the mean
so_survey_df['StackOverflowJobsRecommend'].fillna(so_survey_df['StackOverflowJobsRecommend'].mean(), inplace=True)

# Round the StackOverflowJobsRecommend values
so_survey_df['StackOverflowJobsRecommend'] = round(so_survey_df['StackOverflowJobsRecommend'])

# Print the top 5 rows
print(so_survey_df['StackOverflowJobsRecommend'].head())

--------------
# Apply the measures of central tendency (mean/median etc.) calculated on the train set to both the train and test sets.
------------
# Remove the commas in the column
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace(',', '')
--------
# Remove the dollar signs in the column
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace("$","")
-------
# Attempt to convert the column to numeric values
numeric_vals = pd.to_numeric(so_survey_df['RawSalary'], errors='coerce')

# Find the indexes of missing values
idx = numeric_vals.isnull()

# Print the relevant rows
print(so_survey_df['RawSalary'][idx])
-----------

# Replace the offending characters
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace('£','')

# Convert the column to float
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].astype('float')

# Print the column
print(so_survey_df['RawSalary'])

# Remember that even after removing all the relevant characters, 
# you still need to change the type of the column to numeric if you want to plot these continuous values.

-------------

# Use method chaining
so_survey_df['RawSalary'] = so_survey_df['RawSalary']\
                              .str.replace(",","")\
                              .str.replace("$","")\
                              .str.replace("£","")\
                              .astype("float")\
                              
 
# Print the RawSalary column
print(so_survey_df['RawSalary'])

# Custom functions can be also used when method chaining using the .apply() method
----------
