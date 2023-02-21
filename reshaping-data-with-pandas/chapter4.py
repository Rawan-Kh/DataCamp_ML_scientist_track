# Unstack the first level and calculate the mean of the columns
obesity_general = obesity.unstack(level=0).mean(axis=1)

# Print obesity_general
print(obesity_general)
---------
# Unstack the second level and calculate the mean of the columns
obesity_mean = obesity.unstack(level=1).mean(axis=1)

# Print obesity_mean
print(obesity_mean)
-------
# Unstack the third level and calculate the difference between columns
obesity_variation = obesity.unstack(level=2).diff(axis=1)

# Print obesity_variation
print(obesity_variation)

# Did you see how you can get different statistics about the data? By unstacking different levels and chaining them with statistical functions, you discovered that the obesity percentage went up in the last 10 years in the different countries.
-----------

# Stack obesity, get median of columns and unstack again
median_obesity = obesity.stack().median(axis=1).unstack()

# Print median_obesity
print(median_obesity)
-------
# Stack the first level, get sum, and unstack the second level
obesity_sum = obesity.stack(level=0).sum(axis=1).unstack(level=1)

# Print obesity_max
print(obesity_sum)
#  After performing aggregation functions, it's useful to unstack the DataFrame so we have a clear view of the results. In this exercise, you discovered how the mean percentage of obesity changes over time.
--------

# Stack country level, group by country and get the mean
obesity_mean = obesity.stack(level='country').groupby('country').mean()

# Print obesity_mean
print(obesity_mean)
--------

# Stack country level, group by country and get the median 
obesity_median = obesity.stack(level='country').groupby('country').median()

# Print obesity_median
print(obesity_median)
#  Combining the .stack() process with .groupby() and basic statistical functions results in very fast data manipulation. This approach allowed you to see that the mean and median percentages are not different after all.
----------
# Explode the values of bounds to a separate row
obesity_bounds = obesity['bounds'].explode()

# Merge obesity_bounds with country and perc_obesity columns of obesity using the indexes
obesity_final = obesity[['country', 'perc_obesity']].merge(obesity_bounds, 
                                        right_index=True, 
                                        left_index=True)

# Print obesity_final
print(obesity_final)

# Exploding the values in a column allowed you to get a cleaner DataFrame. However, you had to take an extra step to merge the values back into the original DataFrame. Let's see how to do it easily in the next exercise!
-----------

# Transform the list-like column named bounds  
obesity_explode = obesity.explode('bounds')

# Modify obesity_explode by resetting the index 
obesity_explode.reset_index(drop=True, inplace=True)

# Print obesity_explode
print(obesity_explode)
# Did you notice how the indexes were replicated after exploding the column? It is helpful if you explode a Series. In this case, you had to reset the index to have a cleaner DataFrame
----------
# Split the columns bounds using a hyphen as delimiter
obesity_split = obesity['bounds'].str.split("-")

# Print obesity_split
print(obesity_split)

--------
# Assign the result of the split to the bounds column
obesity_split = obesity.assign(bounds=obesity['bounds'].str.split('-'))

# Print obesity
print(obesity_split)

---------
# Transform the column bounds in the obesity DataFrame
obesity_split = obesity.assign(bounds=obesity['bounds'].str.split('-')).explode('bounds')

# Print obesity_split
print(obesity_split)
# You have successfully chained the split with the explode process to help you get a long DataFrame with just a few steps. Amazing!
-----------

# Import the json_normalize function
from pandas import json_normalize

# Normalize movies and separate the new columns with an underscore 
movies_norm = json_normalize(movies, sep="_")

# Reshape using director and producer as index, create movies from column starting from features
movies_long = pd.wide_to_long(movies_norm, stubnames='features', 
                      i=['director', 'producer'], j='movies', 
                      sep="_", suffix='\w+')

# Print movies_long
print(movies_long)
# Working with nested data is challenging, but you have successfully read it into a DataFrame. Understanding how to use the json_normalize() function will help you deal with nested data easily
--------

# Normalize the JSON contained in movies
normalize_movies = json_normalize(movies)

# Print normalize_movies
print(normalize_movies)
--------
# Specify the features column as the list of records 
normalize_movies = json_normalize(movies, 
                                  record_path='features')

# Print normalize_movies
print(normalize_movies)
-----------
# Specify director and producer to use as metadata for each record 
normalize_movies = json_normalize(movies, 
                                  record_path='features', 
                                  meta=['director' , 'producer'])

# Print normalize_movies
print(normalize_movies)
# The json_normalize() function comes in handy when you are dealing with a JSON that contains multiple records. In this exercise, you read the complex data structure into a DataFrame in just a few steps! Well done
-----------

# Define birds reading names and bird_facts lists into names and bird_facts columns
birds = pd.DataFrame(dict(names=names, bird_facts=bird_facts))

# Apply to bird_facts column the function loads from json module
data_split = birds['bird_facts'].apply(json.loads).apply(pd.Series)

# Remove the bird_facts column from birds
birds = birds.drop(columns='bird_facts')

# Concatenate the columns of birds and data_split
birds = pd.concat([birds,data_split], axis=1)

# Print birds
print(birds)
# Working with JSON files and deeply nested data can be tricky. In this exercise, you handled it by using json.loads and simple pandas functions. This is a common task you may perform as a Data Scientist.
---------

# Apply json.loads to the bird_facts column and transform it to a list
birds_facts = birds['bird_facts'].apply(json.loads).to_list()

# Print birds_facts
print(birds_facts)

----------
# Apply json.loads to the bird_facts column and transform it to a list 
birds_facts = birds['bird_facts'].apply(json.loads).to_list()

# Convert birds_fact into a JSON 
birds_dump = json.dumps(birds_facts)

# Read the JSON birds_dump into a DataFrame
birds_df = pd.read_json(birds_dump)

# Concatenate the 'names' column of birds with birds_df 
birds_final = pd.concat([birds['names'], birds_df], axis=1)

# Print birds_final
print(birds_final)

# You did it! You’re now able to reshape data to different formats in Python. 
-------------
