# Predefined list to use as index
new_index = [['California', 'California', 'New York', 'Ohio'], 
             ['Los Angeles', 'San Francisco', 'New York', 'Cleveland']]

# Create a multi-level index using predefined new_index
churn_new = pd.MultiIndex.from_arrays(new_index, names=['state', 'city'])

# Assign the new index to the churn index
churn.index = churn_new

# Reshape by stacking churn DataFrame
churn_stack = churn.stack()

# Print churn_stack
print(churn_stack)

# Multi-level indexing is very useful for data analysis and manipulation. In this case, assigning a multi-level index to churn allowed you to add useful information
------------

# Set state and city as index modifying the DataFrame
churn.set_index(['state', 'city'], inplace=True)

# Reshape by stacking the second level
churn_stack = churn.stack(level=1)

# Print churn_stack
print(churn_stack)

# Choosing which level to stack can help you discover patterns. In this case, you can see how the total number of calls and minutes spent differ between day and night.
------------

# Stack churn by the time column level
churn_time = churn.stack(level='time')

# Print churn_time
print(churn_time)

---------
# Stack churn by the feature column level
churn_feature = churn.stack(level='feature')

# Print churn_feature
print(churn_feature)
# Stacking different levels helps you discover patterns in the data. For example, 
# you can see that customers use more internet at night. Stacking levels by name is helpful because you don't have to know the order of the indices.
--------

# Reshape the churn DataFrame by unstacking
churn_unstack = churn.unstack()

# Print churn_unstack
print(churn_unstack)

-----------
# Reshape churn by unstacking the first row level
churn_first = churn.unstack(level=0)

# Print churn_zero
print(churn_first)
-------------
# Reshape churn by unstacking the second row level
churn_second = churn.unstack(level=1)

# Print churn_second
print(churn_second)
# Remember that the .unstack() method reshapes the last level by default. 
# Notice how unstacking levels can lead to some missing values. You are going to learn how to handle these later on.
----------

# Unstack the time level from churn
churn_time = churn.unstack(level='time')

# Print churn_time
print(churn_time)
-------
# Sort the index in descending order
churn_time = churn.unstack(level='time').sort_index(ascending=False)

# Print churn_time
print(churn_time)
# Remember that the resulting unstacked DataFrame has the index level sorted in ascending order by default.
# Sorting the index can help you organize the data in a better way.
----------

# Unstack churn by type level
churn_type = churn.unstack(level='type')

# Stack the resulting DataFrame using the first column level
churn_final = churn_type.stack(level=0)

# Print churn_type
print(churn_final)
# Chaining the stacking and unstacking processes helps you reorganize the DataFrame, but it can also generate missing values.
-----------

# Switch the first and third row index levels in churn
churn_swap = churn.swaplevel(0, 2)

# Reshape by unstacking the last row level 
churn_unstack = churn_swap.unstack()

# Print churn_unstack
print(churn_unstack)

# Swapping the indices results in unstacking the exited status instead of the city level.
# This change was an important one to make. If you hadn't done that, you would have ended up with the wrong level unstacked
--------

# Unstack the first and second row level of churn
churn_unstack = churn.unstack(level=[0, 1])

# Stack the resulting DataFrame using plan and year
churn_py = churn_unstack.stack(['plan', 'year'])

# Switch the first and second column levels
churn_switch = churn_py.swaplevel(0, 1, axis=1)

# Print churn_switch
print(churn_switch)

# Unstacking and stacking several levels at a time gives you a lot of flexibility. 
# In this case, you combined them to create a completely reshaped DataFrame with switched column and row indexes.
-------
# Unstack churn level and fill missing values with zero
churn = churn.unstack(level='churn', fill_value=0)

# Sort by descending voice mail plan and ascending international plan
churn_sorted = churn.sort_index(level=['voice_mail_plan','international_plan'], ascending=[False,True])

# Print final DataFrame and observe pattern
print(churn_sorted)
# Unstacking the churn level showed that no customer has churned if they have a voicemail plan.
# Unstacking levels can lead to missing data, but you handled it easily in this exercise by setting fill_value to zero.
------------

# Stack the level type from churn
churn_stack = churn.stack(level='type')

# Fill the resulting missing values with zero 
churn_fill = churn_stack.fillna(0)

# Print churn_fill
print(churn_fill)
------------
# Stack the level scope without dropping rows with missing values
churn_stack = churn.stack(level='scope', dropna=False)

# Fill the resulting missing values with zero
churn_fill = churn_stack.fillna(0)

# Print churn_fill
print(churn_fill)

# Dropping rows containing only missing values is a good practice. 
# However, it is sometimes worth keeping that information. In that situation, stacking allows you to keep those rows by setting the dropna argument.
----------


