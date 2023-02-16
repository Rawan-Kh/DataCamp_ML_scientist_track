# Create two different samples of 200 observations 
sample1 = tic_tac_toe.sample(200, random_state=1111)
sample2 = tic_tac_toe.sample(200, random_state=1171)

# Print the number of common observations 
print(len([index for index in sample1.index if index in sample2.index]))

# Print the number of observations in the Class column for both samples 
print(sample1['Class'].value_counts())
print(sample2['Class'].value_counts())

#  Notice that there are a varying number of positive observations for both sample test sets.
# Sometimes creating a single test holdout sample is not enough to achieve the high levels of 
# model validation you want. You need to use something more robust
#     positive    134
#     negative     66
#     Name: Class, dtype: int64
#     positive    123
#     negative     77
#     Name: Class, dtype: int64
-------------------
