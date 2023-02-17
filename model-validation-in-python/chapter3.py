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

# If our models are not generalizing well or if we have limited data, we should be careful using a single training/validation split.
# You should use the next lesson's topic: cross-validation.
----------------
from sklearn.model_selection import KFold

# Use KFold
kf = KFold(n_splits=5, shuffle=True, random_state=1111)

# Create splits
splits = kf.split(X)

# Print the number of indices
for train_index, val_index in splits:
    print("Number of training indices: %s" % len(train_index))
    print("Number of validation indices: %s" % len(val_index))
  
# This dataset has 85 rows. You have created five splits - 
# each containing 68 training and 17 validation indices. 
# You can use these indices to complete 5-fold cross-validation.  
--------------

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

rfc = RandomForestRegressor(n_estimators=25, random_state=1111)

# Access the training and validation indices of splits
for train_index, val_index in splits:
    # Setup the training and validation data
    X_train, y_train = X[train_index], y[train_index]
    X_val, y_val = X[val_index], y[val_index]
    # Fit the random forest model
    rfc.fit(X_train, y_train)
    # Make predictions, and print the accuracy
    predictions = rfc.predict(X_val)
    print("Split accuracy: " + str(mean_squared_error(y_val, predictions)))
# KFold() is a great method for accessing individual indices when completing cross-validation.
# One drawback is needing a for loop to work through the indices though. 
# In the next lesson, you will look at an automated method for cross-validation using sklearn.    
--------------

# Instruction 1: Load the cross-validation method
from sklearn.model_selection import cross_val_score

# Instruction 2: Load the random forest regression model
from sklearn.ensemble import RandomForestRegressor

# Instruction 3: Load the mean squared error method
# Instruction 4: Load the function for creating a scorer
from sklearn.metrics import mean_squared_error, make_scorer

-------------------
rfc = RandomForestRegressor(n_estimators=25, random_state=1111)
mse = make_scorer(mean_squared_error)

# Set up cross_val_score
cv = cross_val_score(estimator=rfc,
                     X=X_train,
                     y=y_train,
                     cv=10,
                     scoring=mse)

# Print the mean error
print(cv.mean())

# OUTPUT: 155.4061992697056
# You now have a baseline score to build on. If you decide to build additional 
# models or try new techniques, you should try to get an error lower than 155.56.
# Lower errors indicate that your popularity predictions are improving
----------------

# LOOCV can be used for both classification and regression models.
# Which of the following are reasons you might NOT run LOOCV
# A: The X dataset has 122,624 data points, which might be computationally expensive and slow.
# C:You want to test different values for 15 different parameters
-------------------

from sklearn.metrics import mean_absolute_error, make_scorer

# Create scorer
mae_scorer = make_scorer(mean_absolute_error)

rfr = RandomForestRegressor(n_estimators=15, random_state=1111)

# Implement LOOCV
scores = cross_val_score(rfr, X=X, y=y, cv=85, scoring=mae_scorer)

# Print the mean and standard deviation
print("The mean of the errors is: %s." % np.mean(scores))
print("The standard deviation of the errors is: %s." % np.std(scores))

# You have come along way with model validation techniques. The final chapter will wrap up model validation by 
# discussing how to select the best model and give an introduction to parameter tuning
---------------
