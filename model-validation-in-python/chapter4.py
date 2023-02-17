
# Review the parameters of rfr
print(rfr.get_params())

# Maximum Depth
max_depth = [4, 8, 12]

# Minimum samples for a split
min_samples_split = [2, 5, 10]

# Max features 
max_features = [4, 6, 8, 10]

# Hyperparameter tuning requires selecting parameters to tune, as well the possible values these parameters can be set to
------------------

from sklearn.ensemble import RandomForestRegressor

# Fill in rfr using your variables
rfr = RandomForestRegressor(
    n_estimators=100,
    max_depth=random.choice(max_depth),
    min_samples_split=random.choice(min_samples_split),
    max_features=random.choice(max_features))

# Print out the parameters
print(rfr.get_params())

# Notice that min_samples_split was randomly set to 2. Since you specified a random state, 
# min_samples_split will always be set to 2 if you only run this model one time
output:
#     {'bootstrap': True, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': 4, 'max_features': 10,
# 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2,
# 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}
-----------

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error

# Finish the dictionary by adding the max_depth parameter
param_dist = {"max_depth": [2, 4, 6, 8],
              "max_features": [2, 4, 6, 8, 10],
              "min_samples_split": [2, 4, 8, 16]}

# Create a random forest regression model
rfr = RandomForestRegressor(n_estimators=10, random_state=1111)

# Create a scorer to use (use the mean squared error)
scorer = make_scorer(mean_squared_error)

#  To use RandomizedSearchCV(), you need a distribution dictionary, an estimator, and a scorer—once you've got these, 
# you can run a random search to find the best parameters for your model
-------------------

# Import the method for random search
from sklearn.model_selection import RandomizedSearchCV

# Build a random search using param_dist, rfr, and scorer
random_search =\
    RandomizedSearchCV(
        estimator=rfr,
        param_distributions=param_dist,
        n_iter=10,
        cv=5,
        scoring=scorer)

# Although it takes a lot of steps, hyperparameter tuning with random search is well worth it 
# and can improve the accuracy of your models. Plus, you are already using cross-validation to validate your best model.
----------------

# You already ran a random search and saved the results of the most accurate model to rs.
# Which parameter set produces the best classification accuracy?

print(rs.best_estimator_)
# putput: RandomForestClassifier(max_depth=12, min_samples_split=4, n_estimators=20,random_state=1111)
# Perfect! These parameters do produce the best testing accuracy. Good job!
---------------
from sklearn.metrics import precision_score, make_scorer

# Create a precision scorer
precision = make_scorer(precision_score)
# Finalize the random search
rs = RandomizedSearchCV(
  estimator=rfc, param_distributions=param_dist,
  scoring = precision,
  cv=5, n_iter=10, random_state=1111)
rs.fit(X, y)

# print the mean test scores:
print('The accuracy for each run was: {}.'.format(rs.cv_results_['mean_test_score']))
# print the best model score:
print('The best accuracy for a single model was: {}'.format(rs.best_score_))

# Your model's precision was 93%! The best model accurately predicts a winning game 93% of the time. If you look at the mean test scores, 
# you can tell some of the other parameter sets did really poorly. Also, since you used cross-validation, you can be confident in your predictions. Well done
# The accuracy for each run was: [0.87614978 0.75561877 0.67740077 0.89141614 0.87024051 0.85772772 0.68244199 0.82867397 0.88717239 0.91980724].
# The best accuracy for a single model was: 0.9198072369317106
-------------


