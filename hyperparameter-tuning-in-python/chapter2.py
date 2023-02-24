# Create the function
def gbm_grid_search(learning_rate, max_depth):

	# Create the model
    model = GradientBoostingClassifier(learning_rate=learning_rate, max_depth=max_depth)
    
    # Use the model to make predictions
    predictions = model.fit(X_train, y_train).predict(X_test)
    
    # Return the hyperparameters and score
    return([learning_rate, max_depth, accuracy_score(y_test, predictions)])
# You now have a function you can call to test different combinations of two hyperparameters for the GBM algorithm. In the next exercise we will use it to test some values and analyze the results.  
--------------

# Create the relevant lists
results_list = []
learn_rate_list = [0.01, 0.1, 0.5]
max_depth_list = [2, 4, 6]

# Create the for loop
for learn_rate in learn_rate_list:
    for max_depth in max_depth_list:
        results_list.append(gbm_grid_search(learn_rate,max_depth))

# Print the results
print(results_list)   

--------
results_list = []
learn_rate_list = [0.01, 0.1, 0.5]
max_depth_list = [2,4,6]

# Extend the function input
def gbm_grid_search_extended(learn_rate, max_depth, subsample):

	# Extend the model creation section
    model = GradientBoostingClassifier(learning_rate=learn_rate, max_depth=max_depth, subsample=subsample)
    
    predictions = model.fit(X_train, y_train).predict(X_test)
    
    # Extend the return part
    return([learn_rate, max_depth, subsample, accuracy_score(y_test, predictions)])       
 ----------
results_list = []

# Create the new list to test
subsample_list = [0.4 , 0.6]

for learn_rate in learn_rate_list:
    for max_depth in max_depth_list:
    
    	# Extend the for loop
        for subsample in subsample_list:
        	
            # Extend the results to include the new hyperparameter
            results_list.append(gbm_grid_search_extended(learn_rate, max_depth, subsample))
            
# Print results
print(results_list)            
# You have effectively built your own grid search! You went from 2 to 3 hyperparameters and can see how you could extend that to even more values and hyperparameters. That was a lot of effort though. Be warned - we are now entering a world that can get very computationally expensive very fast!
---------

# How many models would be created when running a grid search over the following hyperparameters and values for a GBM algorithm?
# 1215
# For every value of one hyperparameter, we test EVERY value of EVERY other hyperparameter. So you correctly multiplied the number of values (the lengths of the lists).
---------

# By looking at the Scikit Learn documentation (or your excellent memory!) you know that number_attempts is not a valid hyperparameter. This GridSearchCV will not fit to our data.
# Model #3:
#  GridSearchCV(
#     estimator = GradientBoostingClassifier(),
#     param_grid = {'number_attempts': [2, 4, 6], 'max_depth': [3, 6, 9, 12]},
#     scoring='accuracy',
#     n_jobs=2,
#     cv=7,
#     refit=True) 
-------

# Create a Random Forest Classifier with specified criterion
rf_class = RandomForestClassifier(criterion='entropy')

# Create the parameter grid
param_grid = {'max_depth': [2, 4, 8, 15], 'max_features': ['auto', 'sqrt']} 

# Create a GridSearchCV object
grid_rf_class = GridSearchCV(
    estimator=rf_class,
    param_grid=param_grid,
    scoring='roc_auc',
    n_jobs=4,
    cv=5,
    refit=True, return_train_score=True)
print(grid_rf_class)
# You now understand all the inputs to a GridSearchCV object and can tune many different hyperparameters and many different values for each on a chosen algorithm!
-----------

# Which of the following parameters must be set in order to be able to directly use the best_estimator_ property for predictions?
# refit = True
# Correct! When we set this to true, the creation of the grid search object automatically refits the best parameters on the whole training set and creates the best_estimator_ property.
--------

# Read the cv_results property into a dataframe & print it out
cv_results_df = pd.DataFrame(grid_rf_class.cv_results_)
print(cv_results_df)

# Extract and print the column with a dictionary of hyperparameters used
column = cv_results_df.loc[:, [ "params"]]
print(column)

# Extract and print the row that had the best mean test score
best_row = cv_results_df[cv_results_df['rank_test_score'] == 1 ]
print(best_row)
# You have built invaluable skills in looking 'under the hood' at what your grid search is doing by extracting and analysing the cv_results_ propert
--------

# Print out the ROC_AUC score from the best-performing square
best_score = grid_rf_class.best_score_
print(best_score)

# Create a variable from the row related to the best-performing square
cv_results_df = pd.DataFrame(grid_rf_class.cv_results_)
best_row = cv_results_df.loc[[grid_rf_class.best_index_]]
print(best_row)

# Get the n_estimators parameter from the best-performing square and print
best_n_estimators = grid_rf_class.best_params_["n_estimators"]
print(best_n_estimators)

#  Being able to quickly find and prioritize the huge volume of information given back from machine learning modeling output is a great skill. Here you had great practice doing that with cv_results_ by quickly isolating the key information on the best performing square. This will be very important when your grids grow from 12 squares to many more!

-------
# See what type of object the best_estimator_ property is
print(type(grid_rf_class.best_estimator_))

# Create an array of predictions directly using the best_estimator_ property
predictions = grid_rf_class.best_estimator_.predict(X_test)

# Take a look to confirm it worked, this should be an array of 1's and 0's
print(predictions[0:5])

# Now create a confusion matrix 
print("Confusion Matrix \n", confusion_matrix(y_test, predictions))

# Get the ROC-AUC score
predictions_proba = grid_rf_class.best_estimator_.predict_proba(X_test)[:,1]
print("ROC-AUC Score \n", roc_auc_score(y_test, predictions_proba))
#  The .best_estimator_ property is a really powerful property to understand for streamlining your machine learning model building process. You now can run a grid search and seamlessly use the best model from that search to make predictions. Piece of cake!
--------


