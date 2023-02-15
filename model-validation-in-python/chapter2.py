# Create dummy variables using pandas
X = pd.get_dummies(tic_tac_toe.iloc[:,0:9])
y = tic_tac_toe.iloc[:, 9]

# Create training and testing datasets. Use 10% for the test set
X_train, X_test, y_train,y_test  = train_test_split(X, y, test_size=.1, random_state=1111)

-----------
# Create temporary training and final testing datasets
X_temp, X_test, y_temp, y_test  =\
   train_test_split(X, y, test_size=0.20, random_state=1111)

# Create the final training and validation datasets
X_train, X_val, y_train, y_val  =\
   train_test_split(X_temp, y_temp, test_size=0.25, random_state=1111)

# You now have training, validation, and testing datasets, but do you know _when_ you need both validation and testing datasets?
# Anytime we are evaluating model performance repeatedly we need to create training, validation, and testing datasets
---------------

from sklearn.metrics import mean_absolute_error

# Manually calculate the MAE
n = len(predictions)
mae_one = sum(abs(y_test - predictions)) / n
print('With a manual calculation, the error is {}'.format(mae_one))

# Use scikit-learn to calculate the MAE
mae_two = mean_absolute_error(y_test, predictions)
print('Using scikit-learn, the error is {}'.format(mae_two))

# These predictions were about six wins off on average. This isn't too bad considering NBA teams play 82 games a year.
# Let's see how these errors would look if you used the mean squared error instead
--------------- 


from sklearn.metrics import mean_squared_error

n = len(predictions)
# Finish the manual calculation of the MSE
mse_one = sum((y_test - predictions)**2) / n
print('With a manual calculation, the error is {}'.format(mse_one))

# Use the scikit-learn function to calculate MSE
mse_two = mean_squared_error(y_test,predictions)
print('Using scikit-learn, the error is {}'.format(mse_two))

#  If you run any additional models, you will try to beat an MSE of 49.1,
# which is the average squared error of using your model. Although the MSE is not as interpretable as the MAE, 
# it will help us select a model that has fewer 'large' errors
----------------

# Find the East conference teams
east_teams = labels == "E"

# Create arrays for the true and predicted values
true_east = y_test[east_teams]
preds_east = predictions[east_teams]

# Print the accuracy metrics
print('The MAE for East teams is {}'.format(
    mae(true_east, preds_east)))

# Print the West accuracy
print('The MAE for West conference is {}'.format(west_error))

# It looks like the Western conference predictions were about two games better on average. Over the past few seasons, the Western teams have generally
# won the same number of games as the experts have predicted. Teams in the East are just not as predictable as those in the West.
--------------


# Calculate and print the accuracy
accuracy = (491 + 324) / (953)
print("The overall accuracy is {0: 0.2f}".format(accuracy))

# Calculate and print the precision
precision = (491) / (491 + 15)
print("The precision is {0: 0.2f}".format(precision))

# Calculate and print the recall
recall = (491) / (491 + 123)
print("The recall is {0: 0.2f}".format(recall))

# In this case, a true positive is a picture of an actual broken arm that was also predicted to be broken. Doctors are okay with a 
# few additional false positives (predicted broken, not actually broken), as long as you don't miss anyone who needs immediate medical attention.
-----------

from sklearn.metrics import confusion_matrix

# Create predictions
test_predictions = rfc.predict(X_test)

# Create and print the confusion matrix
cm = confusion_matrix(y_test, test_predictions)
print(cm)

# Print the true positives (actual 1s that were predicted 1s)
print("The number of true positives is: {}".format(cm[1, 1]))

# Row 1, column 1 represents the number of actual 1s that were predicted 1s (the true positives). 
# Always make sure you understand the orientation of the confusion matrix before you start using it!
-----------

from sklearn.metrics import precision_score

test_predictions = rfc.predict(X_test)

# Create precision or recall score based on the metric you imported
score = precision_score(y_test, test_predictions)

# Print the final result
print("The precision value is {0:.2f}".format(score))

#  Precision is the correct metric here. Sore-losers can't stand losing when they are certain they will win! For that reason,
# our model needs to be as precise as possible. With a precision of only 79%, you may need to try some other modeling techniques to improve this score
----------

# Update the rfr model
rfr = RandomForestRegressor(n_estimators=25,
                             random_state=1111,
                            max_features=2)
rfr.fit(X_train, y_train)

# Print the training and testing accuracies 
print('The training error is {0:.2f}'.format(
  mae(y_train, rfr.predict(X_train))))
print('The testing error is {0:.2f}'.format(
  mae(y_test, rfr.predict(X_test))))
# The training error is 3.90
# The testing error is 9.15
-------
# Update the rfr model
rfr = RandomForestRegressor(n_estimators=25,
                            random_state=1111,
                            max_features=11)
rfr.fit(X_train, y_train)

# Print the training and testing accuracies 
print('The training error is {0:.2f}'.format(
  mae(y_train, rfr.predict(X_train))))
print('The testing error is {0:.2f}'.format(
  mae(y_test, rfr.predict(X_test))))

#     The training error is 3.59
#     The testing error is 10.00
--------
# Update the rfr model
rfr = RandomForestRegressor(n_estimators=25,
                            random_state=1111,
                            max_features=4)
rfr.fit(X_train, y_train)

# Print the training and testing accuracies 
print('The training error is {0:.2f}'.format(
  mae(y_train, rfr.predict(X_train))))
print('The testing error is {0:.2f}'.format(
  mae(y_test, rfr.predict(X_test))))
#     The training error is 3.60
#     The testing error is 8.79
# The chart below shows the performance at various max feature values. Sometimes, setting parameter values can make a huge difference in model performance.
![chapter2lastimg](https://github.com/Rawan-Kh/DataCamp_ML_scientist_track/blob/main/model-validation-in-python/ch2last.png)
-------------------
from sklearn.metrics import accuracy_score

test_scores, train_scores = [], []
for i in [1, 2, 3, 4, 5, 10, 20, 50]:
    rfc = RandomForestClassifier(n_estimators=i, random_state=1111)
    rfc.fit(X_train, y_train)
    # Create predictions for the X_train and X_test datasets.
    train_predictions = rfc.predict(X_train)
    test_predictions = rfc.predict(X_test)
    # Append the accuracy score for the test and train predictions.
    train_scores.append(round(accuracy_score(y_train, train_predictions), 2))
    test_scores.append(round(accuracy_score(y_test, test_predictions), 2))
# Print the train and test scores.
print("The training scores were: {}".format(train_scores))
print("The testing scores were: {}".format(test_scores))

# Notice that with only one tree, both the train and test scores are low. As you add more trees, both errors improve. Even at 50 trees, this still might not be enough. 
# Every time you use more trees, you achieve higher accuracy. At some point though, more trees increase training time, but do not decrease testing error

#     The training scores were: [0.94, 0.93, 0.98, 0.97, 0.99, 1.0, 1.0, 1.0]
#     The testing scores were: [0.83, 0.79, 0.89, 0.91, 0.91, 0.93, 0.97, 0.98]
----------------


