# Import the logistic regression
from sklearn.linear_model import LogisticRegression

# Define the vector of targets and matrix of features
y = movies.label
X = movies.drop('label', axis=1)

# Build a logistic regression model and calculate the accuracy
log_reg = LogisticRegression().fit(X, y)

print('Accuracy of logistic regression: ', log_reg.score(X, y))
# You have built your first logistic regression model and checked its accuracy! Let's practice some more!
------------
# Define the vector of targets and matrix of features
y = tweets.airline_sentiment
X = tweets.drop('airline_sentiment', axis=1)

# Build a logistic regression model and calculate the accuracy
log_reg = LogisticRegression().fit(X, y)
print('Accuracy of logistic regression: ', log_reg.score(X, y))

# Create an array of prediction
y_predict = log_reg.predict(X)

# Print the accuracy using accuracy score
print('Accuracy of logistic regression: ', accuracy_score(y, y_predict))
# You have built another logistic regression model and calculated its accuracy in two different ways. Have you noticed how the calculated accuracy scores are the same? This will not always be the case for other methods because the .score() function can use other default model performance metrics. So, use accuracy_score() to be certain that you are calculating the accuracy when you are training a different supervised learning model.
------------

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=321)

# Train a logistic regression
log_reg = LogisticRegression().fit(X_train,y_train)

# Predict the probability of the 0 class
prob_0 = log_reg.predict_proba(X_test)[:, 0]
# Predict the probability of the 1 class
prob_1 = log_reg.predict_proba(X_test)[:, 1]

print("First 10 predicted probabilities of class 0: ", prob_0[:10])
print("First 10 predicted probabilities of class 1: ", prob_1[:10])

# Did you notice how the probabilities of class 0 and class 1 add up to 1 for each instance? In problems where the proportion of one class is larger than the other, we might want to work with predicted probabilities instead of predicted classes
-----------



