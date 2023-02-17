# Import pandas under the alias pd
import pandas as pd

# Assign the path to a string variable named data_path
data_path = 'kc_house_data.csv'

# Load the dataset as a dataframe named housing
housing = pd.read_csv(data_path)

# Print the price column of housing
print(housing['price'])

#  Notice that you did not have to specify a delimiter with the sep parameter, since the dataset was stored in the default, comma-separated format.
--------------

# Import numpy and tensorflow with their standard aliases
import numpy as np 
import tensorflow as tf

# Use a numpy array to define price as a 32-bit float
price = np.array(housing['price'], np.float32)

# Define waterfront as a Boolean using cast
waterfront = tf.cast(housing['waterfront'], tf.bool)

# Print price and waterfront
print(price)
print(waterfront)

# Notice that printing price yielded a numpy array; whereas printing waterfront yielded a tf.Tensor().
# output:
#     [221900. 538000. 180000. ... 402101. 400000. 325000.]
#     tf.Tensor([False False False ... False False False], shape=(21613,), dtype=bool)
--------------

# Import the keras module from tensorflow
from tensorflow import keras

# Compute the mean absolute error (mae)
loss = keras.losses.mse(price, predictions)

# Print the mean absolute error (mae)
print(loss.numpy())

------------------
# Import the keras module from tensorflow
from tensorflow import keras

# Compute the mean absolute error (mae)
loss = keras.losses.mae(price, predictions)

# Print the mean absolute error (mae)
print(loss.numpy())

# You may have noticed that the MAE was much smaller than the MSE, even though price and predictions were the same. 
# This is because the different loss functions penalize deviations of predictions from price differently.
# MSE does not like large deviations and punishes them harshly.
--------------

# Initialize a variable named scalar
scalar = Variable(1.0, float32)

# Define the model
def model(scalar, features = features):
  	return scalar * features

# Define a loss function
def loss_function(scalar, features = features, targets = targets):
	# Compute the predicted values
	predictions = model(scalar, features)
    
	# Return the mean absolute error loss
	return keras.losses.mae(targets, predictions)

# Evaluate the loss function and print the loss
print(loss_function(scalar).numpy())

# As you will see in the following lessons, this exercise was the equivalent of evaluating the loss function for a linear regression where the intercept is 0
------------
# Define a linear regression model
def linear_regression(intercept, slope, features = size_log):
	return intercept + features*slope

# Set loss_function() to take the variables as arguments
def loss_function(intercept, slope, features = size_log, targets = price_log):
	# Set the predicted values
	predictions = linear_regression(intercept, slope, features)
    
    # Return the mean squared error loss
	return keras.losses.mse(targets, predictions)


# Compute the loss for different slope and intercept values
print(loss_function(0.1, 0.1).numpy())
print(loss_function(0.1, 0.5).numpy())
# In the next exercise, you will actually run the regression and train intercept and slope.
----------

# Initialize an Adam optimizer
opt = keras.optimizers.Adam(0.5)

for j in range(100):
	# Apply minimize, pass the loss function, and supply the variables
	opt.minimize(lambda: loss_function(intercept, slope), var_list=[intercept, slope])

	# Print every 10th value of the loss
	if j % 10 == 0:
		print(loss_function(intercept, slope).numpy())

# Plot data and regression line
plot_results(intercept, slope)
# Notice that we printed loss_function(intercept, slope) every 10th execution for 100 executions. 
# Each time, the loss got closer to the minimum as the optimizer moved the slope and intercept parameters closer to their optimal values
# output:
#     9.669482
#     11.726698
#     1.1193314
#     1.6605737
#     0.7982884
#     0.8017316
#     0.6106565
#     0.59997976
#     0.5811015
#     0.5576158
----------

# Define the linear regression model
def linear_regression(params, feature1 = size_log, feature2 = bedrooms):
	return params[0] + feature1*params[1] + feature2*params[2]

# Define the loss function
def loss_function(params, targets = price_log, feature1 = size_log, feature2 = bedrooms):
	# Set the predicted values
	predictions = linear_regression(params, feature1, feature2)
  
	# Use the mean absolute error loss
	return keras.losses.mae(targets, predictions)

# Define the optimize operation
opt = keras.optimizers.Adam()

# Perform minimization and print trainable variables
for j in range(10):
	opt.minimize(lambda: loss_function(params), var_list=[params])
	print_results(params)
  
#    Note that params[2] tells us how much the price will increase in percentage terms if we add one more bedroom.
# You could train params[2] and the other model parameters by increasing the number of times we iterate over opt.

---------------

# Define the intercept and slope
intercept = Variable(10.0, float32)
slope = Variable(0.5, float32)

# Define the model
def linear_regression(intercept, slope, features):
	# Define the predicted values
	return intercept + features*slope

# Define the loss function
def loss_function(intercept, slope, targets, features):
	# Define the predicted values
	predictions = linear_regression(intercept, slope, features)
    
 	# Define the MSE loss
	return keras.losses.mse(targets, predictions)

# Notice that we did not use default argument values for the input data, features and targets. 
# This is because the input data has not been defined in advance. Instead, with batch training, we will load it during the training process
-----------
# Initialize Adam optimizer
opt = keras.optimizers.Adam()

# Load data in batches
for batch in pd.read_csv('kc_house_data.csv', chunksize=100):
	size_batch = np.array(batch['sqft_lot'], np.float32)

	# Extract the price values for the current batch
	price_batch = np.array(batch['price'], np.float32)

	# Complete the loss, fill in the variable list, and minimize
	opt.minimize(lambda: loss_function(intercept, slope, price_batch, size_batch), var_list=[intercept, slope])

# Print trained parameters
print(intercept.numpy(), slope.numpy())

# Batch training will be very useful when you train neural networks, which we will do next.

--------
