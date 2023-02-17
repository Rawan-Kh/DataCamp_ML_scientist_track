# Import constant from TensorFlow
from tensorflow import constant

# Convert the credit_numpy array into a tensorflow constant
credit_constant = constant(credit_numpy)

# Print constant datatype
print('\n The datatype is:', credit_constant.dtype)

# Print constant shape
print('\n The shape is:', credit_constant.shape)

# You now understand how constants are used in tensorflow. In the following exercise, you'll practice defining variables
---------------

# Define the 1-dimensional variable A1
A1 = Variable([1, 2, 3, 4])

# Print the variable A1
print('\n A1: ', A1)

# Convert A1 to a numpy array and assign it to B1
B1 = A1.numpy()

# Print B1
print('\n B1: ', B1)
# Did you notice any differences between the print statements for A1 and B1? 
# In our next exercise, we'll review how to check the properties of a tensor after it is already defined
#      A1:  <tf.Variable 'Variable:0' shape=(4,) dtype=int32, numpy=array([1, 2, 3, 4], dtype=int32)>
    
#      B1:  [1 2 3 4]
# ---------------

# Define tensors A1 and A23 as constants
A1 = constant([1, 2, 3, 4])
A23 = constant([[1, 2, 3], [1, 6, 4]])

# Define B1 and B23 to have the correct shape
B1 = ones_like(A1)
B23 = ones_like(A23)

# Perform element-wise multiplication
C1 = multiply(A1, B1)
C23 = multiply(A23, B23)

# Print the tensors C1 and C23
print('\n C1: {}'.format(C1.numpy()))
print('\n C23: {}'.format(C23.numpy()))

# Notice how performing element-wise multiplication with tensors of ones leaves the original tensors unchanged.
----------------

# Define features, params, and bill as constants
features = constant([[2, 24], [2, 26], [2, 57], [1, 37]])
params = constant([[1000], [150]])
bill = constant([[3913], [2682], [8617], [64400]])

# Compute billpred using features and params
billpred = matmul(features,params)

# Compute and print the error
error = bill - billpred
print(error.numpy())

# Understanding matrix multiplication will make things simpler when we start making predictions with linear models.
------------------

# Understanding how to sum over tensor dimensions will be helpful when preparing datasets and training models.
-----------------
# Reshape the grayscale image tensor into a vector
gray_vector = reshape(gray_tensor, (784, 1))

# Reshape the color image tensor into a vector
color_vector = reshape(color_tensor, (2352, 1))

# Notice that there are 3 times as many elements in color_vector as there are in gray_vector, since color_tensor has 3 color channels

----------------
def compute_gradient(x0):
  	# Define x as a variable with an initial value of x0
	x = Variable(x0)
	with GradientTape() as tape:
		tape.watch(x)
        # Define y using the multiply operation
		y = multiply(x,x)
    # Return the gradient of y with respect to x
	return tape.gradient(y, x).numpy()

# Compute and print gradients at x = -1, 1, and 0
print(compute_gradient(-1.0))
print(compute_gradient(1.0))
print(compute_gradient(0.0))
# Notice that the slope is positive at x = 1, which means that we can lower the loss by reducing x. The slope is negative at x = -1,
# which means that we can lower the loss by increasing x. The slope at x = 0 is 0,
# which means that we cannot lower the loss by either increasing or decreasing x. This is because the loss is minimized at x = 0.
--------------------

# Reshape model from a 1x3 to a 3x1 tensor
model = reshape(model, (3, 1))

# Multiply letter by model
output = matmul(letter, model)

# Sum over output and print prediction using the numpy method
prediction = reduce_sum(output)
print(prediction.numpy())

# Your model found that prediction=1.0 and correctly classified the letter as a K. In the coming chapters, you will use data to train a model, model,
# and then combine this with matrix multiplication, matmul(letter, model), as we have done here, to make predictions about the classes of objects.
----------
