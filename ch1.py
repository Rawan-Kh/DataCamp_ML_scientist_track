# Import the Dense layer function from keras.layers.
# Create a Dense layer with 1 unit.
# Pass input_tensor to output_layer().
from tensorflow.keras.layers import Dense

# Input layer
input_tensor = Input(shape=(1, ))

# Dense layer
output_layer = Dense(1)

# Connect the dense layer to the input_tensor
output_tensor = output_layer(input_tensor)

# This network will take the input, apply a linear coefficient to it, and return the result.
--------

# Import the Input and Dense functions from keras.layers.
# Create an input layer of shape 1.
# Again, create a dense layer with 1 unit and pass input_tensor directly to it.

# Use Input() to create an input layer.
# Use Dense() to create a dense layer.
# Pass input_tensor to the output of Dense(1).
# Load layers
from tensorflow.keras.layers import Input, Dense

# Input layer
input_tensor = Input(shape=(1,))

# Create a dense layer and connect the dense layer to the input_tensor in one step
# Note that you did this in 2 steps in the previous exercise, but are doing it in one step now
output_tensor = Dense(1)(input_tensor)

# ------------
# Import Model from keras.models to create a keras model.
# Use the input layer and output layer you already defined as the model's input and output.
# Input/dense/output layers
from tensorflow.keras.layers import Input, Dense
input_tensor = Input(shape=(1, ))
output_tensor = Dense(1)(input_tensor)

# Build the model
from tensorflow.keras.models import Model
model = Model(input_tensor, output_tensor)

# This model is a complete neural network, ready to learn from data and make prediction.
--------
# Compile the model you created (model).
# Use the 'adam' optimizer.
# Use mean absolute error (or 'mean_absolute_error') loss.
# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

----------
# Import the plotting function
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

# Summarize the model
model.summary()

# Plot the model
plot_model(model, to_file='model.png')

# Display the image
data = plt.imread('model.png')
plt.imshow(data)
plt.show()

---------
# Fit the model with seed_diff as the input variable and score_diff as the output variable.
# Use 1 epoch, a batch size of 128, and a 10% validation split
# Now fit the model
model.fit(games_tourney_train['seed_diff'], games_tourney_train['score_diff'],
          epochs=1,
          batch_size=128,
          validation_split=0.1,
          verbose=True)
------
# Assign the test data (seed_diff column) to X_test.
# Assign the target data (score_diff column) to y_test.
# Evaluate the model on X_test and y_test.
# Load the X variable from the test data
X_test = games_tourney_test['seed_diff']

# Load the y variable from the test data
y_test = games_tourney_test['score_diff']

# Evaluate the model on the test data
print(model.evaluate(X_test, y_test, verbose=False))

-----------
