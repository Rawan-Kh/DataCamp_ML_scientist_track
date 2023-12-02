# Create a single input layer with 2 columns.
# Connect this input to a Dense layer with 2 units.
# Create a model with input_tensor as the input and output_tensor as the output.
# Compile the model with 'adam' as the optimizer and 'mean_absolute_error' as the loss function.
# Define the input
input_tensor = Input(shape=(2, ))

# Define the output
output_tensor = Dense(2)(input_tensor)

# Create a model
model = Model(input_tensor, output_tensor)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')
# Now you have a multiple output model!
--------
# Fit the model to the games_tourney_train dataset using 100 epochs and a batch size of 16384.
# The input columns are 'seed_diff', and 'pred'.
# The target columns are 'score_1' and 'score_2'.
# Fit the model
model.fit(games_tourney_train[['seed_diff', 'pred']], 
          games_tourney_train[['score_1', 'score_2']],
          verbose=True,
          epochs=100,
          batch_size=16384)
# Let's look at the model weights.
---------
# Print the model's weights.
# Print the column means of the training data (games_tourney_train).
# Print the model's weights
print(model.get_weights())

# Print the column means of the training data
print(games_tourney_train.mean())
# Did you notice that both output weights are about ~72? This is because, on average, a team will score about 72 points in the tournament.
-------
# Evaluate the model on games_tourney_test.
# Use the same inputs and outputs as the training set.
# Evaluate the model on the tournament test data
print(model.evaluate(games_tourney_test[['seed_diff', 'pred']], 
                     games_tourney_test[['score_1', 'score_2']],
                     verbose=False))
# This model is pretty accurate at predicting tournament scores!
-----------
# Create a single input layer with 2 columns.
# The first output layer should have 1 unit with 'linear' activation and no bias term.
# The second output layer should have 1 unit with 'sigmoid' activation and no bias term. Also, use the first output layer as an input to this layer.
# Create a model with these input and outputs.

# Create an input layer with 2 columns
input_tensor = Input(shape=(2, ))

# Create the first output
output_tensor_1 = Dense(1, activation='linear', use_bias=False)(input_tensor)

# Create the second output(use the first output as input here)
output_tensor_2 = Dense(1, activation='sigmoid', use_bias=False)(output_tensor_1)

# Create a model with 2 outputs
model = Model(input_tensor, [output_tensor_1, output_tensor_2])
# This kind of model is only possible with a neural network.
------------

# Import Adam from keras.optimizers.
# Compile the model with 2 losses: 'mean_absolute_error' and 'binary_crossentropy', and use the Adam optimizer with a learning rate of 0.01.
# Fit the model with 'seed_diff' and 'pred' columns as the inputs and 'score_diff' and 'won' columns as the targets.
# Use 10 epochs and a batch size of 16384.
# Import the Adam optimizer
from tensorflow.keras.optimizers import Adam

# Compile the model with 2 losses and the Adam optimzer with a higher learning rate
model.compile(loss=['mean_absolute_error', 'binary_crossentropy'], optimizer=Adam(learning_rate=0.01))

# Fit the model to the tournament training data, with 2 inputs and 2 outputs
model.fit(games_tourney_train[['seed_diff', 'pred']],
          [games_tourney_train[['score_diff']], games_tourney_train[['won']]],
          epochs=10,
          verbose=True,
          batch_size=16384)
# You just fit a model that is both a classifier and a regressor!
----------

# Print the model's weights.
# Print the column means of the training data (games_tourney_train).
# Print the model weights
print(model.get_weights())

# Print the training data means
print(games_tourney_train.mean())
-----------
# Print the approximate win probability predicted for a close game (1 point difference).
# Print the approximate win probability predicted blowout game (10 point difference).
# Import the sigmoid function from scipy
from scipy.special import expit as sigmoid

# Weight from the model
weight = 0.14

# Print the approximate win probability predicted close game
print(sigmoid(1 * weight))

# Print the approximate win probability predicted blowout game
print(sigmoid(10 * weight))
# So sigmoid(1 * 0.14) is 0.53, which represents a pretty close game and sigmoid(10 * 0.14) is 0.80, which represents a pretty likely win. In other words, if the model predicts a win of 1 point, it is less sure of the win than if it predicts 10 points. Who says neural networks are black boxes?

--------
# Evaluate the model on games_tourney_test.
# Use the same inputs and outputs as the training set.
# Evaluate the model on new data
print(model.evaluate(games_tourney_test[['seed_diff', 'pred']],
                    [games_tourney_test[['score_diff']], games_tourney_test[['won']]], 
                     verbose=False))
# Turns out you can have your cake and eat it too! This model is both a good regressor and a good classifier!

-----------

