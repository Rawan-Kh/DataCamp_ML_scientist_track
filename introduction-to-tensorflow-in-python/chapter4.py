# Define a Keras sequential model
model = keras.Sequential()

# Define the first dense layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(784,)))

# Define the second dense layer
model.add(keras.layers.Dense(8, activation='relu', input_shape=(784,)))

# Define the output layer
model.add(keras.layers.Dense(4,activation='softmax'))

# Print the model architecture
print(model.summary())

# Notice that we've defined a model, but we haven't compiled it. The compilation step in keras allows us to set the optimizer, 
# loss function, and other useful training parameters in a single line of code. Furthermore, the .summary() method allows us to view the model's architecture.
--------------
# Define the first dense layer
model.add(keras.layers.Dense(16, activation='sigmoid',input_shape=(784,)))

# Apply dropout to the first layer's output
model.add(keras.layers.Dropout(0.25))

# Define the output layer
model.add(keras.layers.Dense(4,activation='softmax'))

# Compile the model
model.compile('adam', loss='categorical_crossentropy')

# Print a model summary
print(model.summary())

# You've now defined and compiled a neural network using the keras sequential model. Notice that printing the .summary() method shows the layer type,
# output shape, and number of parameters of each layer.
-----------------

# For model 1, pass the input layer to layer 1 and layer 1 to layer 2
m1_layer1 = keras.layers.Dense(12, activation='sigmoid')(m1_inputs)
m1_layer2 = keras.layers.Dense(4, activation='softmax')(m1_layer1)

# For model 2, pass the input layer to layer 1 and layer 1 to layer 2
m2_layer1 = keras.layers.Dense(12, activation='relu')(m2_inputs)
m2_layer2 = keras.layers.Dense(4, activation='softmax')(m2_layer1)

# Merge model outputs and define a functional model
merged = keras.layers.add([m1_layer2, m2_layer2])
model = keras.Model(inputs=[m1_inputs, m2_inputs], outputs=merged)

# Print a model summary
print(model.summary())

# Notice that the .summary() method yields a new column: connected to. 
# This column tells you how layers connect to each other within the network. 
# We can see that dense_2, for instance, is connected to the input_2 layer. 
# We can also see that the add layer, which merged the two models, connected to both dense_1 and dense_3.
---------

# Define a sequential model
model = keras.Sequential()

# Define a hidden layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(784,)))

# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# Compile the model
model.compile('SGD', loss='categorical_crossentropy')

# Complete the fitting operation
model.fit(sign_language_features, sign_language_labels, epochs=5)

# You probably noticed that your only measure of performance improvement was the value of the loss function in the training sample,
# which is not particularly informative. You will improve on this in the next exercise.

---------------

# Define sequential model
model = keras.Sequential()

# Define the first layer
model.add(keras.layers.Dense(32, activation='sigmoid', input_shape=(784,)))

# Add activation function to classifier
model.add(keras.layers.Dense(4, activation='softmax'))

# Set the optimizer, loss function, and metrics
model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Add the number of epochs and the validation split
model.fit(sign_language_features, sign_language_labels, epochs=10, validation_split=0.1)

# With the keras API, you only needed 14 lines of code to define, compile, train, and validate a model. 
# You may have noticed that your model performed quite well. In just 10 epochs, we achieved a classification accuracy of over 90%
# in the validation sample!
--------------

# Define sequential model
model = keras.Sequential()

# Define the first layer
model.add(keras.layers.Dense(1024, activation='relu', input_shape=(784,)))

# Add activation function to classifier
model.add(keras.layers.Dense(4, activation='softmax'))

# Finish the model compilation
model.compile(optimizer=keras.optimizers.Adam(lr=0.001), 
              loss='categorical_crossentropy', metrics=['accuracy'])

# Complete the model fit operation
model.fit(sign_language_features, sign_language_labels, epochs=50, validation_split=0.5)

# You may have noticed that the validation loss, val_loss, was substantially higher than the training loss, 
# loss. Furthermore, if val_loss started to increase before the training process was terminated, then we may have overfitted.
# When this happens, you will want to try decreasing the number of epochs.
--------------

# Evaluate the small model using the train data
small_train = small_model.evaluate(train_features , train_labels)

# Evaluate the small model using the test data
small_test = small_model.evaluate(test_features , test_labels)

# Evaluate the large model using the train data
large_train = large_model.evaluate(train_features , train_labels)

# Evaluate the large model using the test data
large_test = large_model.evaluate(test_features , test_labels)

# Print losses
print('\n Small - Train: {}, Test: {}'.format(small_train, small_test))
print('Large - Train: {}, Test: {}'.format(large_train, large_test))

#  Notice that the gap between the test and train set losses is high for large_model, suggesting that overfitting may be an issue.
# Furthermore, both test and train set performance is better for large_model. This suggests that we may want to use large_model,
# but reduce the number of training epochs.
------------------

# Define feature columns for bedrooms and bathrooms
bedrooms = feature_column.numeric_column("bedrooms")
bathrooms = feature_column.numeric_column("bathrooms")

# Define the list of feature columns
feature_list = [bedrooms, bathrooms]

def input_fn():
	# Define the labels
	labels = np.array(housing['price'])
	# Define the features
	features = {'bedrooms':np.array(housing['bedrooms']), 
                'bathrooms':np.array(housing['bathrooms'])}
	return features, labels

#  In the next exercise, we'll use the feature columns and data input function to define and train an estimator.
---------------

# Define the model and set the number of steps
model = estimator.DNNRegressor(feature_columns=feature_list, hidden_units=[2,2])
model.train(input_fn, steps=1)

------------
# Define the model and set the number of steps
model = estimator.LinearRegressor(feature_columns=feature_list)
model.train(input_fn, steps=2)

#  Note that you have other premade estimator options, such as BoostedTreesRegressor(), and can also create your own custom estimators.
-------------
