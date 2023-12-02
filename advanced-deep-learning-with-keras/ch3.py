# Create three inputs layers of shape 1, one each for team 1, team 2, and home vs away.
# Lookup the team inputs in team_strength_model().
# Concatenate the team strengths with the home input and pass to a Dense layer.
# Create an Input for each team
team_in_1 = Input(shape=(1, ), name='Team-1-In')
team_in_2 = Input(shape=(1, ), name='Team-2-In')

# Create an input for home vs away
home_in = Input(shape=(1, ), name='Home-In')

# Lookup the team inputs in the team strength model
team_1_strength = team_strength_model(team_in_1)
team_2_strength = team_strength_model(team_in_2)

# Combine the team strengths with the home input using a Concatenate layer, 
# then add a Dense layer

out = Concatenate()([team_1_strength, team_2_strength, home_in])
out = Dense(1)(out)
# Now you have a model with 3 inputs!
--------
# Create a model using team_in_1, team_in_2, and home_in as inputs and out as the output.
# Compile the model using the 'adam' optimizer and 'mean_absolute_error' as the loss function.
# Import the model class
from tensorflow.keras.models import Model

# Make a Model
model = Model([team_in_1, team_in_2, home_in], out)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')
# Now our 3-input model is ready to meet some data!
---------
# Fit the model to the games_season dataset, using 'team_1', 'team_2' and 'home' columns as inputs, and the 'score_diff' column as the target.
# Fit the model using 1 epoch, 10% validation split and a batch size of 2048.
# Evaluate the model on games_tourney, using the same inputs and outputs.

# Fit the model to the games_season dataset
model.fit([games_season['team_1'], games_season['team_2'], games_season['home']],
          games_season['score_diff'],
          epochs=1, verbose=True, validation_split=0.1, batch_size=2048)

# Evaluate the model on the games_touney dataset
print(model.evaluate([games_tourney['team_1'], games_tourney['team_2'], games_tourney['home']], 
                      games_tourney['score_diff'], verbose=False))
# ----------
# Save the model plot to the file 'model.png'.
# Import and display 'model.png' into Python using matplotlib.

# Imports
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

# Plot the model
plot_model(model, to_file='model.png')

# Display the image
data = plt.imread('model.png')
plt.imshow(data)
plt.show()
-----------

# Use the model to predict on the games_tourney dataset. The model has three inputs: 'team_1', 'team_2', and 'home' columns. Assign the predictions to a new column, 'pred'.
# Predict
games_tourney['pred'] = model.predict([games_tourney['team_1'], 
                                       games_tourney['team_2'], 
                                       games_tourney['home']])
# Now you can try building a model for the tournament data based on your regular season predictions.
-----------

# Create a single input layer with 3 columns.
# Connect this input to a Dense layer with 1 unit.
# Create a model with input_tensor as the input and output_tensor as the output.
# Compile the model with 'adam' as the optimizer and 'mean_absolute_error' as the loss function.
# Create an input layer with 3 columns
input_tensor = Input(shape=(3, ))

# Pass it to a Dense layer with 1 unit
output_tensor = Dense(1)(input_tensor)

# Create a model
model = Model(input_tensor, output_tensor)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')
----------
# Fit the model to the games_tourney_train dataset using 1 epoch.
# The input columns are 'home', 'seed_diff', and 'pred'.
# The target column is 'score_diff'.
# Fit the model
model.fit(games_tourney_train[['home', 'seed_diff', 'pred']],
          games_tourney_train['score_diff'],
          epochs=1,
          verbose=True)
# ----------
# Evaluate the model on the games_tourney_test data.
# Recall that the model's inputs are 'home', 'seed_diff', and 'prediction' columns and the target column is 'score_diff'.
# Evaluate the model on the games_tourney_test dataset
print(model.evaluate(games_tourney_test[['home', 'seed_diff', 'prediction']],
                     games_tourney_test['score_diff'],
                     verbose=True))

----------
