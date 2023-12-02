# Count the number of unique teams.
# Create an embedding layer that maps each team ID to a single number representing that team's strength.
# The output shape should be 1 dimension (as we want to represent the teams by a single number).
# The input length should be 1 dimension (as each team is represented by exactly one id).
# Imports
from tensorflow.keras.layers import Embedding
import numpy as np
from numpy import unique

# Count the unique number of teams
n_teams = np.unique(games_season['team_1']).shape[0]

# Create an embedding layer
team_lookup = Embedding(input_dim=n_teams,
                        output_dim=1,
                        input_length=1,
                        name='Team-Strength')
# The embedding layer is a lot like a dictionary, but your model learns the values for each key.
--------------

# Create a 1D input layer for the team ID (which will be an integer). Be sure to set the correct input shape!
# Pass this input to the team strength lookup layer you created previously.
# Flatten the output of the team strength lookup.
# Create a model that uses the 1D input as input and flattened team strength as output.
# Imports
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.models import Model

# Create an input layer for the team ID
teamid_in = Input(shape=(1, ))

# Lookup the input in the team strength embedding layer
strength_lookup = team_lookup(teamid_in)

# Flatten the output
strength_lookup_flat = Flatten()(strength_lookup)

# Combine the operations into a single, re-usable model
team_strength_model = Model(teamid_in, strength_lookup_flat, name='Team-Strength-Model')
# The model will be reusable, so you can use it in two places in your final model.
---------

# Create an input layer to use for team 1. Recall that our input dimension is 1.
# Name the input "Team-1-In" so you can later distinguish it from team 2.
# Create an input layer to use for team 2, named "Team-2-In".

# Load the input layer from tensorflow.keras.layers
from tensorflow.keras.layers import Input

# Input layer for team 1
team_in_1 = Input(shape=(1, ), name='Team-1-In')

# Separate input layer for team 2
team_in_2 = Input(shape=(1, ), name='Team-2-In')
# These two inputs will be used later for the shared layer.
-----------
# Lookup the first team ID in the team strength model.
# Lookup the second team ID in the team strength model.
# Lookup team 1 in the team strength model
team_1_strength = team_strength_model(team_in_1)

# Lookup team 2 in the team strength model
team_2_strength = team_strength_model(team_in_2)
# Now your model knows how strong each team is.
---------
# Import the Subtract layer from keras.layers.
# Combine the two-team strength lookups you did earlier.
# Import the Subtract layer from tensorflow.keras
from tensorflow.keras.layers import Subtract

# Create a subtract layer using the inputs from the previous exercise
score_diff = Subtract()([team_1_strength, team_2_strength])

# This setup subracts the team strength ratings to determine a winner.-
---------
# Define a model with the two teams as inputs and use the score difference as the output.
# Compile the model with the 'adam' optimizer and 'mean_absolute_error' loss.
# Imports
from tensorflow.keras.layers import Subtract
from tensorflow.keras.models import Model

# Subtraction layer from previous exercise
score_diff = Subtract()([team_1_strength, team_2_strength])

# Create the model
model = Model([team_in_1, team_in_2], score_diff)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')
-----------
# Assign the 'team_1' and 'team_2' columns from games_season to input_1 and input_2, respectively.
# Use 'score_diff' column from games_season as the target.
# Fit the model using 1 epoch, a batch size of 2048, and a 10% validation split.
# Get the team_1 column from the regular season data
input_1 = games_season['team_1']

# Get the team_2 column from the regular season data
input_2 = games_season['team_2']

# Fit the model to input 1 and 2, using score diff as a target
model.fit([input_1, input_2], games_season['score_diff'], epochs=1, batch_size=2048, validation_split=0.1, verbose=True)
# Now our model has learned a strength rating for every team.
----------

# Assign the 'team_1' and 'team_2' columns from games_tourney to input_1 and input_2, respectively.
# Evaluate the model.
# Get team_1 from the tournament data
input_1 = games_tourney['team_1']

# Get team_2 from the tournament data
input_2 = games_tourney['team_2']

# Evaluate the model using these inputs
print(model.evaluate([input_1, input_2], games_tourney['score_diff'], verbose=False))
# Its time to move on to models with more than two inputs.
---------

