# Inspect the listening_history_df DataFrame
print(listening_history_df.head())

# Calculate the number of unique values
print(listening_history_df[['Rating', 'Skipped Track']].nunique())

# Display a histogram of the values in the Rating column
listening_history_df['Rating'].hist()
plt.show()
--------------

# A rating is a good example of explicit data as it required the user to directly give feedback, while their dislike for a song is only implied by the action of them skipping it before it ends.
----------
# Get the counts of occurrences of each movie title
movie_popularity = user_ratings_df["title"].value_counts()

# Inspect the most common values
print(movie_popularity.head().index)

--------
# Find the mean of the ratings given to each title
average_rating_df = user_ratings_df[["title", "rating"]].groupby('title').mean()

# Order the entries by highest average rating to lowest
sorted_average_ratings = average_rating_df.sort_values(by='rating', ascending=False)

# Inspect the top movies
print(sorted_average_ratings.head())
# Despite this being a real-world dataset, you might be surprised that the highest-ranked movies are not movies that most people have heard of. This is because very infrequently-viewed movies are skewing the results. You will address this issue in the next exercise.
#                                      rating
#     title                                      
#     Gena the Crocodile (1969)               5.0
#     True Stories (1986)                     5.0
#     Cosmic Scrat-tastrophe (2015)           5.0
#     Love and Pigeons (1985)                 5.0
#     Red Sorghum (Hong gao liang) (1987)     5.0
-------

# Create a list of only movies appearing > 50 times in the dataset
movie_popularity = user_ratings_df["title"].value_counts()
popular_movies = movie_popularity[movie_popularity > 50].index

# Use this popular_movies list to filter the original DataFrame
popular_movies_rankings =  user_ratings_df[user_ratings_df["title"].isin(popular_movies)]

# Find the average rating given to these frequently watched films
popular_movies_average_rankings = popular_movies_rankings[["title", "rating"]].groupby('title').mean()
print(popular_movies_average_rankings.sort_values(by="rating", ascending=False).head())

# You are now able to make intelligent non-personalized recommendations that combine both the ratings of an item and how frequently it has been interacted with.
-------

from itertools import permutations

# Create the function to find all permutations
def find_movie_pairs(x):
  pairs = pd.DataFrame(list(permutations(x.values, 2)),
                       columns=['movie_a', 'movie_b'])
  return pairs

# Apply the function to the title column and reset the index
movie_combinations = user_ratings_df.groupby('userId')['title'].apply(find_movie_pairs)

print(movie_combinations)
---------
from itertools import permutations

# Create the function to find all permutations
def find_movie_pairs(x):
  pairs = pd.DataFrame(list(permutations(x.values, 2)),
                       columns=['movie_a', 'movie_b'])
  return pairs

# Apply the function to the title column and reset the index
movie_combinations = user_ratings_df.groupby('userId')['title'].apply(
  find_movie_pairs).reset_index(drop=True)

print(movie_combinations)
# You now have a clean table of all of the movies that were watched by the same user, which can be used to find the most commonly paired movies.
----------

# Calculate how often each item in movie_a occurs with the items in movie_b
combination_counts = movie_combinations.groupby(['movie_a', 'movie_b']).size()

# Convert the results to a DataFrame and reset the index
combination_counts_df = combination_counts.to_frame(name='size').reset_index()
print(combination_counts_df.head())
# n the next exercise, you will use this aggregated DataFrame to generate recommendations for any movie in the dataset.
----

import matplotlib.pyplot as plt

# Sort the counts from highest to lowest
combination_counts_df.sort_values('size', ascending=False, inplace=True)

# Find the movies most frequently watched by people who watched Thor
thor_df = combination_counts_df[combination_counts_df['movie_a'] == 'Thor']

# Plot the results
thor_df.plot.bar(x="movie_b")
plt.show()

# You can see that 21 Jump Street was the most commonly watched movie by those who watched Thor. This means that it would be a good movie to recommend Thor watchers as it shows they have similar fans.
-------------


