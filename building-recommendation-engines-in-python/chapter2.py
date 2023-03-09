# Content-based models are ideal for creating recommendations for products that have no user feedback data such as reviews or purchases.
----------
# Select only the rows with values in the name column equal to Toy Story
toy_story_genres = movie_genre_df[movie_genre_df['name'] == 'Toy Story']

# Create cross-tabulated DataFrame from name and genre_list columns
movie_cross_table = pd.crosstab(movie_genre_df['name'], movie_genre_df['genre_list'])

# Select only the rows with Toy Story as the index
toy_story_genres_ct = movie_cross_table[movie_cross_table.index == 'Toy Story']
print(toy_story_genres_ct)

# This newly formatted table with a vector contained in a row per movie and a column per feature will allow you to calculate distances and similarities between movies.
---------
# Yogi Bear and Toy Story both have the 'Children' and 'Comedy' attributes. The more genres that two movies have in common, the more likely it is that someone who liked one will like the other, so now we're going to apply this at a larger scale instead of just one pair of movies.
--------
# Import numpy and the distance metric
import numpy as np
from sklearn.metrics import jaccard_score

# Extract just the rows containing GoldenEye and Toy Story
goldeneye_values = movie_cross_table.loc['GoldenEye'].values
toy_story_values = movie_cross_table.loc['Toy Story'].values

# Find the similarity between GoldenEye and Toy Story
print(jaccard_score(goldeneye_values, toy_story_values))

# Repeat for GoldenEye and Skyfall
skyfall_values = movie_cross_table.loc['Skyfall'].values
print(jaccard_score(goldeneye_values, skyfall_values))
# As you can see, based on Jaccard similarity, GoldenEye and Skyfall (both James Bond movies) are more similar than GoldenEye and Toy Story (a spy movie and an animated kids movie).
# 0.14285714285714285
#     0.75
---------

# Import functions from scipy
from scipy.spatial.distance import pdist, squareform

# Calculate all pairwise distances
jaccard_distances = pdist(movie_cross_table.values, metric='jaccard')

# Convert the distances to a square matrix
jaccard_similarity_array = 1 - squareform(jaccard_distances)

# Wrap the array in a pandas DataFrame
jaccard_similarity_df = pd.DataFrame(jaccard_similarity_array, index=movie_cross_table.index, columns=movie_cross_table.index)

# Print the top 5 rows of the DataFrame
print(jaccard_similarity_df.head())
# As you can see, the table has the movies as rows and columns, allowing you to quickly look up any distance of any movie pairing
----------


