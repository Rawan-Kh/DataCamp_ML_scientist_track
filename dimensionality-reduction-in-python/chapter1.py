# dimension is the number of columns in dataset

---------

# Remove the feature without variance from this list
# it was generation column with std 0.0
number_cols = ['HP', 'Attack', 'Defense']
pokemon_df.describe()

--------
# Leave this list as is
number_cols = ['HP', 'Attack', 'Defense']

# Remove the feature without variance from this list
non_number_cols = ['Name', 'Type']

# Create a new DataFrame by subselecting the chosen features
df_selected = pokemon_df[number_cols + non_number_cols]

# Prints the first 5 lines of the new DataFrame
print(df_selected.head())

# Legendary column removed since it contained no variance
#        HP  Attack  Defense                   Name   Type  Legendary
#     0  45      49       49              Bulbasaur  Grass      False
#     1  60      62       63                Ivysaur  Grass      False
#     2  80      82       83               Venusaur  Grass      False
#     3  80     100      123  VenusaurMega Venusaur  Grass      False
#     4  39      52       43             Charmander   Fire      False
-----------

# Create a pairplot and color the points using the 'Gender' feature
sns.pairplot(ansur_df_1, hue="Gender", diag_kind='hist')

# Show the plot
plt.show()

----------
# Remove one of the redundant features
reduced_df = ansur_df_1.drop('stature_m', axis=1)

# Create a pairplot and color the points using the 'Gender' feature
sns.pairplot(reduced_df, hue='Gender')

# Show the plot
plt.show()

-------------

# Create a pairplot and color the points using the 'Gender' feature
sns.pairplot(ansur_df_2, hue="Gender", diag_kind='hist')


# Show the plot
plt.show()

------------
# Remove the redundant feature
reduced_df = ansur_df_2.drop('n_legs', axis=1)

# Create a pairplot and color the points using the 'Gender' feature
sns.pairplot(reduced_df, hue='Gender', diag_kind='hist')

# Show the plot
plt.show()

# the body height (inches) and stature (meters) hold the same information in a different unit + all the individuals in the second sample have two legs.
------------

# What advantage does feature selection have over feature extraction?
# Incorrect
# Nope, feature extraction will typically reduce dimensionality stronger than conservative feature selection.
# Nope, feature extraction algorithms try to preserve as much information as possible and are quite good at this.
# Correct! Extracted features can be quite hard to interpret.

----------
# tsne doesnt work on non numeric data 
------
# t-SNE is super powerful, but do you know exactly when to use it?
# When you want to visually explore the patterns in a high dimensional dataset.
------

# Non-numerical columns in the dataset
non_numeric = ['Branch', 'Gender', 'Component']

# Drop the non-numerical columns from df
df_numeric = df.drop(non_numeric, axis=1)

# Create a t-SNE model with learning rate 50
m = TSNE(learning_rate=50)

# Fit and transform the t-SNE model on the numeric dataset
tsne_features = m.fit_transform(df_numeric)
print(tsne_features.shape)

# t-SNE reduced the more than 90 features in the dataset to just 2 which you can now plot.
---------

# Color the points according to Army Component
sns.scatterplot(x="x", y="y", hue='Component', data=df)

# Show the plot
plt.show()

---------
# Color the points by Army Branch
sns.scatterplot(x="x", y="y", hue='Branch', data=df)

# Show the plot
plt.show()

-----------
# Color the points by Gender
sns.scatterplot(x="x", y="y", hue='Gender', data=df)

# Show the plot
plt.show()

# There is a Male and a Female cluster. t-SNE found these gender differences in body shape without being told about them explicitly!
# From the second plot you learned there are more males in the Combat Arms Branch.
---------
