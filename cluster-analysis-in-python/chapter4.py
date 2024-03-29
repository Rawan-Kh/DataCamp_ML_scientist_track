# Import image class of matplotlib
from matplotlib import image as img

# Read batman image and print dimensions
batman_image = img.imread('batman.jpg')
print(batman_image.shape)

# Store RGB values of all pixels in lists r, g and b
for row in batman_image:
    for temp_r, temp_g, temp_b in row:
        r.append(temp_r)
        g.append(temp_g)
        b.append(temp_b)
      
# You have successfully extracted the RGB values of the image into three lists, one for each color channe
------------

distortions = []
num_clusters = range(1, 7)

# Create a list of distortions from the kmeans function
for i in num_clusters:
    cluster_centers, distortion =  kmeans(batman_df[['scaled_red', 'scaled_blue','scaled_green']], i)
    distortions.append(distortion)

# Create a DataFrame with two lists, num_clusters and distortions
elbow_plot = pd.DataFrame({'num_clusters': num_clusters, 'distortions': distortions})

# Create a line plot of num_clusters and distortions
sns.lineplot(x='num_clusters', y='distortions', data = elbow_plot)
plt.xticks(num_clusters)
plt.show()

#  Notice that there are three distinct colors present in the image, which is supported by the elbow plot
------------

# Get standard deviations of each color
r_std, g_std, b_std = batman_df[['red', 'green', 'blue']].std()

for cluster_center in cluster_centers:
    scaled_r, scaled_g, scaled_b = cluster_center
    # Convert each standardized value to scaled value
    colors.append((
        scaled_r * r_std / 255,
        scaled_g * g_std / 255,
        scaled_b * b_std / 255
    ))

# Display colors of cluster centers
plt.imshow([colors])
plt.show()

#  Notice the three colors resemble the three that are indicative from visual inspection of the image
-------------

# Import TfidfVectorizer class from sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.75, max_features=50,  min_df=0.1,tokenizer=remove_noise )

# Use the .fit_transform() method on the list plots
tfidf_matrix = tfidf_vectorizer.fit_transform(plots)

----------

num_clusters = 2

# Generate cluster centers through the kmeans function
cluster_centers, distortion =kmeans(tfidf_matrix.todense(), num_clusters)

# Generate terms from the tfidf_vectorizer object
terms = tfidf_vectorizer.get_feature_names_out()

for i in range(num_clusters):
    # Sort the terms and print top 3 terms
    center_terms = dict(zip(terms, list(cluster_centers[i])))
    sorted_terms = sorted(center_terms, key=center_terms.get, reverse=True)
    print(sorted_terms[:3])
    
# Notice positive, warm words in the first cluster and words referring to action in the second cluster
# ['father', 'back', 'one']
# ['police', 'man', 'killed']
-----------

# What should you do if you have too many features for clustering?
# it may prove to be too difficult to visualize more than three features at a time.
# With increasing complexity, it may take too much time to perform the analysis.
# You should explore steps to reduce the number of features.
--------------
# Print the size of the clusters
print(fifa.groupby('cluster_labels')['ID'].count())

# Print the mean value of wages in each cluster
print(fifa.groupby('cluster_labels')['eur_wage'].mean())

# In this example, the cluster sizes are not very different, and there are no significant differences that
# can be seen in the wages. Further analysis is required to validate these clusters
------------
# Create centroids with kmeans for 2 clusters
cluster_centers,_ = kmeans(fifa[scaled_features], 2)

# Assign cluster labels and print cluster centers
fifa['cluster_labels'], _ = vq(fifa[scaled_features], cluster_centers)
print(fifa.groupby('cluster_labels')[scaled_features].mean())

# Plot cluster centers to visualize clusters
fifa.groupby('cluster_labels')[scaled_features].mean().plot(legend=True, kind='bar')
plt.show()

# Get the name column of first 5 players in each cluster
for cluster in fifa['cluster_labels'].unique():
    print(cluster, fifa[fifa['cluster_labels'] == cluster]['name'].values[:5])
    

#                     scaled_pac  scaled_sho  scaled_pas  scaled_dri  scaled_def  scaled_phy
#     cluster_labels                                                                        
#     0                     6.68        5.43        8.46        8.51        2.50        8.34
#     1                     5.44        3.66        7.17        6.76        3.97        9.21
#     0 ['Cristiano Ronaldo' 'L. Messi' 'Neymar' 'L. Suárez' 'M. Neuer']
#     1 ['Sergio Ramos' 'G. Chiellini' 'D. Godín' 'Thiago Silva' 'M. Hummels']
    
#     That is correct! The data was sorted before you performed the clustering. 
#     Notice the top players in each cluster are representative of the overall characteristics of the cluster -
#     one of the clusters primarily represents attackers, whereas the other represents defenders.
#     Surprisingly, a top goalkeeper Manuel Neuer is seen in the attackers group, but he is known 
#     for going out of the box and participating in open play, which are reflected in his FIFA 18 attributes.
# --------------

