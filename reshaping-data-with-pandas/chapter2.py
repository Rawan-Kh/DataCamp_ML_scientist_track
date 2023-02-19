# Melt books_gothic using the title column as identifier 
gothic_melted = books_gothic.melt(id_vars='title')

# Print gothic_melted
print(gothic_melted)
--------
# Melt books_gothic using the title, authors, and publisher columns as identifier
gothic_melted_new = books_gothic.melt(id_vars=['title','authors','publisher'])

# Print gothic_melted_new
print(gothic_melted_new)
# As a data scientist, there will be many situations in which you will need to reshape your data from wide to long format. 
# .melt() is a very flexible way to achieve that goal!
--------
# Melt publisher column using title and authors as identifiers
publisher_melted = books_gothic.melt(id_vars=['title','authors'], 
                                     value_vars='publisher')

# Print publisher_melted
print(publisher_melted)
-----------
# Melt rating and rating_count columns using the title as identifier
rating_melted = books_gothic.melt(id_vars='title', 
                                     value_vars=['rating','rating_count'])

# Print rating_melted
print(rating_melted)
-----------
# Melt rating and rating_count columns using title and authors as identifier
books_melted = books_gothic.melt(id_vars=['title','authors'], 
                                     value_vars=['rating','rating_count'])

# Print books_melted
print(books_melted)

#  You can see how specifying different values to melt gives you flexibility in the process of reshaping. 
# This is helpful when you don't want to include all variables in a graph, for example
----------

# Melt the rating and rating_count using title, authors and publisher as identifiers
books_ratings = books_gothic.melt(id_vars=['title','authors','publisher'],
                                     value_vars=['rating','rating_count'])

# Print books_ratings
print(books_ratings)
---------
# Assign the name feature to the new variable column
books_ratings = books_gothic.melt(id_vars=['title', 'authors', 'publisher'], 
                                  value_vars=['rating', 'rating_count'], 
                                  var_name='feature')

# Print books_ratings
print(books_ratings)

------
# Assign the name number to the new column containing the values
books_ratings = books_gothic.melt(id_vars=['title', 'authors', 'publisher'], 
                                  value_vars=['rating', 'rating_count'], 
                                  var_name='feature', 
                                 value_name='number')

# Print books_ratings
print(books_ratings)

# Using clear names for the new value and variable columns allows you to have a customized DataFrame. This feature helps data scientists share clean and useful data
----------------





