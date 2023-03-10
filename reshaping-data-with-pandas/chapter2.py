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
# Reshape wide to long using title as index and version as new name, and extracting isbn prefix 
isbn_long = pd.wide_to_long(golden_age, 
                            stubnames='isbn', 
                            i='title', 
                            j='version')

# Print isbn_long
print(isbn_long)
----------
# Reshape wide to long using title and authors as index and version as new name, and prefix as wide column prefix
prefix_long = pd.wide_to_long(golden_age, 
                            stubnames='prefix', 
                            i=['title','authors'], 
                            j='version')

# Print prefix_long
print(prefix_long)
-----------
# Reshape wide to long using title and authors as index and version as new name, and prefix and isbn as wide column prefixes 
all_long = pd.wide_to_long(golden_age, 
                           stubnames=['isbn', 'prefix'], 
                           i=['title', 'authors'], 
                           j='version')

# Print all_long
print(all_long)

-------------
# Reshape using author and title as index, code as new name and getting the prefix language and publisher
the_code_long = pd.wide_to_long(books_brown, 
                           stubnames=['language', 'publisher'], 
                           i=['author', 'title'], 
                           j='code')

# Print the_code_long
print(the_code_long)
--------
# Specify underscore as the character that separates the variable names
the_code_long = pd.wide_to_long(books_brown, 
                                stubnames=['language', 'publisher'], 
                                i=['author', 'title'], 
                                j='code', sep='_')

# Print the_code_long
print(the_code_long)
----------
# Specify that wide columns have a suffix containing words
the_code_long = pd.wide_to_long(books_brown, 
                                stubnames=['language', 'publisher'], 
                                i=['author', 'title'], 
                                j='code', 
                                sep='_', 
                                suffix='\w+')

# Print the_code_long
print(the_code_long)

-----------
# Modify books_hunger by resetting the index without dropping it
books_hunger.reset_index(drop=False, inplace=True)

# Reshape using title and language as index, feature as new name, publication and page as prefix separated by space and ending in a word
publication_features = pd.wide_to_long(books_hunger, 
                                stubnames=['publication', 'page'], 
                                i=['title', 'language'], 
                                j='feature', 
                                sep=' ', 
                                suffix='\w+')

# Print publication_features
print(publication_features)

----------
# Split the index of books_dys by the hyphen 
books_dys.index = books_dys.index.str.split('-')

# Print books_dys
print(books_dys)
--------
# Get the first element after splitting the index of books_dys
books_dys.index = books_dys.index.str.split('-').str.get(0)

# Print books_dys
print(books_dys)

----------
# Split by the hyphen the index of books_dys
books_dys.index = books_dys.index.str.split('-').str.get(0)

# Concatenate the index with the list author_list separated by a hyphen
books_dys.index = books_dys.index.str.cat(author_list, sep='-')

# Print books_dys
print(books_dys)
#  The str attribute of pandas makes it easy to work with strings, even when they are in the index,
# which is something that would be very difficult to handle otherwise!
--------

# Concatenate the title and subtitle separated by "and" surrounded by spaces
hp_books['full_title'] = hp_books['title'].str.cat( hp_books['subtitle'], sep =' and ') 

# Print hp_books
print(hp_books)
----------
# Concatenate the title and subtitle separated by "and" surrounded by spaces
hp_books['full_title'] = hp_books['title'].str.cat(hp_books['subtitle'], sep =" and ") 

# Split the authors into writer and illustrator columns
hp_books[['writer', 'illustrator']] = hp_books['authors'].str.split('/', expand=True) 

# Print hp_books
print(hp_books)

----------
# Concatenate the title and subtitle separated by "and" surrounded by spaces
hp_books['full_title'] = hp_books['title'].str.cat(hp_books['subtitle'], sep =" and ") 

# Split the authors into writer and illustrator columns
hp_books[['writer', 'illustrator']] = hp_books['authors'].str.split('/', expand=True)

# Melt goodreads and amazon columns into a single column
hp_melt = hp_books.melt(id_vars=['full_title','writer'], 
                        var_name='source', 
                        value_vars=['goodreads','amazon'], 
                        value_name='rating')

# Print hp_melt
print(hp_melt)
# In this exercise, you managed to create new features by splitting and joining columns together. 
# This can help you in your future analysis; particularly, when you are tidying your dataset
----------

# Split main_title by a colon and assign it to two columns named title and subtitle 
books_sh[['title', 'subtitle']] = books_sh['main_title'].str.split(':', expand=True)

# Split version by a space and assign the second element to the column named volume 
books_sh['volume'] = books_sh['version'].str.split(' ').str.get(1)

# Drop the main_title and version columns modifying books_sh
books_sh.drop(['main_title', 'version'], axis=1, inplace=True)

# Reshape using title, subtitle and volume as index, name feature the new variable from columns starting with number, separated by undescore and ending in words 
sh_long = pd.wide_to_long(books_sh, stubnames='number', i=['title', 'subtitle', 'volume'],j='feature', sep='_', suffix='\w+')

# Print sh_long 
print(sh_long)

# Splitting string columns into new columns creates cleaner features to prepare for reshaping. Having a clean and understandable dataset is one of the first tasks before performing any analysis.
-------------





