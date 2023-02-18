# Understanding the difference between long and wide formats is a useful skill. It will help you decide the appropriate format of your data to conduct your analysis
---------------
# Set name as index
fifa_transpose = fifa_players.set_index('name')

# Print fifa_transpose
print(fifa_transpose)

--------
# Modify the DataFrame to keep only height and weight columns
fifa_transpose = fifa_players.set_index('name')[['height', 'weight']]

# Print fifa_transpose
print(fifa_transpose)

--------
# Change the DataFrame so rows become columns and vice versa
fifa_transpose = fifa_players.set_index('name')[['height', 'weight']].transpose()

# Print fifa_transpose
print(fifa_transpose)
# Even though transposing DataFrames can be suitable in some particular situations, 
# it is not good enough to process a dataset for analysis or graphing. Fortunately, we'll see more useful functions in the next lessons.
----------

# Pivot fifa_players to get overall scores indexed by name and identified by movement
fifa_overall = fifa_players.pivot(index='name', columns='movement', values='overall')

# Print fifa_overall
print(fifa_overall)

--------
# Pivot fifa_players to get attacking scores indexed by name and identified by movement
fifa_attacking = fifa_players.pivot(index='name', columns='movement', values='attacking')

# Print fifa_attacking
print(fifa_attacking)
----------
# Use the pivot method to get overall scores indexed by movement and identified by name
fifa_names = fifa_players.pivot(index='movement', columns='name', values='overall')

# Print fifa_names
print(fifa_names)

# The .pivot() method allows us to reshape our DataFrame in different ways to discover relationships between two columns.
# Notice that after pivoting, it is easier to see how the score varies between the two players in fifa_players.
----------

# Pivot fifa_players to get overall and attacking scores indexed by name and identified by movement
fifa_over_attack = fifa_players.pivot(index='name', columns='movement', values=['overall', 'attacking'])
# Print fifa_over_attack
print(fifa_over_attack)
--------
# Use pivot method to get all the scores index by name and identified by movement
fifa_all = fifa_players.pivot(index='name', columns='movement', values='fifa_players')

# Print fifa_over_attack
print(fifa_all)
---------
# Use pivot method to get all the scores index by name and identified by movement
fifa_all = fifa_players.pivot(index='name', columns='movement')

# Print fifa_over_attack
print(fifa_all)

# Notice that overall and attacking scores are different for the same player. The resulting pivoted DataFrame has a hierarchical column index. 
# Omitting the values parameter to pivot all columns works in most cases, but you'll now see some situations where this is not suitable.
----------
# Drop the fifth row to delete all repeated rows
fifa_no_rep = fifa_players.drop(4, axis=0)

# Pivot fifa players to get all scores by name and movement
fifa_pivot = fifa_no_rep.pivot(index='name', columns='movement') 

# Print fifa_pivot
print(fifa_pivot)  

# When the .pivot() method finds two rows with the same index and column, but different values for the values,
# it doesn't know how to handle it. Fortunately, you'll learn a way to solve this next!
------------
# Discard the fifth row to delete all repeated rows
fifa_drop = fifa_players.drop(4,axis=0)

# Use pivot method to get all scores by name and movement
fifa_pivot = fifa_drop.pivot(index='name', columns='movement') 

# Print fifa_pivot
print(fifa_pivot)  

# Use pivot table to get all scores by name and movement
fifa_pivot_table = fifa_players.pivot_table(index='name', columns='movement',aggfunc='mean')

# Print fifa_pivot_table
print(fifa_pivot_table)

# The .pivot_table() method doesn't have the same limitations that .pivot() does. You can see in the resulting DataFrame that the repeated values are aggregated.
# Now, you see the mean in that case. Pivot table is useful when you need a table with summary statistics of a larger DataFrame.
---------------
# Use pivot table to display mean age of players by club and nationality 
mean_age_fifa = fifa_players.pivot_table(index='nationality', columns='club',values='age',aggfunc='mean')

# Print mean_age_fifa
print(mean_age_fifa)

-----------
# Use pivot table to display max height of any player by club and nationality
tall_players_fifa = fifa_players.pivot_table(index='nationality', columns='club',values='height',aggfunc='max')

# Print tall_players_fifa
print(tall_players_fifa)
-------------
# Use pivot table to show the count of players by club and nationality and the total count
players_country = fifa_players.pivot_table(index='nationality', columns='club',values='name',aggfunc='count',margins='count')

# Print players_country
print(players_country)
#  Notice how defining aggregation functions in .pivot_table() allows us to discover trends in the data. 
# Now, for example, you know that the tallest player in Real Madrid is from France and the oldest from Croatia!
-----------

# Set the argument to get the maximum for each row and column
fifa_mean = fifa_players.pivot_table(index=['nationality', 'club'], 
                                     columns='year', 
                                     aggfunc='max', 
                                     margins='max')

# Print fifa_mean
print(fifa_mean)
# You just finished chapter 1! Notice how defining two columns as an index creates hierarchical indexes, 
# and how the pivot table helps us discover patterns in the original DataFrame
-----------


