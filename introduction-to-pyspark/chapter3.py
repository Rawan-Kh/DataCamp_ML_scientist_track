# Spark is just a platform that implements the same algorithms that can be found elsewhere.
---------

# Rename year column
planes = planes.withColumnRenamed('year','plane_year')

# Join the DataFrames
model_data = flights.join(planes, on='tailnum', how="leftouter")
-----------
# Spark needs numeric values (doubles or integers) to do machine learning.
--------------

# Cast the columns to integers
model_data = model_data.withColumn("arr_delay",model_data.arr_delay.cast("integer"))
model_data = model_data.withColumn("air_time", model_data.air_time.cast("integer"))
model_data = model_data.withColumn("month", model_data.month.cast("integer"))
model_data = model_data.withColumn("plane_year", model_data.plane_year.cast("integer"))
# You're a pro at converting columns!
--------

# Create the column plane_age
model_data = model_data.withColumn("plane_age", model_data.year- model_data.plane_year)
# Now you have one more variable to include in your model.
------------

# Create is_late
model_data = model_data.withColumn("is_late", model_data.arr_delay > 0)

# Convert to an integer
model_data = model_data.withColumn("label", model_data.is_late.cast("integer"))

# Remove missing values
model_data = model_data.filter("arr_delay is not NULL and dep_delay is not NULL and air_time is not NULL and plane_year is not NULL")
# Now you've defined the column that you're going to use as the outcome in your model.
-----------

# Remember that Spark can only model numeric features.
-------------

# Create a StringIndexer
carr_indexer = StringIndexer(inputCol="carrier" , outputCol="carrier_index")

# Create a OneHotEncoder
carr_encoder = OneHotEncoder(inputCol="carrier_index" , outputCol="carrier_fact")
# You're ready to include this column in your model now!
-----------
# Create a StringIndexer
dest_indexer = StringIndexer(inputCol="dest" , outputCol="dest_index")

# Create a OneHotEncoder
dest_encoder = OneHotEncoder(inputCol="dest_index" , outputCol="dest_fact")

-----------
# Make a VectorAssembler
vec_assembler = VectorAssembler(inputCols=["month", "air_time", "carrier_fact", "dest_fact", "plane_age"], outputCol="features")
# Your data is all assembled now.
-----------

# Import Pipeline
from pyspark.ml import Pipeline

# Make the pipeline
flights_pipe = Pipeline(stages=[dest_indexer, dest_encoder, carr_indexer, carr_encoder, vec_assembler])
# You've made a fully reproducible machine learning pipeline!
---------

# Fit and transform the data
piped_data = flights_pipe.fit(model_data).transform(model_data)
---------
# Split the data into training and test sets
training, test = piped_data.randomSplit([.6, .4])
#  Now you're ready to start fitting a model!
---------




