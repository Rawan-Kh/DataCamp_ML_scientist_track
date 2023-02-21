# Import LogisticRegression
from pyspark.ml.classification import LogisticRegression

# Create a LogisticRegression Estimator
lr = LogisticRegression()
# That's the first step to any modeling in PySpark
---------

# The cross validation error is an estimate of the model's error on the test set.

--------
# Import the evaluation submodule
import pyspark.ml.evaluation as evals

# Create a BinaryClassificationEvaluator
evaluator = evals.BinaryClassificationEvaluator(metricName="areaUnderROC")
# Now you can compare models using the metric output by your evaluator!
---------

# Import the tuning submodule
import pyspark.ml.tuning as tune

# Create the parameter grid
# grid = tune.ParamGridBuilder()

# Add the hyperparameter
grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))
grid = grid.addGrid(lr.elasticNetParam, [0, 1])

# Build the grid
grid = grid.build()
# hat's the last ingredient in your cross validation recipe!
----------
# Create the CrossValidator
cv = tune.CrossValidator(estimator=lr,
               estimatorParamMaps=grid,
               evaluator=evaluator
               )
-----------
# Call lr.fit()
best_lr = lr.fit(training)

# Print best_lr
print(best_lr)

# You fit your first Spark model!
---------
# The AUC gets better when it's bigger.
# An AUC of one represents a model that always perfectly classifies observations.
----------
# Use the model to predict the test set
test_results = best_lr.transform(test)

# Evaluate the predictions
print(evaluator.evaluate(test_results))

#  What do you think of the AUC? Your model isn't half bad! 
# You went from knowing nothing about Spark to doing advanced machine learning.
# Great job on making it to the end of the course! The next steps are learning how to create large scale Spark clusters 
# and manage and submit jobs so that you can use models in the real world. 
# Check out some of the other DataCamp courses that use Spark!
# And remember, Spark is still being actively developed, so there's new features coming all the time!
---------


