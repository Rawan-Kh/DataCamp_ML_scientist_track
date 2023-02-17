# You don't have timestamps for each data point, so it is not a time series.

--------------------
# Print the first 5 rows of data
print(data.head())

# Print the first 5 rows of data2
print(data2.head())

# Plot the time series in each dataset
fig, axs = plt.subplots(2, 1, figsize=(5, 10))
data.iloc[:1000].plot(y="data_values", ax=axs[0])
data2.iloc[:1000].plot(y="data_values", ax=axs[1])
plt.show()

#  What kind of data do you think each plot represents?
---------------

# Plot the time series in each dataset
fig, axs = plt.subplots(2, 1, figsize=(5, 10))
data.iloc[:1000].plot(x="time", y="data_values", ax=axs[0])
data2.iloc[:1000].plot(x="time", y="data_values", ax=axs[1])
plt.show()

#  As you can now see, each time series has a very different sampling frequency (the amount of time between samples). 
# The first is daily stock market data, and the second is an audio waveform.
-----------

from sklearn.svm import LinearSVC

# Construct data for the model
X = data[["petal length (cm)","petal width (cm)"]]
y = data[['target']]

# Fit the model
model = LinearSVC()
model.fit(X, y)

# You've successfully fit a classifier to predict flower type
------------

# Create input array
X_predict = targets[['petal length (cm)', 'petal width (cm)']]

# Predict with the model
predictions = model.predict(X_predict)
print(predictions)

# Visualize predictions and actual values
plt.scatter(X_predict['petal length (cm)'], X_predict['petal width (cm)'],
            c=predictions, cmap=plt.cm.coolwarm)
plt.title("Predicted class values")
plt.show()

# Note that the output of your predictions are all integers, representing that datapoint's predicted class
------------

from sklearn import linear_model

# Prepare input and output DataFrames
X = housing[["MedHouseVal",]]
y = housing[["AveRooms"]]

# Fit the model
model = linear_model.LinearRegression()
model.fit(X,y)
# In regression, the output of your model is a continuous array of numbers, not class identity.
------------
# Generate predictions with the model using those inputs
predictions = model.predict(new_inputs.reshape(-1, 1))

# Visualize the inputs and predicted values
plt.scatter(new_inputs, predictions, color='r', s=3)
plt.xlabel('inputs')
plt.ylabel('predictions')
plt.show()

# Here the red line shows the relationship that your model found. As the number of rooms grows, the median house value rises linearly.
------------

import librosa as lr
from glob import glob

# List all the wav files in the folder
audio_files = glob(data_dir + '/*.wav')

# Read in the first audio file, create the time array
audio, sfreq = lr.load(audio_files[0])
time = np.arange(0, len(audio)) / sfreq

# Plot audio over time
fig, ax = plt.subplots()
ax.plot(time, audio)
ax.set(xlabel='Time (s)', ylabel='Sound Amplitude')
plt.show()

# There are several seconds of heartbeat sounds in here, though note that most of this time is silence.
# A common procedure in machine learning is to separate the datapoints with lots of stuff happening from the ones that don't.
-----------

# Read in the data
data = pd.read_csv('prices.csv', index_col=0)

# Convert the index of the DataFrame to datetime
data.index = pd.to_datetime(data.index)
print(data.head())

# Loop through each column, plot its values over time
fig, ax = plt.subplots()
for column in data:
    data[column].plot(ax=ax, label=column)
ax.legend()
plt.show()

# --Note that each company's value is sometimes correlated with others, and sometimes not. Also note there are a lot of 'jumps' in there 
# - what effect do you think these jumps would have on a predictive model?
--------------

