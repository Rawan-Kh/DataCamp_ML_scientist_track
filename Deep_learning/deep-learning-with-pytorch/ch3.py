# Create 10 random images of shape (1, 28, 28)
images = torch.rand(10, 1, 28, 28)

# Build 6 conv. filters
conv_filters = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3,  stride=1, padding=1)

# Convolve the image with the filters 
output_feature = conv_filters(images)
print(output_feature.shape)
# You just made the first step in building CNNs.
# torch.Size([10, 6, 28, 28])
----------
# Create 10 random images
image = torch.rand(10, 1, 28, 28)

# Create 6 filters
filters = torch.rand(6, 1, 3, 3)

# Convolve the image with the filters
output_feature = F.conv2d(image, filters, stride=1, padding=1)
print(output_feature.shape)
# Awesome, now you know how to build convolutional layers using functional package.
---------

# Build a pooling operator with size `2`.
max_pooling = torch.nn.MaxPool2d(2)

# Apply the pooling operator
output_feature =  max_pooling(im)

# Use pooling operator in the image
output_feature_F = F.max_pool2d(im, 2)


# print the results of both cases
print(output_feature)
print(output_feature_F)
# You see how the results are the same. In general, torch.nn and torch.nn.functional are equivalent and the choice for one of them is mostly a personal choice.
--------
# Build a pooling operator with size `2`.
avg_pooling = torch.nn.AvgPool2d(2)

# Apply the pooling operator
output_feature = avg_pooling(im)

# Use pooling operator in the image
output_feature_F =  F.avg_pool2d(im, 2)

# print the results of both cases
print(output_feature)
print(output_feature_F)
------------
# Deduct the first size of the weights for the fully connected layers. Images start with shape (1, 28, 28) and two pooling operators (each halving the size of the image) are performed. What is the size of the image fed to the input layer (heigh * width * number_of_channels)?
# In line 16, number_of_channels is the same as the number of channels in self.conv2.
# MNIST images are black and white, so they contain one channel.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Instantiate two convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1)
        
        # Instantiate the ReLU nonlinearity
        self.relu = nn.ReLU()
        
        # Instantiate a max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Instantiate a fully connected layer
        self.fc = nn.Linear(7 * 7 * 10, 10)
 ----------
# Hint
# Your image started with shape (28, 28) and on it were performed 2 pooling layers, each halving the size. Second conv layer had 10 channels. What should be the second argument of view?
# Please write the second argument of view() as (height * width * num_channels) instead of (dimension).
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
		
        # Instantiate the ReLU nonlinearity
        self.relu = nn.ReLU()
        
        # Instantiate two convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1)
        
        # Instantiate a max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Instantiate a fully connected layer
        self.fc = nn.Linear(7 * 7 * 10, 10)

    def forward(self, x):
  
        # Apply conv followd by relu, then in next line pool
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        # Apply conv followd by relu, then in next line pool
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # Prepare the image for the fully connected layer
        x = x.view(-1, 7 * 7 * 10)

        # Apply the fully connected layer and return the result
        return self.fc(x)

------------
for i, data in enumerate(train_loader, 0):
    inputs, labels = data
    optimizer.zero_grad()

    # Compute the forward pass
    outputs = net(inputs)
        
    # Compute the loss function
    loss =  criterion(outputs, labels)
        
    # Compute the gradients
    loss.backward()
    
    # Update the weights
    optimizer.step()
------------

# Iterate over the data in the test_loader
for i, data in enumerate(test_loader):

    # Get the image and label from data
    image, label = data

    # Make a forward pass in the net with your image
    output = net(image)

    # Argmax the results of the net
    _, predicted = torch.max(output.data, 1)
    if predicted == label:
        print("Yipes, your net made the right prediction " + str(predicted))
    else:
        print("Your net prediction was " + str(predicted) + ", but the correct label is: " + str(label))
-------------
