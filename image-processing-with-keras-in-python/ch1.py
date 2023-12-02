# Import the image from the file bricks.png into data.
# Display the image in data on the screen.
# Import matplotlib
import matplotlib.pyplot as plt

# Load the image
data = plt.imread('bricks.png')

# Display the image
plt.imshow(data)
plt.show()

----------
# Modify the bricks image to replace the top left corner of the image (10 by 10 pixels) into a red square.
# Visualize the resulting image.
# Set the red channel in this part of the image to 1
data[:10, :10, 0] = 1

# Set the green channel in this part of the image to 0
data[:10, :10, 1] = 0

# Set the blue channel in this part of the image to 0
data[:10, :10, 2] = 0

# Visualize the result
plt.imshow(data)
plt.show()
# You now know how to manipulate images. By the way, if you set both the green and red channels to 1, that part of the image would be yellow.
---------

