# Import torch
import torch

# Create random tensor of size 3 by 3
your_first_tensor = torch.rand(3, 3)

# Calculate the shape of the tensor
tensor_size = your_first_tensor.shape

# Print the values of the tensor and its shape
print(your_first_tensor)
print(tensor_size)

#  output:
#     tensor([[0.0707, 0.1310, 0.7484],
#             [0.2613, 0.6283, 0.0882],
#             [0.2295, 0.2540, 0.2743]])
#     torch.Size([3, 3])
-------
# Create a matrix of ones with shape 3 by 3
tensor_of_ones = torch.ones(3, 3)

# Create an identity matrix with shape 3 by 3
identity_tensor = torch.eye(3)

# Do a matrix multiplication of tensor_of_ones with identity_tensor
matrices_multiplied = torch.matmul(tensor_of_ones, identity_tensor)
print(matrices_multiplied)

# Do an element-wise multiplication of tensor_of_ones with identity_tensor
element_multiplication =  torch.mul(tensor_of_ones, identity_tensor)
print(element_multiplication)

# output:
#     tensor([[1., 1., 1.],
#             [1., 1., 1.],
#             [1., 1., 1.]])
#     tensor([[1., 0., 0.],
#             [0., 1., 0.],
#             [0., 0., 1.]])
# matrices_multiplied is same as tensor_of_ones (because identity matrix is the neutral element in matrix multiplication, the product of any matrix multiplied with it gives the original matrix), while element_multiplication is same as identity_tensor
------
# Initialize tensors x, y and z
x = torch.rand(1000, 1000)
y = torch.rand(1000, 1000)
z = torch.rand(1000, 1000)

# Multiply x with y
q = torch.mul(x, y)

# Multiply elementwise z with q
f = torch.matmul(z, q)

mean_f = torch.mean(f)
print(mean_f)
# You just built a nice computational graph containing 5'000'001 values. In the next lesson, you are going to compute the gradients of this graph.
-------
# you know how to compute derivatives! While PyTorch computes derivatives for you, mastering them will make you a much better deep learning practitioner and that knowledge will guide you in training neural networks better.
-----------
# Initialize x, y and z to values 4, -3 and 5
x = torch.tensor(4., requires_grad=True)
y = torch.tensor(-3., requires_grad=True)
z = torch.tensor(5., requires_grad=True)

# Set q to sum of x and y, set f to product of q with z
q = x + y
f = q * z

# Compute the derivatives
f.backward()

# Print the gradients
print("Gradient of x is: " + str(x.grad))
print("Gradient of y is: " + str(y.grad))
print("Gradient of z is: " + str(z.grad))
# output:
#     Gradient of x is: tensor(5.)
#     Gradient of y is: tensor(5.)
#     Gradient of z is: tensor(1.)
# the results are the same as when you calculated them by hand!
----------
# Multiply tensors x and y
q = torch.matmul(x, y)

# Elementwise multiply tensors z with q
f = torch.matmul(z,q)

mean_f = torch.mean(f)

# Calculate the gradients
mean_f.backward()
# calculating gradients is as easy as calculating derivatives in PyTorch. Obviously, if the tensors are very large (billions of values) then the calculation might take some time.
---------

# Initialize the weights of the neural network
weight_1 = torch.rand(784, 200)
weight_2 = torch.rand(200, 10)

# Multiply input_layer with weight_1
hidden_1 = torch.matmul(input_layer, weight_1)

# Multiply hidden_1 with weight_2
output_layer = torch.matmul(hidden_1,weight_2)
print(output_layer)
# neural networks are just matrix (tensor) multiplication. This is the reason why we have put so much emphasis on matrices and tensors!
# output:
#     tensor([17037.8711, 18565.3418, 18923.9805, 18905.7012, 17854.7656, 16902.4844,19731.2969, 17921.3281, 18924.6992, 19442.9473])
-----
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Instantiate all 2 linear layers  
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
      
        # Use the instantiated layers and return x
        x = self.fc1(x)
        x = self.fc2(x)
        return x
# You just built your first PyTorch artificial neural network. You are going to build many more during this course.
-----------


