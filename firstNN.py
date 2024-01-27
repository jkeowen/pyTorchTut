import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backend.mps.is_available()
    else "cpu"
)

x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z ,y)
print(loss.backward())
# print(w.grad)
# print(b.grad)





# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(28*28, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10),
#         )

#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits



# model = NeuralNetwork().to(device)

# X = torch.rand(1, 28, 28, device = device)
# logits = model(X)
# pred_probab = nn.Softmax(dim=1)(logits)
# y_pred = pred_probab.argmax(1)
# # print(f"Predicted class: {y_pred}")

# input_image = torch.rand(3, 28, 28)
# # print(input_image.size())

# flatten = nn.Flatten()
# flat_image = flatten(input_image)
# # print(flat_image.size())

# layer1 = nn.Linear(in_features=28*28, out_features=20)
# hidden1 = layer1(flat_image)
# # print(hidden1)

# print(f"Before ReLU: {hidden1}\n\n")
# hidden1 = nn.ReLU()(hidden1)
# print(f"After ReLU: {hidden1}")






# input_matrix = torch.tensor([
#     [-1.0, 5.0],
#     [2.0, -5.0],
#     [15.0, -123.0]
# ])
# relu = nn.ReLU()
# output_matrix = relu(input_matrix)
# linear_layer = nn.Linear(2,5)
# lin_output = linear_layer(input_matrix)
# print(lin_output)

# linear_layer = nn.Linear(in_features=2, out_features=1, bias=True)

# # Set weights and biases to known values for illustration
# linear_layer.weight.data = torch.tensor([[0.5, -0.3]])
# linear_layer.bias.data = torch.tensor([0.2])

# # Sample input matrix
# input_matrix = torch.tensor([
#     [1.0, 2.0],
#     [3.0, 4.0],
#     [5.0, 6.0]
# ], dtype=torch.float32, requires_grad=True)

# # Forward pass
# output_matrix = linear_layer(input_matrix)
# predicted_values = output_matrix.squeeze()  # Removing singleton dimensions for simplicity

# # Target values (ground truth)
# target_values = torch.tensor([3.0, 5.0, 7.0], dtype=torch.float32)

# # Loss computation (squared error)
# loss = torch.sum((predicted_values - target_values) ** 2)

# # Backward pass (manually computing gradients)
# loss.backward()

# # Accessing gradients
# weight_gradients = linear_layer.weight.grad
# bias_gradient = linear_layer.bias.grad
# input_gradients = input_matrix.grad

# # Print results
# print("Input Matrix:")
# print(input_matrix)
# print("\nWeight Matrix:")
# print(linear_layer.weight)
# print("\nBias Vector:")
# print(linear_layer.bias)
# print("\nOutput Matrix:")
# print(output_matrix)
# print("\nPredicted Values:")
# print(predicted_values)
# print("\nTarget Values:")
# print(target_values)
# print("\nLoss:")
# print(loss.item())
# print("\nGradient of Weight Matrix:")
# print(weight_gradients)
# print("\nGradient of Bias Vector:")
# print(bias_gradient)
# print("\nGradient of Input Matrix:")
# print(input_gradients)