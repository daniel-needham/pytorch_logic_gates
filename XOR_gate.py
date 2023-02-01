import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

# data
input_data = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
output_data = torch.tensor([[0.], [1.], [1.], [0.]])

# num of epochs
num_epochs = 50000

# first layer
W_1 = nn.Parameter(torch.rand(2, 4))
b_1 = nn.Parameter(torch.rand(1))
# second layer
W_2 = nn.Parameter(torch.rand(4, 1))
b_2 = nn.Parameter(torch.rand(1))

# loss function
loss_fn = nn.MSELoss()

# optimizer
optimizer = optim.SGD([W_1, W_2, b_1, b_2], lr=0.01)


# predict function (
def predict(x, W_1, W_2, b_1, b_2):
    output = torch.relu(torch.mm(x, W_1) + b_1)
    output = torch.mm(output, W_2) + b_2
    return output


# Training loop
for epoch in range(num_epochs):
    for i in range(input_data.size(0)):
        x = input_data[i].unsqueeze(0)
        y = output_data[i].unsqueeze(0)

        # Clear gradients
        optimizer.zero_grad()
        # Predict outputs
        y_hat = predict(x, W_1, W_2, b_1, b_2)
        # Calculate loss
        loss = loss_fn(y_hat, y)
        # Calculate gradients
        loss.backward()
        # Update weights
        optimizer.step()

    if epoch % 1000 == 0:
        print(f"Testing network @ epoch {epoch}")
        for i in range(input_data.size(0)):
            # Make a prediction
            x = input_data[i].unsqueeze(0)
            y = output_data[i].unsqueeze(0)
            y_hat = predict(x, W_1, W_2, b_1, b_2)
            # Print result
            print("Input:{} Target: {} Predicted:[{}] Error:[{}]".format(
                x.data.numpy(),
                y.data.numpy(),
                np.round(y_hat.data.numpy(), 4),
                np.round(y.data.numpy() - y_hat.data.numpy(), 4)
            ))
