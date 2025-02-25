import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --------------------------------------------------
# 1) Generate the data
# --------------------------------------------------
def target_function(x_np):
    """
    Same piecewise function you used in A2:
      - z=1  for x < 1
      - z=1-(x-1)  for 1 <= x < 2
      - z=0  for x >= 2
    """
    # For convenience, use np.where:
    return np.where(
        x_np < 1,
        1,
        np.where(x_np < 2, 1 - (x_np - 1), 0)
    )

# Make some data points
x_np = np.linspace(0, 3, 100)
y_np = target_function(x_np)

# Convert them to torch tensors with shape (N,1)
# so that PyTorch sees each point as a row vector
x_torch = torch.from_numpy(x_np).float().unsqueeze(1)
y_torch = torch.from_numpy(y_np).float().unsqueeze(1)

# --------------------------------------------------
# 2) Define the model (w1*x + b1 -> ReLU -> w2*u + b2 -> ReLU)
# --------------------------------------------------
class SimpleTwoLayerNet(nn.Module):
    def __init__(self):
        super(SimpleTwoLayerNet, self).__init__()
        
        # We want exactly one hidden neuron (to match w1,b1 and w2,b2).
        # So we go from 1 input -> 1 hidden -> 1 output,
        # with ReLU in between layers.
        self.linear1 = nn.Linear(1, 1)  # w1, b1
        self.linear2 = nn.Linear(1, 1)  # w2, b2
        self.relu = nn.ReLU()
        
    def forward(self, x):
        v = self.linear1(x)     # v = w1*x + b1
        u = self.relu(v)        # u = ReLU(v)
        y = self.linear2(u)     # y = w2*u + b2
        z = self.relu(y)        # z = ReLU(y)
        return z

# Instantiate the model
model = SimpleTwoLayerNet()

# --------------------------------------------------
# 3) Define loss and optimizer
# --------------------------------------------------
criterion = nn.MSELoss()           # Mean-squared error
optimizer = optim.SGD(model.parameters(), lr=0.01)

# --------------------------------------------------
# 4) Training loop
# --------------------------------------------------
epochs = 1000
for epoch in range(epochs):
    # Zero out gradients from the previous step
    optimizer.zero_grad()
    
    # Forward pass
    y_pred = model(x_torch)
    
    # Compute loss
    loss = criterion(y_pred, y_torch)
    
    # Backward pass
    loss.backward()
    
    # Update parameters
    optimizer.step()
    
    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

# --------------------------------------------------
# 5) Evaluate / visualize
# --------------------------------------------------
# We already have x_torch sorted from 0..3 in steps,
# so we can directly feed it to the final model:
y_pred_torch = model(x_torch).detach().numpy()

plt.figure(figsize=(8,6))
plt.plot(x_np, y_np, label="Target Function", linewidth=2)
plt.plot(x_np, y_pred_torch, label="PyTorch Model Output", linestyle="--")
plt.legend()
plt.title("Function Approximation with a 2-Layer ReLU Network (PyTorch)")
plt.xlabel("x")
plt.ylabel("z")
plt.grid(True)
plt.show()
