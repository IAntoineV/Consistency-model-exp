from consistency_train import ConsistencyTraining
import torch
import torch.nn as nn


# Define the Consistency Model
class ConsistencyModel2D(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=2, t_range=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for time embedding
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.t_range = t_range
        self.skip_coef_scheduler = lambda t: 1 - t / t_range

    def forward(self, x, t):
        # Concatenate time embedding with input
        x_t = torch.cat([x, t], dim=-1)
        output = self.net(x_t)
        skip_coefs = self.skip_coef_scheduler(t)
        bary_output = skip_coefs * x + (1 - skip_coefs) * output
        return bary_output


import matplotlib.pyplot as plt


# Plot function
def plot_data(dataset, generated=None, title="Dataset"):
    plt.figure(figsize=(6, 6))
    plt.scatter(dataset[:, 0].cpu(), dataset[:, 1].cpu(), c="blue", label="Dataset")
    if generated is not None:
        plt.scatter(generated[:, 0], generated[:, 1], c="red", label="Generated")
    plt.title(title)
    plt.legend()
    plt.xlim(-1.5, 2.5)
    plt.ylim(-1, 1.5)
    plt.grid()
    plt.show()


# Hyperparameters
lr = 1e-2
num_epochs = 10000
device =  "cpu"

from sklearn import datasets

dataset = torch.from_numpy(datasets.make_moons(n_samples=1000, shuffle=True, noise=None, random_state=None)[0]).to(
    device, torch.float32)


# Schedules and functions
def step_schedule(k):
    return max(1000, 1 + k // 50)  # Increases every 50 steps the number of discretisation points between 0 and 1


def ema_decay_schedule(k):  # Linear augmentation of target model parameters.
    return min(0.9, 0.2 + k / 10000)


distance_fn = lambda pred, target: torch.mean((pred - target) ** 2)  # MSE loss
weight_fn = lambda t: 1.0  # Constant weight

# Initialize components
model = ConsistencyModel2D(t_range=1).to(device)
ct = ConsistencyTraining(model,(2,), lr, step_schedule, ema_decay_schedule, distance_fn, weight_fn, device=device)

b = 64

import tqdm
# Training loop
for epoch in range(num_epochs):
    # Sample a batch from the dataset
    idx = torch.randint(0, len(dataset), (b,))
    x = dataset[idx].to(device)

    loss = ct.train_step(x)

    # Plot generated data every 100 epochs
    if epoch % 100 == 0:
        for N in [1, 10, 100, 1000]:
            generated = ct.generate_samples(num_samples=1000, N=N)
            plot_data(dataset, generated, title=f"Epoch : {epoch} N : {N}")

        print(f"Epoch {epoch}, Loss: {loss:.4g}")