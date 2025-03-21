from consistency_train import ConsistencyTraining
from toy_dataset import Circle2DDataset
import torch
import torch.nn as nn

class ConsistencyModel2D(nn.Module):
    def __init__(self, sigma_data, input_dim=2, hidden_dim=256, output_dim=2, t_range=1, epsilon=0.02,
                 num_frequencies=6):
        super().__init__()

        # Time Embedding: Sinusoidal time embedding (multiple frequencies)
        self.num_frequencies = num_frequencies
        self.t_range = t_range
        self.epsilon = epsilon

        # Define the frequencies used for time embedding
        self.frequencies = torch.logspace(0., num_frequencies - 1, num_frequencies) * (2 * torch.pi / t_range)

        # Define the network
        self.net1 = nn.Sequential(
            nn.LayerNorm(input_dim+ 2 * num_frequencies+1),
            nn.Linear(input_dim+ 2 * num_frequencies+1, hidden_dim),  # +2 * num_frequencies for time embedding
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.net2 = nn.Sequential(
            nn.LayerNorm(input_dim+1),
            nn.Linear(input_dim+1, hidden_dim),  # +2 * num_frequencies for time embedding
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.net3 = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, output_dim))

        # Define the skip and output coefficient schedulers
        self.skip_coef_scheduler = lambda t: sigma_data ** 2 / (
                    torch.clamp((t - epsilon) ** 2, min=0.) + sigma_data ** 2)
        self.out_coef_scheduler = lambda t: sigma_data * torch.clamp((t - epsilon), min=0.) / torch.sqrt(
            sigma_data ** 2 + t ** 2)
        self.sigma_data = sigma_data



    def time_embedding(self, t):
        # Time Embedding using Sinusoidal Functions
        emb = []
        for freq in self.frequencies:
            emb.append(torch.sin(freq * t))
            emb.append(torch.cos(freq * t))
        return torch.cat(emb, dim=-1)  # Concatenate sin/cos pairs

    def forward(self, x_in, t):

        x = torch.cat([x_in, torch.linalg.norm(x_in, dim=-1).unsqueeze(-1)], dim=-1)
        # Get the time embedding
        t_emb = self.time_embedding(t)

        # Concatenate the time embedding with the input
        x_t = torch.cat([x, t_emb], dim=-1)

        output1 = self.net1(x_t)
        output2 = self.net2(x)
        output = self.net3(torch.cat([output1, output2], dim=-1))
        # Compute skip coefs
        skip_coefs = self.skip_coef_scheduler(t)
        out_coefs = self.out_coef_scheduler(t)

        # Calculate barycentric output
        bary_output = skip_coefs * x_in + out_coefs * output
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
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.grid()
    plt.show()


# Hyperparameters
lr = 1e-2
num_epochs = 10000
device =  "cuda" if torch.cuda.is_available() else "cpu"
epsilon=0.1
T=10
b = 64

from sklearn import datasets

#dataset = torch.from_numpy(datasets.make_moons(n_samples=1000*64, shuffle=True, noise=None, random_state=None)[0]).to(
#    device, torch.float32)
dataset = Circle2DDataset(1000*64, torch.Tensor([0,0]), 8)


sigma_data = dataset.data.std()
print("sigma", sigma_data.item())
# Schedules and functions
def step_schedule(k):
    return min(10 + 10*k, 10000)  # Increases steps


def ema_decay_schedule(k):  # Fixed decay
    return 0.1


distance_fn = lambda pred, target: torch.sqrt(torch.mean((pred - target) ** 2))  # MSE loss
weight_fn = lambda t: 1/torch.clamp(t,1)**2  # Decreasing weight

model = ConsistencyModel2D(t_range=T, epsilon=epsilon,sigma_data=sigma_data).to(device)
ct = ConsistencyTraining(model,(2,), lr, step_schedule, ema_decay_schedule, distance_fn, weight_fn, T,
                         epsilon=epsilon, device=device)



import tqdm
for epoch in range(num_epochs):
    idx = torch.randint(0, len(dataset), (b,))
    x = dataset[idx].to(device)

    loss = ct.train_step(x, multiple_t=10)
    if epoch % 1 == 0:
        for N in [1]:
            generated = ct.generate_samples(num_samples=500, N=N)
            plot_data(dataset, generated, title=f"Epoch : {epoch} N : {N}")

        print(f"Epoch {epoch}, Loss: {loss:.4g}")