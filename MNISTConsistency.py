import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from consistency_train import ConsistencyTraining


# Define a simple CNN-based Consistency Model for MNIST
class ConsistencyModelMNIST(nn.Module):
    def __init__(self, t_range=1):
        super().__init__()
        self.t_range = t_range
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
        )
        self.skip_coef_scheduler = lambda t: 1 - t / t_range

    def forward(self, x, t):
        output = self.conv(x)
        skip_coefs = self.skip_coef_scheduler(t)
        return skip_coefs * x + (1 - skip_coefs) * output


# Training setup
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 1e-3
batch_size = 128
num_epochs = 10

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Define schedules and loss functions
def step_schedule(k):
    return max(100, 1 + k // 50)


def ema_decay_schedule(k):
    return min(0.9, 0.2 + k / 5000)


distance_fn = lambda pred, target: torch.mean((pred - target) ** 2)
weight_fn = lambda t: 1.0

# Initialize model and trainer
model = ConsistencyModelMNIST().to(device)
ct = ConsistencyTraining(model, (1, 28, 28), lr, step_schedule, ema_decay_schedule, distance_fn, weight_fn,
                         device=device)


import matplotlib.pyplot as plt
def show_images(images, title):
    fig, axes = plt.subplots(4, 4, figsize=(5, 5))
    for ax,img in zip(axes.flat, images):
        ax.imshow(img.squeeze(), cmap='gray')
        ax.axis("off")
    fig.suptitle(title)
    plt.show()


# Training loop
for epoch in range(num_epochs):
    for x, _ in train_loader:
        x = x.to(device)
        loss = ct.train_step(x)
    print(f"Epoch {epoch}, Loss: {loss:.4g}")
    # Generate samples
    for N in [1, 5, 10, 100]:
        samples = ct.generate_samples(num_samples=16, N=10)


        titles = f"epoch : {epoch}, N : {N}"
        show_images(samples, titles)


