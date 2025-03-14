import torch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


class DDPM(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=128, out_dim=2, T=100, mlp_model=None):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.T = T
        self.betas = torch.linspace(0.0001, 0.02, T).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.model = mlp_model if mlp_model else mlp(in_dim, hidden_dim, out_dim)
        self.model.to(device)
        self.device = device

    def forward_diffusion(self, x0, t):
        noise = torch.randn_like(x0)
        mean = torch.sqrt(self.alphas_cumprod[t])[:, None] * x0
        std = torch.sqrt(1 - self.alphas_cumprod[t])[:, None]
        return mean + std * noise, noise

    def get_eps_from_model(self, x, t):
        # t is a number
        with torch.no_grad():
            return self.model(torch.cat((x, torch.full((x.shape[0], 1), t / self.T, device=self.device)), dim=1))

    def get_eps_from_model_batch(self, x, t):
        # t in a tensor
        with torch.no_grad():
            t = t / self.T
            t = t.float().view(-1, 1)  # Reshape t to be a column vector
            # Concatenate x and t along the feature dimension
            return self.model(torch.cat((x, t), dim=1))

    def compute_mu(self, x, eps, t):
        return (x - self.betas[t] / torch.sqrt(1 - self.alphas_cumprod[t]) * eps) / torch.sqrt(self.alphas[t])

    def compute_mu_batch(self, x, eps, t):
        t = t.long()
        betas_t = self.betas[t].view(-1, 1)
        alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1)
        alphas_t = self.alphas[t].view(-1, 1)
        mu = (x - betas_t / torch.sqrt(1 - alphas_cumprod_t) * eps) / torch.sqrt(alphas_t)
        return mu

    def sample(self, n_samples=1000):
        x = torch.randn(n_samples, 2, device=self.device)
        for t in reversed(range(self.T)):
            z = torch.randn_like(x, device=self.device) if t > 0 else 0
            eps = self.get_eps_from_model(x, t)
            mu = self.compute_mu(x, eps, t)
            x = mu + torch.sqrt(self.betas[t]) * z
        return x.detach().cpu().numpy()

    def get_x0_from_xt(self, xt, t):
        """Extracts x0 (clean sample) from xt using the estimated noise epsilon."""
        t = t.long()

        # Retrieve noise estimate (eps) from the model
        eps = self.get_eps_from_model_batch(xt, t)

        # Get alphas_cumprod for the given timesteps
        alphas_cumprod_t = torch.tensor(self.alphas_cumprod[t]).view(-1, 1)  # Shape: (batch_size, 1)

        # Compute x0 using the formula
        x0 = (xt - torch.sqrt(1 - alphas_cumprod_t) * eps) / torch.sqrt(alphas_cumprod_t)

        return x0


def train_ddpm(ddpm_model, dataset, epochs=5000, batch_size=128, lr=3e-2, n_samples=5000, noise_dataset=0.1, log_every=500):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = optim.Adam(ddpm_model.model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    ddpm_model.to(device)
    for epoch in range(epochs):
        idx = torch.randint(0, dataset.shape[0], (batch_size,))
        x0 = dataset[idx].to(device)
        t = np.random.randint(0, ddpm_model.T, (batch_size,))
        xt, noise = ddpm_model.forward_diffusion(x0, t)
        noise_pred = ddpm_model.model(torch.cat((xt, torch.tensor(t[:, None], device=device) / ddpm_model.T), dim=1))
        loss = criterion(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % log_every == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


def mlp(in_dim, hidden_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim + 1, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 2 * hidden_dim),
        nn.ReLU(),
        nn.Linear(2 * hidden_dim, 2 * hidden_dim),
        nn.ReLU(),
        nn.Linear(2 * hidden_dim, out_dim)
    )

class ConsistencyModel(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=128, out_dim=2, dropout_prob=0.1, T=120):
        super().__init__()
        # self.model = nn.Sequential(
        #     nn.Linear(in_dim+1, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout_prob),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout_prob),
        #     nn.Linear(hidden_dim, out_dim)
        # )
        self.model = mlp(in_dim, hidden_dim, out_dim)
        self.T = T
        self.eps = 0.002
        self.input_dim = in_dim
        self.output_dim = out_dim
        self.hidden_dim = hidden_dim

    def c_skip(self, t):
        """Skip coefficient ensuring c_skip(eps) = 1."""
        return 1 - ((t - self.eps) / self.T)

    def c_out(self, t):
        """Output coefficient ensuring c_out(eps) = 0."""
        return (t - self.eps) / self.T

    def forward(self, x, t):
        t = t.view(-1, 1)  # to use broadcast
        x_t = torch.cat([x, t], dim=-1)
        f_theta = self.model(x_t)

        # Compute skip and output coefficients
        c_skip_t = self.c_skip(t)
        c_out_t = self.c_out(t)

        output = c_skip_t * x + c_out_t * f_theta

        return output

def compute_time_steps(N, epsilon=2e-3, T=80, rho=7):
    """
    Compute discretized time steps following Karras et al. (2022).
    """
    return torch.tensor([
        (epsilon**(1/rho) + (i-1)/(N-1) * (T**(1/rho) - epsilon**(1/rho)))**rho
        for i in range(1, N+1)
    ])


def compute_score_function(ddpm, xt, t, n_idx):
    """ Computes the score function s_phi(x_t, t) """
    with torch.no_grad():
        n_idx = n_idx.view(-1, 1)
        eps = ddpm.model(torch.cat((xt, n_idx / ddpm.T), dim=1))
    std = torch.sqrt(1 - ddpm.alphas_cumprod[n_idx].view(-1, 1))

    score = -eps / std
    return score


def compute_x_hat(ddpm, x_tn1, t_n, t_n1, n_idx1):
    """
    Computes the consistency update rule:
    x̂_tn = x_t(n+1) - (t_n - t_n1) * t_n1 * s_phi(x_t(n+1), t_n1)
    """
    # Compute score function s_phi(x_t(n+1), t_n1)
    score_tn1 = compute_score_function(ddpm, x_tn1, t_n1, n_idx1)

    # Compute update step
    x_hat_tn = x_tn1 - (t_n - t_n1).view(-1, 1) * t_n1.view(-1, 1) * score_tn1

    return x_hat_tn


def consistency_distillation(cd_model, ddpm, dataset, epochs=5000, batch_size=128, lr=1e-4, mu=0.99,
                             log_every=500):
    """
    Train a consistency model using Consistency Distillation (CD) algorithm.

    Args:
        cd_model: The consistency model to be trained.
        dataset: The training dataset.
        epochs: Number of training iterations.
        batch_size: Number of samples per batch.
        lr: Learning rate for optimization.
        mu: EMA update rate for shadow model.
        N: Number of discrete timesteps.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = optim.Adam(cd_model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # Define distance function d(·,·)

    N = ddpm.T
    # Compute discretized time steps
    time_steps = compute_time_steps(N, T=1).to(device)

    # Shadow model for exponential moving average (EMA)
    cd_model_shadow = ConsistencyModel()
    cd_model_shadow.load_state_dict(cd_model.state_dict())
    cd_model_shadow.eval()

    cd_model.to(device)
    cd_model_shadow.to(device)

    for epoch in range(epochs):
        cd_model.train()
        cd_model_shadow.eval()
        idx = torch.randint(0, dataset.shape[0], (batch_size,))
        x = dataset[idx].to(device)

        # Sample index from precomputed time steps
        n_idx = torch.randint(0, N - 1, (batch_size,), device=device)
        t_n = time_steps[n_idx]
        n_idx1 = n_idx + 1
        t_n1 = time_steps[n_idx1]  # Ensure t_n1 > t_n

        # Sample x_{t_{n+1}} ~ N(x; t_{n+1}^2 I)
        noise = torch.randn_like(x)
        x_tn1 = x + (t_n1.view(-1, 1)) * noise

        # Compute x̂_{t_n} using the update rule
        with torch.no_grad():
            x_hat_tn = compute_x_hat(ddpm, x_tn1, t_n, t_n1, n_idx1)

            # Compute the target f_{\theta^-}(x̂_{t_n}, t_n)
            cm_target = cd_model_shadow(x_hat_tn, t_n)

        # Model prediction f_θ(x_{t_{n+1}}, t_{n+1})
        cm_pred = cd_model(x_tn1, t_n1)

        lambda_tn = 1
        loss = (lambda_tn * criterion(cm_pred, cm_target)).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # EMA update for shadow model
        with torch.no_grad():
            for param, shadow_param in zip(cd_model.parameters(), cd_model_shadow.parameters()):
                shadow_param.data = mu * shadow_param.data + (1 - mu) * param.data
        print(f"Epoch {str(epoch).rjust(5)}, Loss: {loss.item():.6f}")
        if epoch % log_every == 0:
            # samples = multistep_consistency_sampling(cd_model, N, n_samples=1000, device=device).numpy()
            samples = one_step_consistency_sampling(cd_model, time_steps, n_samples=1000, device=device).numpy()
            plt.scatter(samples[:, 0], samples[:, 1], alpha=0.1)
            plt.title(f"Generated Samples (Consistency Model) epoch={epoch}")
            plt.show()


def multistep_consistency_sampling(cm, N, n_samples=1000, device="cuda"):
    """
    Multistep Consistency Sampling algorithm for multiple samples.

    Args:
        cm: Trained consistency model.
        N: Number of time steps.
        n_samples: Number of samples to generate.
        device: Computation device.

    Returns:
        Sampled data from the model.
    """
    with torch.no_grad():
        time_steps = torch.flip(compute_time_steps(N).to(device), dims=[0])  # Reverse the time steps
        x = torch.randn(n_samples, cm.input_dim, device=device)  # Initial noise samples

        # Ensure time_steps[0] has the same batch size as x
        t = time_steps[0].expand(n_samples)
        x = cm(x, t)  # Initial denoising step at t = T
        stds = [torch.sqrt(tau_n ** 2 - time_steps[-1] ** 2) for tau_n in time_steps]
        for n in range(N - 1):
            z = torch.randn_like(x, device=device)  # Gaussian noise sample
            tau_n = time_steps[n]
            x_hat_tau_n = x + stds[n] * z
            x = cm(x_hat_tau_n, tau_n.expand(n_samples))
    return x.cpu()


def one_step_consistency_sampling(cm,time_steps,n_samples=1000, device="cuda" ):
    """
    One-Step Consistency Sampling algorithm.

    Args:
        cm: Trained consistency model.
        n_samples: Number of samples to generate.
        device: Computation device.

    Returns:
        Sampled data from the model.
    """
    with torch.no_grad():
        # Generate initial random noise samples
        x = torch.randn(n_samples, cm.input_dim, device=device)

        n_idx = torch.randint(0, len(time_steps), (n_samples,), device=device)
        t_n = time_steps[n_idx]
        # Perform a single consistency model denoising step
        x = cm(x, t_n)

    return x.cpu()

if __name__=="__main__":
    ddpm = torch.load("ddpm.pt")
    dataset = torch.load("dataset.pt")
    cd_model = ConsistencyModel()
    consistency_distillation(cd_model, ddpm, dataset, epochs=10000, batch_size=128, lr=1e-2, mu=0.95,
                             log_every=250)

