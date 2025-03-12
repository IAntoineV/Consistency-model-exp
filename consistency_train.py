import torch
from copy import deepcopy







# Define the ConsistencyTraining class
class ConsistencyTraining:
    def __init__(self, model, shape, lr, step_schedule, ema_decay_schedule, distance_fn, weight_fn, device="cpu"):
        self.model = model
        self.lr = lr
        self.step_schedule = step_schedule
        self.ema_decay_schedule = ema_decay_schedule
        self.distance_fn = distance_fn
        self.weight_fn = weight_fn

        # Initialize EMA model (θ')
        self.ema_model = deepcopy(model)
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.k = 0  # Training step counter
        self.device=device
        self.shape=list(shape)
    def sample_noise(self, x):
        return torch.randn_like(x)
    def train_step(self, x):
        """Performs one training iteration."""
        self.optimizer.zero_grad()
        b = x.size(0)


        Nk = self.step_schedule(self.k)
        n = torch.randint(low=1, high=Nk, size=(b,), device=self.device)  # n ∈ [1, Nk-1]

        # Calculate time steps
        tn =  (n / Nk).view(-1, *([1] * (x.dim() - 1)))
        tn_plus_1 = ((n + 1) / Nk).view(-1, *([1] * (x.dim() - 1)))

        # Generate noise and perturb inputs
        z = self.sample_noise(x)
        x_next = x + tn_plus_1 * z
        x_current = x + tn * z

        # Forward passes
        pred_next = self.model(x_next, tn_plus_1)
        with torch.no_grad():
            pred_current = self.ema_model(x_current, tn)




        # Compute loss
        loss = self.weight_fn(tn) * self.distance_fn(pred_next, pred_current)

        loss.backward()
        self.optimizer.step()

        # Update EMA model
        mu = self.ema_decay_schedule(self.k)
        self._update_ema(mu)
        self.k += 1

        return loss.item()

    def _update_ema(self, mu):
        """Updates EMA model parameters using current model."""
        with torch.no_grad():
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.data = mu * ema_param + (1 - mu) * param.detach()

    def generate_samples(self, num_samples=100, N=10):
        """Generates samples from the EMA model using Multistep Consistency Sampling."""
        self.ema_model.eval()
        T = 1
        with torch.no_grad():
            # Initial noise
            e = torch.zeros([num_samples] +self.shape).to(self.device)
            z = self.sample_noise(e)

            # Sequence of time points
            tau = torch.linspace(0, T, N + 1, device=self.device)[1:]  # Exclude t=0

            # Initial sample
            x = z

            for n in range(N ):
                # Sample z from N(0, I)
                z = self.sample_noise(x)

                # Update x
                x_tau_n = x - (tau[n] ** 2) * z
                t_used = tau[n].expand(num_samples).view(-1, *([1] * (x.dim() - 1)))
                x = self.ema_model(x_tau_n, t_used)

        return x.cpu()
