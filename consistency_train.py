import torch
from copy import deepcopy


class ConsistencyTraining:
    """Implement the Consistency training steps."""
    def __init__(self, model, shape, lr, step_schedule, ema_decay_schedule, distance_fn, weight_fn, t_range, epsilon,
                 device="cpu"):
        self.model = model
        self.lr = lr
        self.step_schedule = step_schedule
        self.ema_decay_schedule = ema_decay_schedule
        self.distance_fn = distance_fn
        self.weight_fn = weight_fn

        # Initialize EMA
        self.ema_model = deepcopy(model)
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.k = 0  # Training step counter
        self.device = device
        self.shape = list(shape)
        self.t_range = t_range # Tmax for diffusion
        self.rho = 7  # Karras schedule parameter
        self.epsilon=epsilon
    def compute_timesteps(self, N):
        """Compute time steps using Karras et al. (2022) discretization."""
        t_values = (self.epsilon ** (1 / self.rho) + (torch.arange(N, device=self.device) / (N - 1)) *
                    (self.t_range ** (1 / self.rho) - self.epsilon ** (1 / self.rho))) ** self.rho
        return t_values


    def train_step(self, x, multiple_t=10):
        """Performs one training iteration. x corresponds to a batch of input data.
        multiple_t corresponds to the number of timestep to samples per sample in x."""
        self.optimizer.zero_grad()
        b = x.size(0)

        Nk = self.step_schedule(self.k)
        n = torch.randint(low=0, high=Nk-1, size=(b*multiple_t,), device=self.device)
        # Compute timesteps
        timesteps = self.compute_timesteps(Nk)
        tn = timesteps[n].view(-1, *([1] * (x.dim() - 1)))
        tn_plus_1 = timesteps[n + 1].view(-1, *([1] * (x.dim() - 1)))
        x=  x.tile(multiple_t, 1,1)
        x = x.view(b*multiple_t, -1)
        # Generate noise and perturb inputs
        z = torch.randn_like(x)
        x_next = x + tn_plus_1 * z
        x_current = x + tn * z

        pred_next = self.model(x_next, tn_plus_1)
        with torch.no_grad():
            pred_current = self.ema_model(x_current, tn)

        # Compute loss
        loss = (self.weight_fn(tn) * self.distance_fn(pred_next, pred_current)).mean()
        loss.backward()
        self.optimizer.step()

        # Update EMA
        mu = self.ema_decay_schedule(self.k)
        self._update_ema(mu)
        self.k += 1

        return loss.item()

    def _update_ema(self, mu):
        """Updates EMA model parameters using current model."""
        with torch.no_grad():
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.data = mu * ema_param + (1 - mu) * param.detach()

    def generate_samples(self,num_samples=100, N=10):
        """Generates samples from the EMA model using Multistep Consistency Sampling with Karras timesteps."""
        timesteps = self.compute_timesteps(N)

        with torch.no_grad():
            z = torch.randn([num_samples] + self.shape).to(self.device)


            T_torch = self.t_range * torch.ones(1,).expand(num_samples).view(-1, *([1] * (z.dim() - 1))).to(self.device)
            x = self.model(z * T_torch, T_torch) # Initial sample

            for n in range(N - 1, 0, -1):

                z = torch.randn_like(x)

                # Update x
                x_tau_n = x + torch.sqrt(timesteps[n] ** 2 - self.epsilon ** 2) * z
                t_used = timesteps[n].expand(num_samples).view(-1, *([1] * (x.dim() - 1)))
                x = self.model(x_tau_n, t_used)

        return x.cpu()
