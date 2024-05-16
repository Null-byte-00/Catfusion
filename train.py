from noise_scheduler import add_noise
from model import DiffusionModel
import torch


def train_sample(model, data, total_timesteps, device="cpu"):
    for t in range(total_timesteps):
        t = torch.tensor(t).to(device)
        total_timesteps = torch.tensor(total_timesteps).to(device)
        noisy_tensor, noise = add_noise(data, t, total_timesteps, device)
        noisy_tensor = noisy_tensor.unsqueeze(0)
        noise = noise.unsqueeze(0)
        model.to(device)
        model.training_step(noisy_tensor, noise, t)
    
    return model


def denoise_timestep(model, data, timestep, total_timesteps, device="cpu"):
    t = torch.tensor(timestep).to(device)
    total_timesteps = torch.tensor(total_timesteps).to(device)
    
    noise = model(data, t)
    denoised = data - noise
    return torch.clamp(denoised, -1, 1).clone().detach()