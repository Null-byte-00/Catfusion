from noise_scheduler import add_noise
from model import DiffusionModel
import torch


def train_sample(model, data, total_timesteps, device="cpu"):
    for t in range(total_timesteps):
        t = torch.tensor(t).to(device)
        total_timesteps = torch.tensor(total_timesteps).to(device)
        model.to(device)
        x_1 = add_noise(data, t, total_timesteps)
        x_0 = add_noise(data, t + 1, total_timesteps)

        model.training_step(x_0.unsqueeze(0) , x_1.unsqueeze(0) , t)
    
    return model

