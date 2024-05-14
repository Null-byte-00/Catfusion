import torch

def add_noise(tensor, timestep, total_timesteps, device="cpu"):
    how_much_noise = torch.sqrt(1 - ((total_timesteps - timestep) / total_timesteps)).to(device)

    noise = torch.randn_like(tensor) * 0.45 * how_much_noise

    noisy_tensor = tensor + noise

    noisy_tensor = torch.clamp(noisy_tensor, -1, 1)

    return noisy_tensor, noise