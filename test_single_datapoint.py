import torch
from noise_scheduler import add_noise
from model import DiffusionModel
from torchvision import transforms
import matplotlib.pyplot as plt


device = "cuda"
to_pil = transforms.ToPILImage()
model = DiffusionModel(lr=0.004)
dataset = torch.load('datasets/cats.pth')
data = dataset[1][0].unsqueeze(0).to(device)
timestep = torch.tensor(30).to(device)
total_timesteps = torch.tensor(100).to(device)
model.to(device)


def show_3_images(original, noisy, denoised):
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(to_pil(original[0].cpu()))
    axs[0].set_title("Original")
    axs[1].imshow(to_pil(noisy[0].cpu()))
    axs[1].set_title("Noisy")
    axs[2].imshow(to_pil(denoised[0].cpu()))
    axs[2].set_title("Denoised")
    plt.show()


def test_model(verbose=True):
    for epoch in range(20000):
        noisy_tensor, noise = add_noise(data, timestep, total_timesteps, device)
        loss = model.training_step(noisy_tensor, noise, timestep)
        if verbose:
            print(f"Epoch {epoch}: Loss: {loss}")
    
    noisy_tensor, noise = add_noise(data, timestep, total_timesteps, device)
    pred_noise = model(noisy_tensor, timestep)
    denoised = noisy_tensor - pred_noise
    denoised = torch.clamp(denoised, -1, 1)
    show_3_images(data, noisy_tensor, denoised)


if __name__ == '__main__':
    test_model()