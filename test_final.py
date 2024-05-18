import torch
from torchvision import datasets, transforms
from train import train_sample, denoise_timestep
from model import DiffusionModel
from noise_scheduler import add_noise
import matplotlib.pyplot as plt


device = "cuda"
dataset = torch.load('datasets/cats.pth')
data = dataset[1][0].to(device)
total_timesteps = torch.tensor(20).to(device)
to_pil = transforms.ToPILImage()


def show_images(images=[]):
    fig, axs = plt.subplots(1, len(images), figsize=(11, 3))
    for i, image in enumerate(images):
        axs[i].imshow(to_pil(image))
    plt.show()


def test_model(verbose=True):
    model = DiffusionModel(lr=0.01).to(device)
    for epoch in range(200):
        model = train_sample(model, data, total_timesteps, device, verbose=verbose)
        if verbose:
            print(f"Epoch {epoch}")

    process_images = []
    random_img, _ = add_noise(data.unsqueeze(0), 19 , total_timesteps, device)  #torch.randn(1, 3, 64, 64).clamp(-1, 1).to(device)
    process_images.append(random_img[0].cpu())
    for timestep in range(total_timesteps)[::-1]:
        random_img = denoise_timestep(model, random_img, timestep, total_timesteps, device, beta=0.15).clamp(-1, 1)
        if timestep in [0, 4, 10, 14, 18, 20]:
            process_images.append(random_img[0].cpu())
    show_images(process_images)

if __name__ == '__main__':
    test_model()
