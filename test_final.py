import torch
from torchvision import datasets, transforms
from train import train_sample, denoise_timestep
from model import DiffusionModel
import matplotlib.pyplot as plt


device = "cuda"
dataset = torch.load('datasets/cats.pth')
data = dataset[1][0].to(device)
total_timesteps = torch.tensor(10).to(device)
to_pil = transforms.ToPILImage()


def show_images(images=[]):
    fig, axs = plt.subplots(1, len(images))
    for i, image in enumerate(images):
        axs[i].imshow(to_pil(image))
    plt.show()


def test_model():
    for epoch in range(10000):
        model = DiffusionModel(lr=0.01)
        model = train_sample(model, data, total_timesteps, device, verbose=True)
        print(f"Epoch {epoch}")

    process_images = []
    random_img = torch.randn(1, 3, 64, 64).clamp(-1, 1).to(device)
    for timestep in range(100)[::-1]:
        random_img = denoise_timestep(model, random_img, timestep, total_timesteps, device).clamp(-1, 1)
        if timestep in [0, 2, 5, 7, 9]:
            process_images.append(random_img[0].cpu())
        #print(f"Timestep {timestep}")
    show_images(process_images)

if __name__ == '__main__':
    test_model()
