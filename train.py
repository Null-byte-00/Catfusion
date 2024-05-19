from noise_scheduler import add_noise
from model import DiffusionModel
import matplotlib.pyplot as plt
from torchvision import transforms
import torch


def train_sample(model, data, total_timesteps, device="cpu", verbose=False):
    losses = []
    for t in range(total_timesteps)[::-1]:
        t = torch.tensor(t).to(device).clone().detach()
        total_timesteps = torch.tensor(total_timesteps).to(device).clone().detach()
        noisy_tensor, noise = add_noise(data, t, total_timesteps, device)
        noisy_tensor = noisy_tensor.unsqueeze(0)
        noise = noise.unsqueeze(0)
        model.to(device)
        loss = model.training_step(noisy_tensor, noise, t)
        losses.append(loss)
    
    avg_loss = sum(losses) / len(losses)
    if verbose:
        print(f"Average loss: {avg_loss}")
    return model


def denoise_timestep(model, data, timestep, total_timesteps, device="cpu", beta=1.0):
    t = torch.tensor(timestep).to(device)
    total_timesteps = torch.tensor(total_timesteps).to(device).clone().detach()
    
    noise = model(data, t)
    denoised = data - (noise * beta)
    return torch.clamp(denoised, -1, 1).clone().detach()


def train_model(model, dataset, total_timesteps,
                epochs=3, device="cuda", verbose=True):
    for epoch in range(epochs):
        for i, data in enumerate(dataset):
            data = data[0].to(device)
            model = train_sample(model, data, total_timesteps, device, verbose=verbose)
            if verbose:
                print(f"Epoch {epoch} - Sample {i}")
            if i%100 == 0:
                model.save(f"models/model_{epoch}_{i}.pth")


def show_images(images=[]):
    to_pil = transforms.ToPILImage()
    fig, axs = plt.subplots(1, len(images), figsize=(11, 3))
    for i, image in enumerate(images):
        axs[i].imshow(to_pil(image))
    plt.show()


def test_model(model, dataset, total_timesteps, device="cuda", beta=0.025):
    process_images = []
    data = dataset[5][0].to(device)
    random_img, _ = add_noise(data.unsqueeze(0), 99, total_timesteps, device)
    process_images.append(random_img[0].cpu())
    for timestep in range(total_timesteps)[::-1]:
        random_img = denoise_timestep(model, random_img, timestep, total_timesteps, device, beta=beta).clamp(-1, 1)
        if timestep in [0, 20, 50, 70, 90, 100]:
            process_images.append(random_img[0].cpu())
    show_images(process_images)


if __name__ == '__main__':
    dataset = torch.load('datasets/cats.pth')
    #dataset = torch.utils.data.Subset(dataset, range(10))
    model = DiffusionModel(lr=0.01)
    total_timesteps = 100
    train_model(model, dataset, total_timesteps, epochs=1, device="cuda")
    test_model(model, torch.load('datasets/cats.pth'), total_timesteps)