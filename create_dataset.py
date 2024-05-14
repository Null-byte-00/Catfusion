from torchvision import datasets, transforms
import torch


def create_dataset(data_dir, dataset_name='datasets/cats.pth'):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # Normalize the images to the range [-1, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    torch.save(dataset, dataset_name)


if __name__ == '__main__':
    create_dataset('datasets')
