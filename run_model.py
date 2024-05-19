from model import DiffusionModel
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from train import test_model



if __name__ == '__main__':
    model = DiffusionModel(lr=0.01).to("cuda")
    model.load("models/model_1_100.pth")
    dataset = torch.load('datasets/cats.pth')
    test_model(model, dataset, 100, device="cuda", beta=0.025)
