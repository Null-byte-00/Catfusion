"""
Test if the model can distinguish between different time steps
"""
from model import DiffusionModel
import torch

def rand_img():
    return torch.randn(1, 3, 64, 64).clamp(-1, 1)

input_img = rand_img()

def test_model():
    model = DiffusionModel(lr=0.01)
    t_1_img = rand_img()
    t_10_img = rand_img()
    t_50_img = rand_img()
    t_100_img = rand_img()
    for epoch in range(1000):
        loss_1 = model.training_step(input_img, t_1_img, torch.tensor(1))
        loss_10 = model.training_step(input_img, t_10_img, torch.tensor(10))
        loss_50 = model.training_step(input_img, t_50_img, torch.tensor(50))
        loss_100 = model.training_step(input_img, t_100_img, torch.tensor(100))
        print(f"Epoch {epoch}: Losses: {loss_1}, {loss_10}, {loss_50}, {loss_100}")
    

if __name__ == '__main__':
    test_model()