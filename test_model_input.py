"""
Test if the model can distinguish between different inputs
"""
from model import DiffusionModel
import torch


def rand_img():
    return torch.randn(1, 3, 64, 64).clamp(-1, 1)


time = torch.tensor(10)
i_1, i_2, i_3, i_4 = rand_img(), rand_img(), rand_img(), rand_img()
t_1, t_2, t_3, t_4 = rand_img(), rand_img(), rand_img(), rand_img()


def test_model():
    model = DiffusionModel(lr=0.01)
    for epoch in range(1000):
        loss_1 = model.training_step(i_1, t_1, time)
        loss_2 = model.training_step(i_2, t_2, time)
        loss_3 = model.training_step(i_3, t_3, time)
        loss_4 = model.training_step(i_4, t_4, time)
        print(f"Epoch {epoch}: Losses: {loss_1}, {loss_2}, {loss_3}, {loss_4}")


if __name__ == '__main__':
    test_model()