from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F
import torch
import numpy as np
from time import time

traindata = datasets.MNIST(root="./dataset/data", train=True, download=True, transform=transforms.ToTensor())
validdata = datasets.MNIST(root="./dataset/data", train=True, download=True, transform=transforms.ToTensor())

train_dataloader = DataLoader(traindata, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(validdata, batch_size=32)

# init
num_epoch = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr=1e-3
valid_epoch_num = 5

# model
model = nn.Conv2d(1, 256, 7, 1, 3, bias=False).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_func = F.cross_entropy


def train(model: nn.Module, data: torch.Tensor) -> float:
    model.train()
    data = data.to(device)
    target = (data.reshape(-1, 28, 28) * 255).long()
    output = model(data)

    loss = loss_func(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.cpu().item()


@torch.inference_mode()
def eval(model: nn.Module, data: torch.Tensor) -> float:
    model.eval()
    data = data.to(device)
    target = (data.reshape(-1, 28, 28) * 255).long()
    output = model(data)
    loss = loss_func(output, target)
    return loss.cpu().item()


@torch.inference_mode()
def sample(model: nn.Module, sample_num: int, channel: int, height: int, width: int, epoch: int) -> None:
    sample_tensor = torch.zeros((sample_num, channel, height, width)).to(device)

    for c in range(channel):
        for h in range(height):
            for w in range(width):
                y_prob = F.softmax(model(sample_tensor)[:, :, h, w])
                sample_tensor[:, c, h, w] = torch.multinomial(y_prob, 1).squeeze(-1).float() / 255.

    utils.save_image(sample_tensor, f'./result/sample_{epoch+1}.png', nrow=6, padding=0)

for epoch in range(num_epoch):
    train_loss_list = []
    for data, label in train_dataloader:
        loss = train(model, data)
        train_loss_list.append(loss)
    trian_epoch_loss = np.mean(train_loss_list)
    print(f"epoch:{epoch}, train_loss: {trian_epoch_loss:.6f}")

    if epoch % valid_epoch_num == 0:
        valid_loss_list = []
        for data, label in valid_dataloader:
            loss = eval(model, data)
            valid_loss_list.append(loss)
        valid_epoch_loss = np.mean(valid_loss_list)
        print(f"epoch:{epoch}, valid_loss: {valid_epoch_loss:.6f}")

        # image sample
        sample(model, 36, 1, 28, 28, epoch)
