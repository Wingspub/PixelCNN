from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F
import torch
import numpy as np
from time import time
from model.PixelCNN import PixelCNN

traindata = datasets.MNIST(root="./dataset/data", train=True, download=True, transform=transforms.ToTensor())
validdata = datasets.MNIST(root="./dataset/data", train=False, download=True, transform=transforms.ToTensor())

train_dataloader = DataLoader(traindata, batch_size=128, shuffle=True, pin_memory=True)
valid_dataloader = DataLoader(validdata, batch_size=32, pin_memory=True)

# init
num_epoch = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr=1e-3

# model
model = PixelCNN(1, 128, 256, 12).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_func = F.cross_entropy


def train(model: nn.Module, data: torch.Tensor) -> float:
    model.train()
    data = data.to(device)
    target = (data.reshape(-1, 28, 28) * 255).long()
    output = model(data)

    loss = loss_func(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

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
    s1 = time()
    for data, label in train_dataloader:
        loss = train(model, data)
        train_loss_list.append(loss)
    train_time = time() - s1
    trian_epoch_loss = np.mean(train_loss_list)

    valid_loss_list = []
    s1 = time()
    for data, label in valid_dataloader:
        loss = eval(model, data)
        valid_loss_list.append(loss)
    valid_time = time() - s1
    valid_epoch_loss = np.mean(valid_loss_list)

    print(f"epoch:{epoch+1}, trian NLL:{trian_epoch_loss:.4f}, train cost:{train_time:.2f}s, valid NLL:{valid_epoch_loss:.4f}, valid cost:{valid_time:.2f}s")

    # image sample
    sample(model, 36, 1, 28, 28, epoch)
