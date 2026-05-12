from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F
import torch
import numpy as np
from time import time
from model.PixelCNN import PixelCNN


# init
num_epoch = 200
batch_size = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr=1e-3
func_num_K = 10

rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5

ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])
# data
traindata = datasets.MNIST(root="./dataset/data", train=True, download=True, transform=ds_transforms)
validdata = datasets.MNIST(root="./dataset/data", train=False, download=True, transform=ds_transforms)

train_dataloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, pin_memory=True)
valid_dataloader = DataLoader(validdata, batch_size=batch_size, pin_memory=True)


# model
output_num = 3 * func_num_K
model = PixelCNN(in_channel=1, filters_num=128, out_channel=output_num, residual_num=12).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications
# http://arxiv.org/abs/1701.05517


def log_sum_exp(x: torch.Tensor) -> torch.Tensor:
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis  = len(x.size()) - 1
    m, _  = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def log_prob_from_logits(x: torch.Tensor) -> torch.Tensor:
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))


def mix_logistic_loss_1_color_channel(output: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    '''
    output : [B, Output_num, h, w]
    X : [B, channel=1, h, w]
    '''
    output = output.permute(0, 2, 3, 1)
    X = X.permute(0, 2, 3, 1)
    X = X.unsqueeze(-1)

    # capture the pi_i, mu_i, gamma_i
    pi = output[:, :, :, :func_num_K]   # [B, h, w, K]
    mu = output[:, :, :, func_num_K:func_num_K*2].unsqueeze(-1).transpose(-1, -2)   # [B, h, w, 1, K]
    log_gamma = torch.clamp(output[:, :, :, func_num_K*2:func_num_K*3].unsqueeze(-1).transpose(-1, -2), min=-7.)

    # cal the maximum likelihood estimation: $\sum_{i} \log \pi p(x; \mu, \gamma) = \sum_i \log \pi + sum_i \log p(x; \mu, \gamma)
    # and then the p(x; \mu, \gamma) \propto F(x + \Delta x; \mu, \gamma) - F(x - \Delta x; \mu, \gamma)
    # F(x) = \frac{1}{1+e^{-(x - \mu)\gamma}} = \text{sigmoid}((x - \mu)/\gamma)

    x_mu = X - mu
    inv_gamma = torch.exp(-log_gamma)
    # \Delta x = 1/255
    Delta_x = 1/255
    plus_in = inv_gamma*(x_mu + Delta_x)
    sub_in = inv_gamma*(x_mu - Delta_x)
    cdf_plus = F.sigmoid(plus_in)
    cdf_sub = F.sigmoid(sub_in)
    delta_cdf_mid = cdf_plus - cdf_sub

    # boundary value process
    log_cdf_left = plus_in - F.softplus(plus_in)    # left = F(x+\Delta x) - F(-1) = F(x+\Delta x)
    log_cdf_right = - F.softplus(sub_in)            # right = F(1) - F(x-\Delta x) = 1 - F(x-\Delta x)
    ## some extreme cases directly use the PDF
    mid_in = inv_gamma * x_mu
    log_pdf_mid = mid_in - log_gamma - 2. * F.softplus(mid_in)  # F'(x) and softplus(x) = x + softplus(-x)
    ## boundary
    extreme_cond = (delta_cdf_mid > 1e-5).float()
    log_probs = extreme_cond * torch.log(torch.clamp(delta_cdf_mid, min=1e-12)) + (1. - extreme_cond) * (log_pdf_mid - np.log(127.5))
    left_cond = (X < -0.999).float()
    log_probs = left_cond * log_cdf_left + (1. - left_cond) * log_probs
    right_cond = (X > 0.999).float()
    log_probs = right_cond * log_cdf_right + (1. - right_cond) * log_probs

    # log weight sum
    log_probs = torch.sum(log_probs, dim=3) + log_prob_from_logits(pi)

    return -torch.sum(log_sum_exp(log_probs))


def sample_from_mix_logistic_1_color_channel(output: torch.Tensor):
    '''
    output [B, output_num, h, w]
    '''
    output = output.permute(0, 2, 3, 1)

    # use the Gumbel-Max to choose sample probability function
    pi = output[:, :, :, :func_num_K]
    temp = torch.zeros_like(pi).uniform_(1e-5, 1-1e-5)
    temp = pi - torch.log(-torch.log(temp))
    select_index = temp.argmax(dim=3, keepdim=True)

    mu = torch.gather(output[:, :, :, func_num_K:2*func_num_K], dim=3, index=select_index)
    log_gamma = torch.clamp(
        torch.gather(output[:, :, :, 2*func_num_K:], dim=3, index=select_index), min=-7.)
    u = torch.zeros_like(mu).uniform_(1e-5, 1-1e-5)
    x = torch.clamp(mu + torch.exp(log_gamma) * (torch.log(u) - torch.log(1. - u)), min=-1., max=1.)
    out = x.squeeze(-1).unsqueeze(1)
    return out

loss_func = mix_logistic_loss_1_color_channel
sample_func = sample_from_mix_logistic_1_color_channel


def train(model: nn.Module, data: torch.Tensor) -> float:
    model.train()
    data = data.to(device)
    output = model(data)

    loss = loss_func(output, data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # bpd: byte per dim
    NLL_loss = loss.cpu().item()
    deno = torch.prod(torch.tensor(traindata.data[0].shape)) * batch_size * np.log(2)
    bpd = NLL_loss / deno

    return bpd


@torch.inference_mode()
def eval(model: nn.Module, data: torch.Tensor) -> float:
    model.eval()
    data = data.to(device)
    output = model(data)
    loss = loss_func(output, data)

    # bpd: byte per dim
    NLL_loss = loss.cpu().item()
    deno = torch.prod(torch.tensor(traindata.data[0].shape)) * batch_size * np.log(2)
    bpd = NLL_loss / deno

    return bpd


@torch.inference_mode()
def sample(model: nn.Module, sample_num: int, channel: int, height: int, width: int, epoch: int) -> None:
    sample_tensor = torch.zeros((sample_num, channel, height, width)).to(device)

    for h in range(height):
        for w in range(width):
            y_prob = model(sample_tensor)
            output = sample_func(y_prob)
            sample_tensor[:, :, h, w] = output[:, :, h, w]

    sample_tensor = rescaling_inv(sample_tensor)
    utils.save_image(sample_tensor, f'./result/pp_sample_{epoch+1}.png', nrow=6, padding=0)


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

    print(f"epoch:{epoch+1}, trian bpd:{trian_epoch_loss:.4f}, train cost:{train_time:.2f}s, valid bpd:{valid_epoch_loss:.4f}, valid cost:{valid_time:.2f}s")

    # image sample
    sample(model, 36, 1, 28, 28, epoch)
