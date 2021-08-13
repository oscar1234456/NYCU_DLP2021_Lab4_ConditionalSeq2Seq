import torch

def reparameter(mean, logVar):
    std = torch.exp(logVar*0.5) #TODO: Change not std
    esp = torch.randn_like(std)
    return mean + std * esp