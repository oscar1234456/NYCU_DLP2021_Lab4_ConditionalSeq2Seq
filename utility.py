##
import torch
##
def reparameter(mean, logVar):
    std = torch.exp(logVar*0.5) #TODO: Change not std
    esp = torch.randn_like(std)
    return mean + std * esp

##
def KLDLoss(mean, logVar):
    return -0.5*(torch.sum(1+logVar-mean.pow(2)-logVar.exp()))

##
mean = torch.Tensor([[1,2,3]])
var = torch.Tensor([[4,5,6]])
print(KLDLoss(mean, var))