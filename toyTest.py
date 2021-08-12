##
import torch
##
with open("./data/train.txt", 'r') as f:
    a = f.read()

print(a.split())
c = a.split()
with open("./data/test.txt", 'r') as f:
    d = f.read()
e = d.split()
##
t = torch.rand(4, 3)
b = t.view(-1,1)
