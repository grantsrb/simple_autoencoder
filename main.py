import torch
import torchvision
from torch.autograd import Variable
from trainer import Trainer
from encoder import Encoder
import matplotlib.pyplot as plt
import numpy as np
import sys

def cuda_if(tobj):
    if torch.cuda.is_available():
        tobj = tobj.cuda()
    return tobj

def preprocess(imgs, mean, std):
    return (imgs-mean)/(std+1e-7)

def postprocess(remakes, mean, std):
    return remakes*(std+1e-7) + mean

n_epochs = 800
batch_size = 128
lr = .0009
pred_coef = .2 # Portion of loss from classification
process = False
resume = False

if len(sys.argv) > 1:
    for i in range(len(sys.argv)):
        str_arg = str(sys.argv[i])
        if 'lr=' == str_arg[:3]: lr = float(str_arg[3:])
        if 'pred_coef=' in str_arg: pred_coef = float(str_arg[len('pred_coef='):])

print("lr:", lr)
print("pred_coef:", pred_coef)

cifar = torchvision.datasets.CIFAR10("/Users/satchelgrant/Datasets/cifar10", train=True, download=True)
imgs = cifar.train_data
imgs = cuda_if(torch.FloatTensor(imgs.transpose((0,3,1,2))))
mean = imgs.mean()
std = imgs.std()
if process:
    imgs = preprocess(imgs, mean, std)
labels = cuda_if(torch.LongTensor(cifar.train_labels))
perm = cuda_if(torch.randperm(len(imgs)))
train_imgs = imgs[perm[:45000]]
train_labels = labels[perm[:45000]]
val_imgs = imgs[perm[45000:]]
val_labels = labels[perm[45000:]]
 
net = Encoder(imgs.shape, torch.max(labels)+1)
net = cuda_if(net)
if resume:
    net.load_state_dict(torch.load('network.p'))
trainer = Trainer(net, lr=lr, pred_coef=pred_coef)

for epoch in range(n_epochs):
    print("Begin Epoch", epoch)
    losses, accuracies = trainer.train(train_imgs, train_labels, batch_size)
    print("Avg Loss:", np.mean(losses), "– Avg Acc:", np.mean(accuracies))
    if (epoch % 10) == 0:
        acc = 0
        val_batch_size = 300
        for i in range(0,len(val_imgs), val_batch_size):
            zs, remakes = net.forward(Variable(val_imgs[i:i+val_batch_size]))
            logits = net.classify(zs)
            _, max_idxs = torch.max(logits.data, dim=-1)
            acc += torch.eq(max_idxs, val_labels[i:i+val_batch_size]).float().mean()
        acc = acc/(len(val_imgs)/float(val_batch_size))
        torch.save(net.state_dict(), 'network.p')
        print("Val Acc:", acc)

zs, remakes = net.forward(Variable(val_imgs[:20]))
torch.save(net.state_dict(), 'network.p')
if process:
    remakes = postprocess(remakes, mean, std)
remakes = remakes.data.cpu().numpy()
reals = val_imgs[:20]
if process:
    reals = postprocess(reals, mean, std)
reals = reals.cpu().numpy()
np.save('remakes.npy', remakes)
np.save('reals.npy', reals[:20])
