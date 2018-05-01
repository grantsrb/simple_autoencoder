import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Trainer():

    def cuda_if(self, tobj):
        if torch.cuda.is_available():
            tobj = tobj.cuda()
        return tobj

    def __init__(self, net, lr=1e-3, pred_coef=.1):
        self.net = net
        self.pred_coef = pred_coef
        self.optim = optim.Adam(self.net.parameters(), lr=lr)

    def batch_loss(self, x, y=None):
        loss = 0
        zs, remakes = self.net.forward(x)
        acc = None
        if y is not None:
            logits = self.net.classify(zs)
            loss = self.pred_coef * F.cross_entropy(logits, y)
            maxes, max_idxs = torch.max(logits.data, dim=-1)
            acc = torch.eq(max_idxs, y.data).float().mean()
        return loss + F.mse_loss(remakes, x), acc

    def train(self, X, y=None, batch_size=128):
        idxs = self.cuda_if(torch.randperm(len(X)).long())
        losses = []
        accuracies = []
        for i in range(0, len(X), batch_size):
            self.optim.zero_grad()
            batch_idxs = idxs[i:i+batch_size]
            batch_x = X[batch_idxs]
            if y is not None: batch_y = y[batch_idxs]
            else: batch_y = None
            loss, acc = self.batch_loss(Variable(batch_x), Variable(batch_y))
            loss.backward()
            self.optim.step()
            losses.append(loss.data[0])
            accuracies.append(acc)
            print(i,"/", len(X), "– Loss:", losses[-1], "– Acc:", acc, end='\r')
        return losses, accuracies

