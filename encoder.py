import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):

    def cuda_if(self, tobj):
        if torch.cuda.is_available():
            tobj = tobj.cuda()
        return tobj

    def __init__(self, obs_size, n_classes, emb_size=256, bnorm=True):
        """
        obs_size - the size of the input data. Shape = (..., C, H, W)
        """
        super(Encoder, self).__init__()
        self.obs_size = obs_size
        self.emb_size = emb_size
        self.bnorm = bnorm

        # Encoder
        self.convs = nn.ModuleList([])
        shape = [*self.obs_size[-3:]]

        ksize=3; padding=1; stride=1; out_depth = 16
        self.convs.append(self.conv_block(obs_size[-3], out_depth, ksize=ksize, padding=padding, stride=stride, bnorm=self.bnorm))
        shape = self.get_new_shape(shape, out_depth, ksize=ksize, stride=stride, padding=padding)

        ksize=3; padding=0; stride=2; in_depth=out_depth
        out_depth = 32
        self.convs.append(self.conv_block(in_depth, out_depth, ksize=ksize, padding=padding, stride=stride, bnorm=self.bnorm))
        shape = self.get_new_shape(shape, out_depth, ksize=ksize, stride=stride, padding=padding)

        ksize=3; padding=1; stride=2; in_depth = out_depth
        out_depth = 64
        self.convs.append(self.conv_block(in_depth, out_depth, ksize=ksize, padding=padding, stride=stride, bnorm=self.bnorm))
        shape = self.get_new_shape(shape, out_depth, ksize=ksize, stride=stride, padding=padding)

        ksize=3; padding=1; stride=2; in_depth = out_depth
        out_depth = 64
        self.convs.append(self.conv_block(in_depth, out_depth, ksize=ksize, padding=padding, stride=stride, bnorm=self.bnorm))
        shape = self.get_new_shape(shape, out_depth, ksize=ksize, stride=stride, padding=padding)

        self.features = nn.Sequential(*self.convs)
        self.feat_shape = shape
        self.flat_size = int(np.prod(shape))
        self.mu = nn.Linear(self.flat_size, emb_size)
        self.sigma = nn.Linear(self.flat_size, emb_size)

        # Reconstructor
        self.deconvs = nn.ModuleList([])

        ksize=5; padding=0; stride=1; in_depth = out_depth
        out_depth = 64
        self.deconvs.append(self.deconv_block(in_depth, out_depth, ksize=ksize, padding=padding, stride=stride, bnorm=self.bnorm))

        ksize=5; padding=0; stride=1; in_depth = out_depth
        out_depth = 64
        self.deconvs.append(self.deconv_block(in_depth, out_depth, ksize=ksize, padding=padding, stride=stride, bnorm=self.bnorm))

        ksize=4; padding=0; stride=2; in_depth = out_depth
        out_depth = 32
        self.deconvs.append(self.deconv_block(in_depth, out_depth, ksize=ksize, padding=padding, stride=stride, bnorm=self.bnorm))

        ksize=5; padding=0; stride=1; in_depth = out_depth
        out_depth = 16
        self.deconvs.append(self.deconv_block(in_depth, out_depth, ksize=ksize, padding=padding, stride=stride, bnorm=self.bnorm))

        ksize=3; padding=0; stride=1; in_depth = out_depth
        out_depth = obs_size[-3]
        self.deconvs.append(self.deconv_block(in_depth, out_depth, ksize=ksize, padding=padding, stride=stride, bnorm=False))

        self.rwd_flat = nn.Sequential(nn.Linear(emb_size, self.flat_size),nn.ReLU(), nn.BatchNorm1d(1024))
        self.rwd_features = nn.Sequential(*self.deconvs)

        # Classifier
        block = []
        block.append(nn.Linear(self.emb_size, 200))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm1d(200))
        block.append(nn.Linear(200,int(n_classes)))
        self.classifier = nn.Sequential(*block)

    def get_new_shape(self, old_shape, depth, ksize=3, stride=1, padding=1):
        new_shape = [depth]
        for i in range(len(old_shape[1:])):
            new_shape.append((old_shape[i+1] - ksize + 2*padding)//stride + 1)
        return new_shape

    def forward(self, x):
        fx = self.features(x)
        fx = fx.view(-1, self.flat_size)
        mu = self.mu(fx)
        sigma = self.sigma(fx)
        z = mu + sigma*Variable(self.cuda_if(torch.normal(means=torch.zeros(mu.shape), std=1)))
        fx = self.rwd_flat(z)
        fx = fx.view(-1, *self.feat_shape)
        remake = self.rwd_features(fx)
        return z, remake

    def classify(self, z):
        return self.classifier(z)

    def conv_block(self,in_depth,out_depth,ksize=3,stride=1,padding=1,activation='relu',bnorm=False):
        block = []
        block.append(nn.Conv2d(in_depth, out_depth, ksize, stride=stride, padding=padding))
        if activation is None:
            pass
        elif activation.lower() == 'relu':
            block.append(nn.ReLU())
        elif activation.lower() == 'tanh':
            block.append(nn.Tanh())
        if bnorm:
            block.append(nn.BatchNorm2d(out_depth))
        return nn.Sequential(*block)

    def deconv_block(self,in_depth,out_depth,ksize=3,stride=1,padding=1,activation='relu',bnorm=False):
        block = []
        block.append(nn.ConvTranspose2d(in_depth, out_depth, ksize, stride=stride, padding=padding))
        if activation is None:
            pass
        elif activation.lower() == 'relu':
            block.append(nn.ReLU())
        elif activation.lower() == 'tanh':
            block.append(nn.Tanh())
        if bnorm:
            block.append(nn.BatchNorm2d(out_depth))
        return nn.Sequential(*block)
        
