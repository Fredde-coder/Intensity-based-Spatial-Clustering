'''
Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering

Based on https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip
'''

#from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import numpy as np
import torch.nn.init
import random
from copy import deepcopy
use_cuda = torch.cuda.is_available()





# CNN model
class MyNet(nn.Module):
    def __init__(self,input_dim, nChannel, nConv, use_cuda):
        super(MyNet, self).__init__()
        self.nChannel = nChannel
        self.nConv    = nConv
        self.use_cuda = use_cuda


        self.conv1 = nn.Conv2d(input_dim, nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(nConv-1):
            self.conv2.append( nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(nChannel) )
        self.conv3 = nn.Conv2d(nChannel, nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(self.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

class DFC:
    def __init__(self, nChannel = 100, lr = 0.1, nConv = 2, minLabels = 3, stepsize_sim = 1, stepsize_con = 1, use_cuda = False, max_iters = 50):
        self.nChannel = nChannel
        self.lr       = lr
        self.nConv    = nConv
        self.stepsize_sim = stepsize_sim
        self.stepsize_con = stepsize_con
        self.use_cuda     = use_cuda
        self.minLabels    = minLabels
        self.maxIters     = max_iters

    def re_init(self):
        "Reinitalizing for trial testing"
        self.initialize_clustering(self.im)

    def initialize_clustering(self, im):
        self.dim3 = False
        if len(im.shape) == 3:
            data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')/255.]) )
            self.dim3 = True
        else:
            data = np.expand_dims(np.expand_dims(im, 0), 0)
            data = torch.from_numpy(data.astype('float32')/255.)
        if self.use_cuda:
            data = data.cuda()
        self.data = Variable(data)
        self.im = im
        
        # train
        self.model = MyNet( data.size(1), self.nChannel, self.nConv, use_cuda = self.use_cuda)
        if self.use_cuda:
            self.model.cuda()
        self.model.train()

        # similarity loss definition
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # scribble loss definition
        self.loss_fn_scr = torch.nn.CrossEntropyLoss()

        # continuity loss definition
        self.loss_hpy = torch.nn.L1Loss(size_average = True)
        self.loss_hpz = torch.nn.L1Loss(size_average = True)

        self.HPy_target = torch.zeros(im.shape[0]-1, im.shape[1], self.nChannel)
        self.HPz_target = torch.zeros(im.shape[0], im.shape[1]-1, self.nChannel)
        if self.use_cuda:
            self.HPy_target = HPy_target.cuda()
            self.HPz_target = HPz_target.cuda()
            
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.label_colours = np.random.randint(255,size=(100,3))
    
    def step(self):
    
        self.optimizer.zero_grad()
        output = self.model( self.data )[ 0 ]
        response_map = output.clone().detach().numpy()

        output = output.permute( 1, 2, 0 ).contiguous().view( -1, self.nChannel )

        outputHP = output.reshape( (self.im.shape[0], self.im.shape[1], self.nChannel) )
        HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
        HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
        lhpy = self.loss_hpy(HPy,self.HPy_target)
        lhpz = self.loss_hpz(HPz,self.HPz_target)

        ignore, target = torch.max( output, 1 )
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))
        
        loss = self.stepsize_sim * self.loss_fn(output, target) + self.stepsize_con * (lhpy + lhpz)
            
        loss.backward()
        self.optimizer.step()

        print (' label num :', nLabels, ' | loss :', loss.item())

        if nLabels <= self.minLabels:
            print ("nLabels", nLabels, "reached minLabels", self.minLabels, ".")

        im_target_rgb = np.array([self.label_colours[ c % self.nChannel ] for c in im_target])
        if not self.dim3:
            return loss.item(), im_target_rgb.reshape( ( *self.im.shape, 3) ).astype( np.uint8 ), im_target.reshape(self.im.shape), response_map, nLabels
        else:
            return loss.item(), im_target_rgb.reshape( *self.im.shape ).astype( np.uint8 ), im_target.reshape((self.im.shape[0], self.im.shape[1])), response_map, nLabels

    def run(self, im, eps=1e-7, n_iter=0, minLabel_conv=True, minLabel_return=True, **kwargs):
        '''
        Runs the iterative process of clustering

        im  - np array (h, w, c)
        eps - convergence criteria

        Yields statistics for each iteration
        '''
        if n_iter==0:
            n_iter=self.maxIters
        prev_loss = 10000
        prev_labels = None
        prev_membership = None
        for c in range(n_iter):
            loss, im, labels, membership, nrLabels = self.step()
            membership = membership.reshape(membership.shape[0], membership.shape[1]*membership.shape[2]).T
            label_set = np.delete(np.array([x for x in range(self.nChannel)]), np.unique(labels))
            
            membership = np.delete(membership, label_set, axis=1)

            label_conv = False
            if minLabel_conv:
                if nrLabels==self.minLabels:
                    label_conv=True
                elif nrLabels<self.minLabels:
                    return {"im":im,"labels":labels, "membership":membership}, True
            else:
                label_conv=True

            if abs(prev_loss-loss)<eps and label_conv:
                return {"im":im,"labels":labels, "membership":membership}, True
            else:
                prev_loss=loss
            prev_labels = np.copy(labels)
            prev_membership = np.copy(membership)

            yield {"im":im,"labels":labels, "membership":membership}, False
        

if __name__ == "__main__":
    dfc = DFC(minLabels=10, nChannel=100, nConv=2, lr=0.01, stepsize_con=5)

    im = cv2.imread('PCiDS/sFCM/74.jpeg')

    h, w = 95, 95
    y, x = (im.shape[0] - h)//2, (im.shape[1] - w)//2

    im = im[y:y + h, x: x + w]

    dfc.initialize_clustering(im)

    for i in range(0, 100):
        im, labels, r_map, n_labels = dfc.step()
        cv2.imshow('{}'.format(i), im)
        cv2.waitKey(10)
