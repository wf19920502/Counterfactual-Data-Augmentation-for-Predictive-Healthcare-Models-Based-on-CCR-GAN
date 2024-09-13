'''
基础gan
'''
import  torch
from torch import nn
#https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py

class Generator(nn.Module):
    '''
    生成器，主要由三层网络组成，每次节点数为【128，128，128】
    '''
    def __init__(self,inputDim,generatorDims):
        '''

        :param randomDim: 输入维度
        :param generatorDims: 生成器网络层级
        :param dataType:
        '''
        super().__init__()

        tempDim = inputDim

        self.layer = nn.Sequential()
        for genDim in generatorDims[:-1]:
            layer = nn.Sequential(
            nn.Linear(tempDim,genDim,bias=False),
            nn.BatchNorm1d(genDim,momentum=0.001),
            nn.ReLU()
            )
            tempDim = genDim
            self.layer.append(layer)

        layer = nn.Sequential(
            nn.Linear(tempDim,generatorDims[-1],bias=False),
            nn.BatchNorm1d(generatorDims[-1],momentum=0.001),
            nn.Sigmoid()
            )
        self.layer.append(layer)

    def forward(self,X):
        tempVec = self.layer(X)

        return tempVec


    def getloss(self,fake_Y):
        '''
        :param fake_Y: fake_Y = D(G(x))
        :return: 损失值
        '''
        return -torch.mean(fake_Y)

        pass



class Discriminator(nn.Module):
    '''
    辨别器三层网络': (256, 128, 1)
    '''
    def __init__(self,inputDim,discriminatorDims):
        '''

        :param inputDim:输入维度
        :param discriminatorDims:辨别器网络层级
        :param keepRate:
        '''
        super().__init__()

        self.module = nn.Sequential()
        tempDim = inputDim

        for ind,disDim in enumerate(discriminatorDims):
            self.module.append(nn.Linear(tempDim,disDim))
            self.module.append(nn.ReLU())
            tempDim = disDim

        self.module.append(nn.Linear(tempDim,1))
        self.module.append(nn.Sigmoid())
        pass

    def forward(self,X):
        tempVec = self.module(X)

        return torch.squeeze(tempVec)

    def getloss(self,real_Y,fake_Y):
        '''

        :param real_Y: real_Y = D(x)
        :param fake_Y:fake_Y = D(G(x))
        :return: 损失值
        '''

        loss = torch.mean(fake_Y) - torch.mean(real_Y)
        return loss

        pass
