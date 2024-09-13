import numpy as np
import  torch
from torch import nn
import os, re
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset


__all__=["medgan_Generator","medgan_Discriminator","medbgan_Generator",
         "medbgan_Discriminator","emrwgan_Generator","emrwgan_Discriminator","medwgan_Generator","medwgan_Discriminator"]


class medgan_Generator(nn.Module):
    '''
    生成器，主要由三层网络组成，每次节点数为【128，128，128】
    '''
    def __init__(self,randomDim,generatorDims):
        '''

        :param randomDim: 输入维度
        :param generatorDims: 生成器网络层级
        :param dataType:
        '''
        super().__init__()

        tempDim = randomDim

        self.layer = nn.Sequential()
        for genDim in generatorDims[:-1]:
            layer = nn.Sequential(
            nn.Linear(tempDim,genDim,bias=False),
            nn.BatchNorm1d(genDim,momentum=0.009),
            nn.ReLU()
            )
            tempDim = genDim
            self.layer.append(layer)

        layer = nn.Sequential(
            nn.Linear(tempDim,generatorDims[-1],bias=False),
            nn.BatchNorm1d(generatorDims[-1],momentum=0.01),
            nn.Sigmoid()
            )
        self.layer.append(layer)

    def forward(self,X):
        tempVec = X

        tempVec = self.layer[0](tempVec)
        for layer in self.layer[1:]:
            tempVec = layer(tempVec) + tempVec

        return tempVec

    def getloss(self,fake_Y):
        '''
        :param fake_Y: fake_Y = D(G(x))
        :return: 损失值
        '''
        return torch.mean(-torch.log(fake_Y + 1e-12))

        pass



class medgan_Discriminator(nn.Module):
    '''
    辨别器三层网络': (256, 128, 1)
    '''
    def __init__(self,inputDim,discriminatorDims,):
        '''

        :param inputDim:输入维度
        :param discriminatorDims:辨别器网络层级
        :param keepRate:
        '''
        super().__init__()

        self.module = nn.Sequential()
        tempDim = inputDim * 2

        for ind,disDim in enumerate(discriminatorDims):
            self.module.append(nn.Linear(tempDim,disDim))
            self.module.append(nn.ReLU())
            tempDim = disDim

        self.module.append(nn.Linear(tempDim,1))
        self.module.append(nn.Sigmoid())
        pass

    def forward(self,X):
        batchsize = X.shape[0]
        inputMean = torch.tile(torch.mean(X,dim=0,keepdim=True),(batchsize,1))
        tempVec = torch.concat([X,inputMean],dim=1)
        tempVec = self.module(tempVec)

        return torch.squeeze(tempVec)


    def getloss(self,real_Y,fake_Y):
        '''

        :param real_Y: real_Y = D(x)
        :param fake_Y:fake_Y = D(G(x))
        :return: 损失值
        '''

        loss = -torch.mean(torch.log(real_Y + 1e-12)) - torch.mean(torch.log(1 - fake_Y + 1e-12))
        return loss

        pass


class medbgan_Generator(nn.Module):
    '''
    生成器，主要由三层网络组成，每次节点数为【128，128，128】
    '''
    def __init__(self,randomDim,generatorDims):
        '''

        :param randomDim: 输入维度
        :param generatorDims: 生成器网络层级
        :param dataType:
        '''
        super().__init__()

        tempDim = randomDim

        self.layer = nn.Sequential()
        for genDim in generatorDims[:-1]:
            layer = nn.Sequential(
            nn.Linear(tempDim,genDim,bias=False),
            nn.BatchNorm1d(genDim),
            nn.ReLU()
            )
            tempDim = genDim
            self.layer.append(layer)

        layer = nn.Sequential(
            nn.Linear(tempDim,generatorDims[-1],bias=False),
            nn.BatchNorm1d(generatorDims[-1]),
            nn.Tanh()
            )
        self.layer.append(layer)

    def forward(self,X):
        tempVec = X

        tempVec = self.layer[0](tempVec)
        for layer in self.layer[1:]:
            tempVec = layer(tempVec) + tempVec

        return tempVec


    def getloss(self,fake_Y):
        '''
        :param fake_Y: fake_Y = D(G(x))
        :return: 损失值
        '''
        return 0.5 * torch.mean(torch.square(torch.log(fake_Y + 1e-12) - torch.log(1 - fake_Y + 1e-12)))

        pass




class medbgan_Discriminator(nn.Module):
    '''
    辨别器三层网络': (256, 128, 1)
    '''
    def __init__(self,inputDim,discriminatorDims,keepRate=1):
        '''

        :param inputDim:输入维度
        :param discriminatorDims:辨别器网络层级
        :param keepRate:
        '''
        super().__init__()

        self.module = nn.Sequential()
        tempDim = inputDim * 2
        dropRate = 1- keepRate

        for ind,disDim in enumerate(discriminatorDims):
            self.module.append(nn.Linear(tempDim,disDim))
            self.module.append(nn.ReLU())
            self.module.append(nn.Dropout(dropRate))
            tempDim = disDim

        self.module.append(nn.Linear(tempDim,1))
        self.module.append(nn.Sigmoid())
        pass

    def forward(self,X):
        batchsize = X.shape[0]
        inputMean = torch.tile(torch.mean(X,dim=0,keepdim=True),(batchsize,1))
        tempVec = torch.concat([X,inputMean],dim=1)
        tempVec = self.module(tempVec)

        return torch.squeeze(tempVec)


    def getloss(self,real_Y,fake_Y):
        '''

        :param real_Y: real_Y = D(x)
        :param fake_Y:fake_Y = D(G(x))
        :return: 损失值
        '''

        loss = -torch.mean(torch.log(real_Y + 1e-12)) - torch.mean(torch.log(1 - fake_Y + 1e-12))
        return loss

        pass



class emrwgan_Generator(nn.Module):
    '''
    生成器，主要由三层网络组成，每次节点数为【128，128，128】
    '''
    def __init__(self,randomDim,generatorDims):
        '''

        :param randomDim: 输入维度
        :param generatorDims: 生成器网络层级
        :param dataType:
        '''
        super().__init__()

        tempDim = randomDim

        self.layer = nn.Sequential()
        for genDim in generatorDims[:-1]:
            layer = nn.Sequential(
            nn.Linear(tempDim,genDim,bias=False),
            nn.BatchNorm1d(genDim),
            nn.ReLU()
            )
            tempDim = genDim
            self.layer.append(layer)

        layer = nn.Sequential(
            nn.Linear(tempDim,generatorDims[-1],bias=False),
            # nn.BatchNorm1d(generatorDims[-1]),
            nn.Sigmoid()
            )
        self.layer.append(layer)


    def forward(self,X):
        tempVec = X

        tempVec = self.layer[0](tempVec)
        for layer in self.layer[1:]:
            tempVec = layer(tempVec) + tempVec

        return tempVec

    def getloss(self,fake_Y):
        '''
        :param fake_Y: fake_Y = D(G(x))
        :return: 损失值
        '''
        return -torch.mean(fake_Y)

        pass



class emrwgan_Discriminator(nn.Module):
    '''
    辨别器三层网络': (256, 128, 1)
    '''
    def __init__(self,inputDim,discriminatorDims,keepRate=1):
        '''

        :param inputDim:输入维度
        :param discriminatorDims:辨别器网络层级
        :param keepRate:
        '''
        super().__init__()

        self.module = nn.Sequential()
        tempDim = inputDim
        dropRate = 1- keepRate

        for ind,disDim in enumerate(discriminatorDims):
            layer = nn.Sequential(
                nn.Linear(tempDim, disDim),
                nn.LayerNorm(disDim),
                nn.ReLU(),
                nn.Dropout(dropRate)
            )
            self.module.append(layer)
            tempDim = disDim

        self.module.append(nn.Linear(tempDim,1))
        # self.module.append(nn.Sigmoid())
        pass

    def forward(self, X):
        tempVec = X

        tempVec = self.module[0](tempVec)
        for layer in self.module[1:]:
            tempVec = layer(tempVec) + tempVec

        return tempVec

    def getloss(self,real_Y,fake_Y):
        '''

        :param real_Y: real_Y = D(x)
        :param fake_Y:fake_Y = D(G(x))
        :return: 损失值
        '''

        loss = torch.mean(fake_Y) - torch.mean(real_Y)
        return loss

        pass



class medwgan_Generator(nn.Module):
    '''
    生成器，主要由三层网络组成，每次节点数为【128，128，128】
    '''
    def __init__(self,randomDim,generatorDims):
        '''

        :param randomDim: 输入维度
        :param generatorDims: 生成器网络层级
        :param dataType:
        '''
        super().__init__()

        tempDim = randomDim

        self.layer = nn.Sequential()
        for genDim in generatorDims[:-1]:
            layer = nn.Sequential(
            nn.Linear(tempDim,genDim,bias=False),
            nn.BatchNorm1d(genDim),
            nn.ReLU()
            )
            tempDim = genDim
            self.layer.append(layer)

        layer = nn.Sequential(
            nn.Linear(tempDim,generatorDims[-1],bias=False),
            nn.BatchNorm1d(generatorDims[-1]),
            nn.Tanh()
            )
        self.layer.append(layer)

    def forward(self,X):
        tempVec = X

        tempVec = self.layer[0](tempVec)
        for layer in self.layer[1:]:
            tempVec = layer(tempVec) + tempVec

        return tempVec

    def getloss(self,fake_Y):
        '''
        :param fake_Y: fake_Y = D(G(x))
        :return: 损失值
        '''
        return -torch.mean(fake_Y)

        pass



class medwgan_Discriminator(nn.Module):
    '''
    辨别器三层网络': (256, 128, 1)
    '''
    def __init__(self,inputDim,discriminatorDims,keepRate=1):
        '''

        :param inputDim:输入维度
        :param discriminatorDims:辨别器网络层级
        :param keepRate:
        '''
        super().__init__()

        self.module = nn.Sequential()
        tempDim = inputDim * 2
        dropRate = 1- keepRate

        for ind,disDim in enumerate(discriminatorDims):
            self.module.append(nn.Linear(tempDim,disDim))
            self.module.append(nn.ReLU())
            self.module.append(nn.Dropout(dropRate))
            tempDim = disDim

        self.module.append(nn.Linear(tempDim,1))
        # self.module.append(nn.Sigmoid())
        pass

    def forward(self,X):
        batchsize = X.shape[0]
        inputMean = torch.tile(torch.mean(X,dim=0,keepdim=True),(batchsize,1))
        tempVec = torch.concat([X,inputMean],dim=1)
        tempVec = self.module(tempVec)

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
