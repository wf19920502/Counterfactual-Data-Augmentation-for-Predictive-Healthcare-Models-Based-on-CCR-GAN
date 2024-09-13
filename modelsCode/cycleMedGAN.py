import itertools

import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
import os
from AE import AE
from MedModel import *
from setting import STS

__all__=["cycleMedGANtrainer","cycleGANtrainer"]

class cycletrainer():

    def __init__(self,Gclassname,GinputDim,Glayer,Dclassname,DinputDim,Dlayer,Optionfunc,argparam,AEpath=None):

        self.G10 = Gclassname(GinputDim,Glayer)
        self.G01 = Gclassname(GinputDim, Glayer)
        self.D0 = Dclassname(DinputDim,Dlayer)
        self.D1 = Dclassname(DinputDim, Dlayer)

        # print(self.G,self.D)

        self.setOptim(Optionfunc,**argparam)
        self.setAE(AEpath)

        self.GinputDim = GinputDim


    def setOptim(self,optimfunc,**arg):

        self.G_optim = optimfunc(itertools.chain(self.G10.parameters(),self.G01.parameters()),**arg)
        self.D0_optim = optimfunc(self.D0.parameters(), **arg)
        self.D1_optim = optimfunc(self.D1.parameters(), **arg)


    def eval(self):
        self.G10.eval()
        self.G01.eval()

    def setAE(self,modelpath):
        if os.path.exists(modelpath):  # 获取decoder对象
            self.AE = torch.load(modelpath)
            self.AE.eval()

        else:
            raise modelpath + " not exists"

    def getAE(self):

        return self.AE

    def getGinputDim(self):
        return self.GinputDim

    # def stepG(self,loss):
    #     self.G_optim.zero_grad()
    #     loss.backward()
    #     self.G_optim.step()
    #
    # def stepD0(self,loss):
    #     self.D0_optim.zero_grad()
    #     loss.backward()
    #     self.D0_optim.step()
    #
    # def stepD1(self,loss):
    #     self.D1_optim.zero_grad()
    #     loss.backward()
    #     self.D1_optim.step()

    def step(self,optim,loss):
        optim.zero_grad()
        loss.backward()
        optim.step()


    def trainG(self,x0,x1):
        pass

        fake_X0 = self.createData0(x1)
        pred_fake_X = self.D0(fake_X0)
        loss_GAN_10 = self.G10.getloss( pred_fake_X)

        fake_X1 = self.createData1(x0)
        pred_fake_X = self.D1(fake_X1)
        loss_GAN_01 = self.G01.getloss(pred_fake_X)

        cycle0 = self.createData0(fake_X1)
        L_con0 = torch.mean(torch.sum(torch.abs(cycle0 - x0),1))

        cycle1 = self.createData1(fake_X0)
        L_con1 = torch.mean(torch.sum(torch.abs(cycle1 - x1),1))

        return  loss_GAN_10 + loss_GAN_01 +L_con1 + L_con0

    def trainD0(self,x1,x0):
        fake_X = self.createData0(x1)

        pred_X = self.D0(x0)
        pred_fake_X = self.D0(fake_X)
        return self.D0.getloss(pred_X, pred_fake_X)

    def trainD1(self,x0,x1):
        fake_X = self.createData0(x0)

        pred_X = self.D1(x1)
        pred_fake_X = self.D1(fake_X)
        return self.D0.getloss(pred_X, pred_fake_X)


    # def getGrad(self,X,fake_X):
    #     alpha = torch.FloatTensor(X.shape[0], 1).uniform_(0, 1)
    #
    #     alpha = alpha.expand_as(X)
    #     differences = fake_X - X
    #     # print(differences.shape)
    #     interpolates = X + (alpha * differences)
    #     prob_interpolated = self.D(interpolates)
    #     gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolates,
    #                                     grad_outputs=torch.ones(
    #                                         prob_interpolated.size()),
    #                                     create_graph=True, retain_graph=True)[0]
    #     slopes = torch.sqrt(torch.mean(torch.square(gradients), 1))
    #     grad_penalty = torch.mean((slopes - 1) ** 2)
    #
    #     return grad_penalty

    # def getD(self):
    #     return  self.D
    #
    # def getG(self):
    #     return self.G

    def saveG(self,dirpath=STS["modelpath"]):
        torch.save(self.G10, dirpath+self.__name__+"10.path")
        torch.save(self.G01, dirpath + self.__name__ + "01.path")

    def createData0(self,x1):
        pass

    def createData1(self,x0):
        pass

class cycleMedGANtrainer(cycletrainer):
    def __init__(self, Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath=None):
        super().__init__( Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath)

    def createData0(self,x1):
        return self.AE.onlydecoder(self.G10(x1))
        pass

    def createData1(self,x0):
        return self.AE.onlydecoder(self.G01(x0))
        pass

class cycleGANtrainer(cycletrainer):
    def __init__(self, Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath=None):
        super().__init__( Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath)

    def createData0(self,x1):
        return self.G10(x1)

    def createData1(self,x0):
        return self.G01(x0)
