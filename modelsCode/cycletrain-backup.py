import itertools

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data
import os
from AE import AE
from cyclesetting import STS
from cycleModel import  *
from EMRRGAN import *

__all__=["cycleMedGANtrainer","cycleGANtrainer","cycleRGANConsistTrainer"]

def resize_tensors(fake_X, X):
    num_rows_fake_X, num_cols_fake_X = fake_X.size()
    num_rows_X, num_cols_X = X.size()

    if num_rows_fake_X < num_rows_X:
        fake_X = torch.cat([fake_X] * (num_rows_X // num_rows_fake_X) +
                           [fake_X[:num_rows_X % num_rows_fake_X]], dim=0)

    elif num_rows_fake_X > num_rows_X:
        X = torch.cat([X] * (num_rows_fake_X // num_rows_X) +
                   [X[:num_rows_fake_X % num_rows_X]], dim=0)

    return fake_X, X

def onehot_softmax(logits, dim=-1, dtype=None):
    y_soft = F.softmax(logits, dim=dim, dtype=dtype)
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret

def sampling(X, Cols=[], active= False):
    #active== False do nothing;active== gumbel do gumbel_softmax;active== onehot do onehot_softmax;
    if active:
        sampled_columns = []
        current_col_index = 0
        if Cols:
            for col_type, col_size in Cols:
                current_col_range = slice(current_col_index, current_col_index + col_size)
                current_col_index += col_size
                current_col = X[:, current_col_range]
                if col_type == "D":# 处理离散型变量
                    if active=='gumbel':# 获取离散型变量的列，并应用 Gumbel Softmax 采样
                        current_col = F.gumbel_softmax(current_col, tau=1, hard=True, eps=1e-10)
                    else:#active=='onehot':
                        current_col = onehot_softmax(current_col)
                elif col_type == "C":# 处理连续型变量
                    pass# 获取连续型变量的列，无需额外处理
                sampled_columns.append(current_col) 
            sampled_X = torch.cat(sampled_columns, dim=1) # 将处理过的列拼接在一起
        else:
            sampled_X = X
    else:
        sampled_X = X
    return sampled_X

class cycletrainer():

    def __init__(self,Gclassname,GinputDim,Glayer,Dclassname,DinputDim,Dlayer,Optionfunc,argparam,AEpath=None,superarg=None):

        self.G10 = Gclassname(GinputDim,Glayer)
        self.G01 = Gclassname(GinputDim, Glayer)
        self.D0 = Dclassname(DinputDim,Dlayer)
        self.D1 = Dclassname(DinputDim, Dlayer)
        self.alph,self.bate,self.gamm = superarg

        # print(self.G10,self.D0)

        self.setOptim(Optionfunc,**argparam)
        if AEpath is not None:
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

    def stepG(self,loss):
        self.step(self.G_optim,loss)

    def stepD0(self,loss):
        self.step(self.D0_optim,loss)

    def stepD1(self,loss):
        self.step(self.D1_optim,loss)

    def step(self,optim,loss):
        optim.zero_grad()
        loss.backward()
        optim.step()


    def trainG(self,x0,x1):
        fake_X0 = self.createData0(x1)
        pred_fake_X0 = self.D0(fake_X0)
        loss_GAN_10 = self.G10.getloss( pred_fake_X0)

        fake_X1 = self.createData1(x0)
        pred_fake_X1 = self.D1(fake_X1)
        loss_GAN_01 = self.G01.getloss(pred_fake_X1)

        cycle0 = self.createData0(fake_X1)
        L_con0 = torch.mean(torch.sum(torch.abs(cycle0 - x0),1))

        cycle1 = self.createData1(fake_X0)
        L_con1 = torch.mean(torch.sum(torch.abs(cycle1 - x1),1))

        return  loss_GAN_10 + loss_GAN_01 +L_con1 + L_con0

    def trainD0(self,x0,x1):
        fake_X = self.createData0(x1)

        pred_X = self.D0(x0)
        pred_fake_X = self.D0(fake_X)
        return self.D0.getloss(pred_X, pred_fake_X)

    def trainD1(self,x0,x1):
        fake_X = self.createData1(x0)

        pred_X = self.D1(x1)
        pred_fake_X = self.D1(fake_X)
        return self.D1.getloss(pred_X, pred_fake_X)


    def getGrad(self,X,fake_X,D):
        fake_X, X = resize_tensors(fake_X, X)
        alpha = torch.FloatTensor(X.shape[0], 1).uniform_(0, 1)

        alpha = alpha.expand_as(X)
        differences = fake_X - X
        # print(differences.shape)
        interpolates = X + (alpha * differences)
        prob_interpolated = D(interpolates)
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolates,
                                        grad_outputs=torch.ones(
                                            prob_interpolated.size()),
                                        create_graph=True, retain_graph=True)[0]
        slopes = torch.sqrt(torch.mean(torch.square(gradients), 1))
        grad_penalty = torch.mean((slopes - 1) ** 2)

        return grad_penalty

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
    
    def sampleData0(self,x1):
        return sampling(self.createData0(x1), Cols, active)

    def sampleData1(self,x0):
        return sampling(self.createData1(x0), Cols, active)

    def train(self, x0, x1):
        lossvec = []
        # 训练D0
        loss = self.trainD0(x0, x1)
        self.stepD0(loss)
        lossvec.append(loss.item())

        # 训练D1
        loss = self.trainD1(x0, x1)
        self.stepD1(loss)
        lossvec.append(loss.item())

        # 训练G
        for loop in range(2):
            loss = self.trainG(x0, x1)
            self.stepG(loss)
        lossvec.append(loss.item())
        return lossvec

    def syndata(self,data0,data1):

        self.eval()
        value0 = self.createData0(data1)
        value0 = value0.cpu().detach().numpy()
        zeros = np.zeros((value0.shape[0], 1))
        value0 = np.concatenate((value0, zeros), axis=1)

        value1 = self.createData1(data0)
        value1 = value1.cpu().detach().numpy()
        ones = np.ones((value1.shape[0], 1))
        value1 = np.concatenate((value1,ones),axis=1)

        return np.concatenate((value0,value1),axis=0)


class cycleMedGANtrainer(cycletrainer):
    def __init__(self, Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath=None,superarg=None):
        super().__init__( Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath,superarg)

    def createData0(self,x1):
        return self.AE.onlydecoder(self.G10(x1))
        pass

    def createData1(self,x0):
        return self.AE.onlydecoder(self.G01(x0))
        pass

class cycleGANtrainer(cycletrainer):
    def __init__(self, Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath=None,superarg=None):
        super().__init__( Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath,superarg)

    def createData0(self,x1):
        return (self.G10(x1)+1)/2

    def createData1(self,x0):
        return (self.G01(x0)+1)/2

class RGANtrainer(cycletrainer):
    def __init__(self, Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath=None,superarg=None,Classifier=None):
        super().__init__( Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath,superarg)

        self.setClassifier(Classifier)


    def createData0(self,x1):
        return torch.clamp(self.G10(x1) + x1, min=0, max=1)

    def createData1(self,x0):
        return torch.clamp(self.G01(x0) + x0, min=0, max=1)

    def createG0(self,x):
        return self.G10(x)

    def createG1(self,x):
        return self.G01(x)

    def setClassifier(self,Classifier):
        if os.path.exists(Classifier):  # 获取decoder对象
            self.classifier = torch.load(Classifier)
            self.classifier.eval()
        else:
            raise Exception(f"{Classifier} is not exist")

    def trainG(self,x0,x1):

        fake_X0 = self.createData0(x1)
        pred_fake_X0 = self.D0(fake_X0)
        loss_GAN_10 = self.G10.getloss( pred_fake_X0)

        fake_X1 = self.createData1(x0)
        pred_fake_X1 = self.D1(fake_X1)
        loss_GAN_01 = self.G01.getloss(pred_fake_X1)

        y_hat0 = self.classifier(fake_X0)
        C0_loss = self.classifier.getloss(y_hat0, torch.zeros_like(y_hat0))

        y_hat1 = self.classifier(fake_X1)
        C1_loss = self.classifier.getloss(y_hat1,torch.ones_like(y_hat1))

        X0_G = self.createG0(x1)
        L1_10 = torch.mean(torch.sum(torch.abs(X0_G),1))
        L2_10 = torch.mean(torch.sum(torch.square(X0_G),1))

        X1_G = self.createG1(x0)
        L1_01 = torch.mean(torch.sum(torch.abs(X1_G),1))
        L2_01 = torch.mean(torch.sum(torch.square(X1_G),1))

        return  loss_GAN_10 + loss_GAN_01 +C0_loss + C1_loss + self.alph *(L1_10+L1_01)+ self.bate * (L2_10 +L2_01)


class RGAN_ncTrainer(RGANtrainer):
    def __init__(self, Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath=None,superarg=None,
                 Classifier=None):
        super().__init__( Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath,superarg,Classifier)


    def trainG(self, x0, x1):
        fake_X0 = self.createData0(x1)
        pred_fake_X0 = self.D0(fake_X0)
        loss_GAN_10 = self.G10.getloss(pred_fake_X0)

        fake_X1 = self.createData1(x0)
        pred_fake_X1 = self.D1(fake_X1)
        loss_GAN_01 = self.G01.getloss(pred_fake_X1)

        X0_G = self.G10(x1)
        L1_10 = torch.mean(torch.sum(torch.abs(X0_G), 1))
        L2_10 = torch.mean(torch.sum(torch.square(X0_G), 1))

        X1_G = self.G01(x0)
        L1_01 = torch.mean(torch.sum(torch.abs(X1_G), 1))
        L2_01 = torch.mean(torch.sum(torch.square(X1_G), 1))

        return loss_GAN_10 + loss_GAN_01 + self.alph * (L1_10 + L1_01) + self.bate * (L2_10 + L2_01)

class cycleRGANTrainer(RGANtrainer):
    def __init__(self, Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath=None,superarg=None,
                 Classifier=None):
        super().__init__( Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath,superarg,Classifier)

        self.Dloss_f = lambda real_Y, fake_Y,rfake_Y:-2*torch.mean(torch.log(real_Y + 1e-12)) - torch.mean(torch.log(1 - fake_Y + 1e-12)) - torch.mean(torch.log(1 - rfake_Y + 1e-12))


    def trainD0(self,x0,x1):
        fake_X = self.createData0(x1)

        pred_X = self.D0(x0)
        pred_fake_X = self.D0(fake_X)
        rpred_fake_X = self.D0(self.createData0(self.createData1(x0)))
        return self.Dloss_f(pred_X,pred_fake_X,rpred_fake_X)

    def trainD1(self,x0,x1):
        fake_X = self.createData1(x0)

        pred_X = self.D1(x1)
        pred_fake_X = self.D1(fake_X)
        rpred_fake_X = self.D1(self.createData1(self.createData0(x1)))
        return self.Dloss_f(pred_X,pred_fake_X,rpred_fake_X)


    def trainG(self,x0,x1):

        fake_X0 = self.createData0(x1)
        pred_fake_X = self.D0(fake_X0)
        loss_GAN_10 = self.G10.getloss( pred_fake_X)

        fake_X1 = self.createData1(x0)
        pred_fake_X = self.D1(fake_X1)
        loss_GAN_01 = self.G01.getloss(pred_fake_X)

        y_hat = self.classifier(fake_X0)
        C0_loss = self.classifier.getloss(y_hat, torch.zeros_like(y_hat))

        y_hat = self.classifier(fake_X1)
        C1_loss = self.classifier.getloss(y_hat,torch.ones_like(y_hat))

        X0_G = self.createG0(x1)
        L1_10 = torch.mean(torch.sum(torch.abs(X0_G),1))
        L2_10 = torch.mean(torch.sum(torch.square(X0_G),1))

        X1_G = self.createG1(x0)
        L1_01 = torch.mean(torch.sum(torch.abs(X1_G),1))
        L2_01 = torch.mean(torch.sum(torch.square(X1_G),1))

        cyc_loss_GAN_10 = self.G10.getloss(self.D0(self.createData0(self.createData1(x0).detach())))
        cyc_loss_GAN_01 = self.G01.getloss(self.D1(self.createData1(self.createData0(x1).detach())))

        return  loss_GAN_10 + loss_GAN_01 +C0_loss + C1_loss + cyc_loss_GAN_10 + cyc_loss_GAN_01+ self.alph *(L1_10+L1_01)+ self.bate * (L2_10 +L2_01)


class cycleRGAN_ncTrainer(cycleRGANTrainer):
    def __init__(self, Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath=None,superarg=None,
                 Classifier=None):
        super().__init__( Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath,superarg,Classifier)

    def trainG(self,x0,x1):

        fake_X0 = self.createData0(x1)
        pred_fake_X = self.D0(fake_X0)
        loss_GAN_10 = self.G10.getloss( pred_fake_X)

        fake_X1 = self.createData1(x0)
        pred_fake_X = self.D1(fake_X1)
        loss_GAN_01 = self.G01.getloss(pred_fake_X)

        X0_G = self.createG0(x1)
        L1_10 = torch.mean(torch.sum(torch.abs(X0_G),1))
        L2_10 = torch.mean(torch.sum(torch.square(X0_G),1))

        X1_G = self.createG1(x0)
        L1_01 = torch.mean(torch.sum(torch.abs(X1_G),1))
        L2_01 = torch.mean(torch.sum(torch.square(X1_G),1))

        cyc_loss_GAN_10 = self.G10.getloss(self.D0(self.createData0(self.createData1(x0).detach())))
        cyc_loss_GAN_01 = self.G01.getloss(self.D1(self.createData1(self.createData0(x1).detach())))

        return  loss_GAN_10 + loss_GAN_01  + cyc_loss_GAN_10 + cyc_loss_GAN_01+ self.alph *(L1_10+L1_01)+ self.bate * (L2_10 +L2_01)



class cycleRGANConsistTrainer(RGANtrainer):
    def __init__(self, Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath=None,superarg=None,
                 Classifier=None):
        super().__init__( Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath,superarg,Classifier)


    def trainG(self,x0,x1):

        fake_X0 = self.createData0(x1)
        pred_fake_X = self.D0(fake_X0)
        loss_GAN_10 = self.G10.getloss( pred_fake_X)

        fake_X1 = self.createData1(x0)
        pred_fake_X = self.D1(fake_X1)
        loss_GAN_01 = self.G01.getloss(pred_fake_X)

        y_hat = self.classifier(fake_X0)
        C0_loss = self.classifier.getloss(y_hat, torch.zeros_like(y_hat))

        y_hat = self.classifier(fake_X1)
        C1_loss = self.classifier.getloss(y_hat,torch.ones_like(y_hat))

        cycle0 = self.createData0(fake_X1)
        L_con0 = torch.mean(torch.sum(torch.abs(cycle0 - x0),1))

        cycle1 = self.createData1(fake_X0)
        L_con1 = torch.mean(torch.sum(torch.abs(cycle1 - x1),1))

        X0_G = self.createG0(x1)
        L1_10 = torch.mean(torch.sum(torch.abs(X0_G),1))
        L2_10 = torch.mean(torch.sum(torch.square(X0_G),1))

        X1_G = self.createG1(x0)
        L1_01 = torch.mean(torch.sum(torch.abs(X1_G),1))
        L2_01 = torch.mean(torch.sum(torch.square(X1_G),1))


        return  loss_GAN_10 + loss_GAN_01 +C0_loss + C1_loss + L_con0 + L_con1 + self.alph *(L1_10+L1_01)+ self.bate * (L2_10 +L2_01)


class Dual_RGANTrainer(RGANtrainer):
    def __init__(self, Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath=None,superarg=None,
                 Classifier=None,DualVAEpath=None):
        super().__init__( Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath,superarg,Classifier)
        self.VAE:DualVAE = torch.load(DualVAEpath)
        self.VAE.eval()
    pass

    def getsplitdata(self,x):
        xd = x[:,:self.VAE.D_infeature]
        xc = x[:, self.VAE.D_infeature:]

        return xd,xc

    def createData0(self,x1):
        xd, xc = self.getsplitdata(x1)
        *_,z = self.VAE.encoder(xd,xc)

        z = self.G10(z) + z
        *_,x_hat =self.VAE.decoder(z)
        return x_hat

    def createData1(self,x0):
        xd, xc = self.getsplitdata(x0)
        *_, z = self.VAE.encoder(xd, xc)

        z = self.G01(z) + z
        *_,x_hat = self.VAE.decoder(z)
        return x_hat

    def createG0(self,x):
        xd, xc = self.getsplitdata(x)
        *_, z = self.VAE.encoder(xd, xc)

        return self.G10(z)

    def createG1(self,x):
        xd, xc = self.getsplitdata(x)
        *_, z = self.VAE.encoder(xd, xc)

        return self.G01(z)


class Dual_cycleRGANTrainer(Dual_RGANTrainer):
    def __init__(self, Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath=None,superarg=None,
                 Classifier=None,DualVAEpath=None):
        super().__init__( Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath,superarg,Classifier,DualVAEpath)

        self.Dloss_f = lambda real_Y, fake_Y, rfake_Y: -2 * torch.mean(torch.log(real_Y + 1e-12)) - torch.mean(
            torch.log(1 - fake_Y + 1e-12)) - torch.mean(torch.log(1 - rfake_Y + 1e-12))

    def trainD0(self,x0,x1):
        fake_X = self.createData0(x1)

        pred_X = self.D0(x0)
        pred_fake_X = self.D0(fake_X)
        rpred_fake_X = self.D0(self.createData0(self.createData1(x0)))
        return self.Dloss_f(pred_X,pred_fake_X,rpred_fake_X)

    def trainD1(self,x0,x1):
        fake_X = self.createData1(x0)

        pred_X = self.D1(x1)
        pred_fake_X = self.D1(fake_X)
        rpred_fake_X = self.D1(self.createData1(self.createData0(x1)))
        return self.Dloss_f(pred_X,pred_fake_X,rpred_fake_X)


    def trainG(self,x0,x1):

        fake_X0 = self.createData0(x1)
        pred_fake_X = self.D0(fake_X0)
        loss_GAN_10 = self.G10.getloss( pred_fake_X)

        fake_X1 = self.createData1(x0)
        pred_fake_X = self.D1(fake_X1)
        loss_GAN_01 = self.G01.getloss(pred_fake_X)

        y_hat = self.classifier(fake_X0)
        C0_loss = self.classifier.getloss(y_hat, torch.zeros_like(y_hat))

        y_hat = self.classifier(fake_X1)
        C1_loss = self.classifier.getloss(y_hat,torch.ones_like(y_hat))

        X0_G = self.createG0(x1)
        L1_10 = torch.mean(torch.sum(torch.abs(X0_G),1))
        L2_10 = torch.mean(torch.sum(torch.square(X0_G),1))

        X1_G = self.createG1(x0)
        L1_01 = torch.mean(torch.sum(torch.abs(X1_G),1))
        L2_01 = torch.mean(torch.sum(torch.square(X1_G),1))

        cyc_loss_GAN_10 = self.G10.getloss(self.D0(self.createData0(self.createData1(x0).detach())))
        cyc_loss_GAN_01 = self.G01.getloss(self.D1(self.createData1(self.createData0(x1).detach())))

        return  loss_GAN_10 + loss_GAN_01 +C0_loss + C1_loss + cyc_loss_GAN_10 + cyc_loss_GAN_01+ self.alph *(L1_10+L1_01)+ self.bate * (L2_10 +L2_01)


class Dual_cycleRGANConsistTrainer(Dual_RGANTrainer):
    def __init__(self, Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath=None,superarg=None,
                 Classifier=None,DualVAEpath=None):
        super().__init__( Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath,superarg,Classifier,DualVAEpath)



    def trainG(self,x0,x1):

        fake_X0 = self.createData0(x1)
        pred_fake_X = self.D0(fake_X0)
        loss_GAN_10 = self.G10.getloss( pred_fake_X)

        fake_X1 = self.createData1(x0)
        pred_fake_X = self.D1(fake_X1)
        loss_GAN_01 = self.G01.getloss(pred_fake_X)

        y_hat = self.classifier(fake_X0)
        C0_loss = self.classifier.getloss(y_hat, torch.zeros_like(y_hat))

        y_hat = self.classifier(fake_X1)
        C1_loss = self.classifier.getloss(y_hat,torch.ones_like(y_hat))

        cycle0 = self.createData0(fake_X1)
        L_con0 = torch.mean(torch.sum(torch.abs(cycle0 - x0),1))

        cycle1 = self.createData1(fake_X0)
        L_con1 = torch.mean(torch.sum(torch.abs(cycle1 - x1),1))

        X0_G = self.createG0(x1)
        L1_10 = torch.mean(torch.sum(torch.abs(X0_G),1))
        L2_10 = torch.mean(torch.sum(torch.square(X0_G),1))

        X1_G = self.createG1(x0)
        L1_01 = torch.mean(torch.sum(torch.abs(X1_G),1))
        L2_01 = torch.mean(torch.sum(torch.square(X1_G),1))


        return  loss_GAN_10 + loss_GAN_01 +C0_loss + C1_loss + L_con0 + L_con1 + self.alph *(L1_10+L1_01)+ self.bate * (L2_10 +L2_01)



class EMR_RGANTrainer(RGANtrainer):
    def __init__(self, Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath=None,superarg=None,
                 Classifier=None,EMRAEpath=None):
        super().__init__( Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath,superarg,Classifier)
        self.EMRAE:AE = torch.load(EMRAEpath)
        self.EMRAE.eval()
    pass

    def createData0(self,x1):
        z = self.EMRAE.encoder(x1)

        z = self.G10(z) + z
        x_hat =self.EMRAE.decoder(z)
        return x_hat

    def createData1(self,x0):
        z = self.EMRAE.encoder(x0)

        z = self.G01(z) + z
        x_hat = self.EMRAE.decoder(z)
        return x_hat

    def createG0(self,x):
        z = self.EMRAE.encoder(x)

        return self.G10(z)

    def createG1(self,x):
        z = self.EMRAE.encoder(x)

        return self.G01(z)

    def trainD0(self,x0,x1):
        fake_X = self.createData0(x1)

        pred_X = self.D0(x0)
        pred_fake_X = self.D0(fake_X)
        return self.D0.getloss(pred_X, pred_fake_X) + super().getGrad(x0, fake_X,self.D0)

    def trainD1(self,x0,x1):
        fake_X = self.createData1(x0)

        pred_X = self.D1(x1)
        pred_fake_X = self.D1(fake_X)
        return self.D1.getloss(pred_X, pred_fake_X) + super().getGrad(x1, fake_X,self.D1)


class EMR_cycleRGANTrainer(EMR_RGANTrainer):
    def __init__(self, Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath=None,superarg=None,
                 Classifier=None,EMRAEpath=None):
        super().__init__( Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath,superarg,Classifier,EMRAEpath)

        self.Dloss_f = lambda real_Y, fake_Y, rfake_Y: - torch.mean(real_Y ) + torch.mean(fake_Y)+ torch.mean(rfake_Y)

    def trainD0(self,x0,x1):
        fake_X = self.createData0(x1)
        rfake_X = self.createData0(self.createData1(x0))

        pred_X = self.D0(x0)
        pred_fake_X = self.D0(fake_X)
        rpred_fake_X = self.D0(rfake_X)

        D_L = self.Dloss_f(pred_X, pred_fake_X, rpred_fake_X)

        grad_penalty1 = super().getGrad(x0, fake_X,self.D0)
        grad_penalty2 = super().getGrad(x0, rfake_X,self.D0)
        return D_L + grad_penalty1 + grad_penalty2

    def trainD1(self,x0,x1):
        fake_X = self.createData1(x0)
        rfake_X =self.createData1(self.createData0(x1))

        pred_X = self.D1(x1)
        pred_fake_X = self.D1(fake_X)
        rpred_fake_X = self.D1(rfake_X)
        D_L = self.Dloss_f(pred_X,pred_fake_X,rpred_fake_X)

        grad_penalty1  = super().getGrad(x1, fake_X,self.D1)
        grad_penalty2 = super().getGrad(x1, rfake_X,self.D1)
        return D_L +grad_penalty1 +grad_penalty2



    def trainG(self,x0,x1):

        fake_X0 = self.createData0(x1)
        pred_fake_X = self.D0(fake_X0)
        loss_GAN_10 = self.G10.getloss( pred_fake_X)

        fake_X1 = self.createData1(x0)
        pred_fake_X = self.D1(fake_X1)
        loss_GAN_01 = self.G01.getloss(pred_fake_X)

        y_hat = self.classifier(fake_X0)
        C0_loss = self.classifier.getloss(y_hat, torch.zeros_like(y_hat))

        y_hat = self.classifier(fake_X1)
        C1_loss = self.classifier.getloss(y_hat,torch.ones_like(y_hat))

        X0_G = self.createG0(x1)
        L1_10 = torch.mean(torch.sum(torch.abs(X0_G),1))
        L2_10 = torch.mean(torch.sum(torch.square(X0_G),1))

        X1_G = self.createG1(x0)
        L1_01 = torch.mean(torch.sum(torch.abs(X1_G),1))
        L2_01 = torch.mean(torch.sum(torch.square(X1_G),1))

        cyc_loss_GAN_10 = self.G10.getloss(self.D0(self.createData0(self.createData1(x0).detach())))
        cyc_loss_GAN_01 = self.G01.getloss(self.D1(self.createData1(self.createData0(x1).detach())))

        return  loss_GAN_10 + loss_GAN_01 +C0_loss + C1_loss + cyc_loss_GAN_10 + cyc_loss_GAN_01+ self.alph *(L1_10+L1_01)+ self.bate * (L2_10 +L2_01)

class EMR_cycleRGANConsistTrainer(EMR_RGANTrainer):
    def __init__(self, Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath=None,superarg=None,
                 Classifier=None,EMRAEpath=None):
        super().__init__( Gclassname, GinputDim, Glayer, Dclassname, DinputDim, Dlayer, Optionfunc, argparam, AEpath,superarg,Classifier,EMRAEpath)

    def trainG(self,x0,x1):

        fake_X0 = self.createData0(x1)
        pred_fake_X = self.D0(fake_X0)
        loss_GAN_10 = self.G10.getloss( pred_fake_X)

        fake_X1 = self.createData1(x0)
        pred_fake_X = self.D1(fake_X1)
        loss_GAN_01 = self.G01.getloss(pred_fake_X)

        y_hat = self.classifier(fake_X0)
        C0_loss = self.classifier.getloss(y_hat, torch.zeros_like(y_hat))

        y_hat = self.classifier(fake_X1)
        C1_loss = self.classifier.getloss(y_hat,torch.ones_like(y_hat))

        cycle0 = self.createData0(fake_X1)
        L_con0 = torch.mean(torch.sum(torch.abs(cycle0 - x0),1))

        cycle1 = self.createData1(fake_X0)
        L_con1 = torch.mean(torch.sum(torch.abs(cycle1 - x1),1))

        X0_G = self.createG0(x1)
        L1_10 = torch.mean(torch.sum(torch.abs(X0_G),1))
        L2_10 = torch.mean(torch.sum(torch.square(X0_G),1))

        X1_G = self.createG1(x0)
        L1_01 = torch.mean(torch.sum(torch.abs(X1_G),1))
        L2_01 = torch.mean(torch.sum(torch.square(X1_G),1))


        return  loss_GAN_10 + loss_GAN_01 +C0_loss + C1_loss + L_con0 + L_con1 + self.alph *(L1_10+L1_01)+ self.bate * (L2_10 +L2_01)


def getChart(x,dloss,name):

    y = []
    if isinstance(dloss[0],list):
        for i in range(len(dloss[0])):
            tmp = [j[i] for j in dloss]
            y.append(tmp)
    else:
        y.append(dloss)

    savedir = "./picture"
    import matplotlib.pyplot as plt
    plt.clf()
    plt.title(name  )
    for lable,i in enumerate(y):
        plt.plot(x,i,label=str(lable))
    plt.legend()
    if os.path.exists(savedir) == False:
        os.makedirs(savedir)
    if os.path.exists(os.path.join(savedir,name.replace(".","_")+".png")):
        os.remove(os.path.join(savedir,name.replace(".","_")+".png"))
    plt.savefig(os.path.join(savedir,name.replace(".","_")))

def train(model:cycletrainer,data0,data1,epochs=100,createsize=3000):
    #
    # Gl = []
    # Dl = []
    # epochl = []
    for epoch in range(epochs):
        loss = model.train(data0,data1)

        if epoch %20 == 19:
            print(f"epochs = {epoch}",loss)
    #做损失图
    # getChart(epochl,Dl,model.__class__.__name__+"_D")
    # getChart(epochl, Gl, model.__class__.__name__ + "_G")


    data = model.syndata(data0,data1)

    np.save(STS["dataSavePath"]+model.__class__.__name__,data)
    print(STS["dataSavePath"]+model.__class__.__name__)

def trainAE(arg):
    filename = arg["filename"]
    lr = arg["lr"]
    epochs = arg["epochs"]
    Elayer = arg["Elayer"]
    Dlayer = arg["Dlayer"]
    data = np.load(filename)
    data = data[:,:STS["Label"]]
    inputdim = data.shape[1]

    ae = AE(inputdim,Elayer,Dlayer)

    ae_optim = torch.optim.RMSprop(ae.parameters(), lr=lr, weight_decay=0.001, eps=1e-07)
    print(ae)
    data = torch.tensor(data,dtype=torch.float)
    torch_dataset = Data.TensorDataset(data)
    loader = Data.DataLoader(dataset=torch_dataset,shuffle=True,batch_size=100)

    for epoch in range(epochs):
        lossvec = []
        for step ,batch in enumerate(loader):
            x_hat = ae(batch[0])
            loss = ae.getloss(batch[0],x_hat,type="Entropy")
            lossvec.append(loss.item())

            #梯度下降
            ae_optim.zero_grad()
            loss.backward()
            ae_optim.step()
        if epoch %10 == 9:
            print(f"epoch = {epoch},loss = {np.mean(lossvec)}")

    torch.save(ae,STS["AEpath"])
    np.save(STS["dataSavePath"]+"cycleAE",ae(data).cpu().detach().numpy())

def trainDualVAE(args):
    data = np.load(args["filename"])

    data = torch.tensor(data,dtype=torch.float)
    Xd = data[:,:STS["D_size"]]
    Xc = data[:,STS["D_size"]:-1]

    torch_dataset = Data.TensorDataset(Xd,Xc)

    loader = Data.DataLoader(dataset=torch_dataset,shuffle=True,batch_size=200)

    model = DualVAE(**args["modelarg"])
    optim = args["Optionfunc"](model.parameters(),**args["argparam"])

    for epoch in range(args["epochs"]):
        lossvec = []
        for xd,xc in loader:
            d_mu,d_log_var,zd,c_mu,c_log_var,zc,Xd_hat, Xc_hat = model(xd,xc)
            loss = model.getloss(xd,xc,d_mu,d_log_var,zd,c_mu,c_log_var,zc,Xd_hat, Xc_hat)
            optim.zero_grad()
            loss.backward()
            optim.step()
            lossvec.append(loss.item())

        if epoch %20 == 19:
            print(f"epoch = {epoch}, loss = {np.mean(lossvec)}")
            #
            # with torch.no_grad():
            #     Y_hat = model(X)
            #     acc,cmp0,cmp1 ,F1 = model.accuracy(Y_hat,Y)
            #     print(f"accuracy = {acc},cmp0 = {cmp0},cmp1 = {cmp1},F1={F1}")

    torch.save(model,args["DualVAEpath"])

    latent_size = args["modelarg"]["latent_size"]
    z = torch.randn((len(data),latent_size))
    *_,X_hat = model.decoder(z)

    X_hat =torch.cat((X_hat,torch.ones(X_hat.shape[0],1)),dim=1)
    np.save(STS["dataSavePath"] + "DualVAE", X_hat.cpu().detach().numpy())
    print(STS["dataSavePath"] + "DualVAE")

    *_,Xd_hat, Xc_hat = model(Xd, Xc)
    X_hat = torch.concat((Xd_hat,Xc_hat),dim=1)

    X_hat = torch.cat((X_hat,data[:,-1].view(X_hat.shape[0],-1)),dim=1)
    np.save(STS["dataSavePath"] + "DualVAE_X", X_hat.cpu().detach().numpy())
    print(STS["dataSavePath"] + "DualVAE_X")
    pass

    pass


def trainSingleVAE(args):
    data = np.load(args["filename"])

    data = torch.tensor(data,dtype=torch.float)


    torch_dataset = Data.TensorDataset(data)

    loader = Data.DataLoader(dataset=torch_dataset,shuffle=True,batch_size=200)

    model = SingleVAE(**args["modelarg"])
    optim = args["Optionfunc"](model.parameters(),**args["argparam"])

    latent_size = args["modelarg"]["latent_size"]

    for epoch in range(args["epochs"]):
        lossvec = []
        for (x,) in loader:
            c_mu,c_log_var,zc, Xc_hat = model(x)
            loss = model.getloss(x,c_mu,c_log_var, Xc_hat)
            optim.zero_grad()
            loss.backward()
            optim.step()
            lossvec.append(loss.item())

        if epoch %20 == 19:
            print(f"epoch = {epoch}, loss = {np.mean(lossvec)}")
            #
            # with torch.no_grad():
            #     Y_hat = model(X)
            #     acc,cmp0,cmp1 ,F1 = model.accuracy(Y_hat,Y)
            #     print(f"accuracy = {acc},cmp0 = {cmp0},cmp1 = {cmp1},F1={F1}")

    # torch.save(model,args["SingleVAE"])
    z = torch.randn((len(data),latent_size))
    np.save(STS["dataSavePath"] + "SingleVAE", model.decoder(z).cpu().detach().numpy())
    print(STS["dataSavePath"] + "SingleVAE")
    *_, Xc_hat = model(data)
    np.save(STS["dataSavePath"] + "SingleVAE_X", Xc_hat.cpu().detach().numpy())
    print(STS["dataSavePath"] + "SingleVAE_X")
    pass

def trainclassifier(args):
    data = np.load(args["filename"])
    data = torch.tensor(data,dtype=torch.float)
    X = data[:,:STS["Label"]]
    Y = data[:,STS["Label"]].long()

    torch_dataset = Data.TensorDataset(X,Y)

    loader = Data.DataLoader(dataset=torch_dataset,shuffle=True,batch_size=200)

    model = ClassifierModel(args["inputDim"],args["layer"])
    optim = args["Optionfunc"](model.parameters(),**args["argparam"])

    for epoch in range(args["epochs"]):
        lossvec = []
        for x,y in loader:
            y_hat = model(x)
            loss = model.getloss(y_hat,y,args["alph"],args["gamm"])
            optim.zero_grad()
            loss.backward()
            optim.step()
            lossvec.append(loss.item())

        if epoch %20 == 19:
            print(f"epoch = {epoch}, loss = {np.mean(lossvec)}")

            with torch.no_grad():
                Y_hat = model(X)
                acc,cmp0,cmp1 ,F1 = model.accuracy(Y_hat,Y)
                print(f"accuracy = {acc},cmp0 = {cmp0},cmp1 = {cmp1},F1={F1}")

    torch.save(model,args["classifierpath"])

def trainEMRAE(arg):
    filename = arg["filename"]
    lr = arg["lr"]
    epochs = arg["epochs"]
    Elayer = arg["Elayer"]
    Dlayer = arg["Dlayer"]
    data = np.load(filename)
    data = data[:,:STS["Label"]]
    inputdim = data.shape[1]

    ae = AE(inputdim,Elayer,Dlayer)

    ae_optim = torch.optim.RMSprop(ae.parameters(), lr=lr, weight_decay=0.001, eps=1e-07)
    print(ae)
    data = torch.tensor(data,dtype=torch.float)
    torch_dataset = Data.TensorDataset(data)
    loader = Data.DataLoader(dataset=torch_dataset,shuffle=True,batch_size=100)

    for epoch in range(epochs):
        lossvec = []
        for step ,batch in enumerate(loader):
            x_hat = ae(batch[0])
            loss = ae.getloss(batch[0],x_hat,type="Entropy")
            lossvec.append(loss.item())

            #梯度下降
            ae_optim.zero_grad()
            loss.backward()
            ae_optim.step()
        if epoch %10 == 9:
            print(f"epoch = {epoch},loss = {np.mean(lossvec)}")

    torch.save(ae,STS["EMRAEpath"])
    print(STS["EMRAEpath"])
    # np.save(STS["dataSavePath"]+"cycleAE",ae(data).cpu().detach().numpy())

def main():
    data = np.load(STS["filename"])


    data = torch.tensor(data, dtype=torch.float)

    data0 = data[data[:,STS["Label"]] == 0]
    data0 = data0[:,:STS["Label"]]
    data1 = data[data[:,STS["Label"]] == 1]
    data1 = data1[:,:STS["Label"]]

#     emrdata0 = data0[:20]
#     emrdata1 = data1[:20]

    # test single trainer
    for trainname in [EMR_cycleRGANTrainer,EMR_RGANTrainer,EMR_cycleRGANConsistTrainer]:  # EMR_RGANTrainer,EMR_cycleRGANTrainer,EMR_cycleRGANConsistTrainer,cycleRGANConsistTrainer
        print("test ing ")
        print(trainname.__name__)
        model = trainname(**STS["models"][trainname.__name__])
        print(model.__class__.__name__)
#         train(model=model, data0=emrdata0, data1=emrdata1, epochs=STS["epoch"], createsize=STS["createsize"])
        train(model=model, data0=data0, data1=data1, epochs=STS["epoch"], createsize=STS["createsize"])
        
    for trainname in [ RGAN_ncTrainer,cycleRGAN_ncTrainer,cycleRGANTrainer,cycleMedGANtrainer,cycleGANtrainer,RGANtrainer,cycleRGANConsistTrainer]:#,Dual_RGANTrainer,Dual_cycleRGANConsistTrainer,Dual_cycleRGANTrainer

        model = trainname(**STS["models"][trainname.__name__])
        print(model.__class__.__name__)
        train(model=model,data0=data0,data1=data1,epochs=STS["epoch"],createsize=STS["createsize"])

    pass
if __name__=="__main__":
    #
    trainEMRAE(STS["models"]["EMRAE"])

    trainAE(STS["models"]["AE"])
    trainclassifier(STS["models"]["ClassifierModel"])
# #     trainDualVAE(STS["models"]["DualVAE"])
#     # trainSingleVAE(STS["models"]["SingleVAE"])
    main()