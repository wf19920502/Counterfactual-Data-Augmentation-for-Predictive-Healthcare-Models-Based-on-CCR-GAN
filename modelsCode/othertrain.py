import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
import os
from AE import AE
from MedModel import *
from setting import STS

class trainer():

    def __init__(self,Gclassname,GinputDim,Glayer,Dclassname,DinputDim,Dlayer,Optionfunc,argparam,AEpath):

        self.G = Gclassname(GinputDim,Glayer)
        self.D = Dclassname(DinputDim,Dlayer)

        # print(self.G,self.D)

        self.setOptim(Optionfunc,**argparam)
        self.setAE(AEpath)

        self.GinputDim = GinputDim


    def setOptim(self,optimfunc,**arg):

        self.G_optim = optimfunc(self.G.parameters(),**arg)
        self.D_optim = optimfunc(self.D.parameters(), **arg)


    def eval(self):
        self.G.eval()

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
        self.G_optim.zero_grad()
        loss.backward()
        self.G_optim.step()

    def stepD(self,loss):
        self.D_optim.zero_grad()
        loss.backward()
        self.D_optim.step()


    def trainG(self,z):
        fake_X = self.G(z)
        pred_fake_X = self.D(fake_X)
        return self.G.getloss( pred_fake_X)

    def trainD(self,z,X):
        fake_X = self.G(z)

        pred_X = self.D(X)
        pred_fake_X = self.D(fake_X)
        return self.D.getloss(pred_X, pred_fake_X)
        pass


    def getGrad(self,X,fake_X):
        alpha = torch.FloatTensor(X.shape[0], 1).uniform_(0, 1)

        alpha = alpha.expand_as(X)
        differences = fake_X - X
        # print(differences.shape)
        interpolates = X + (alpha * differences)
        prob_interpolated = self.D(interpolates)
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolates,
                                        grad_outputs=torch.ones(
                                            prob_interpolated.size()),
                                        create_graph=True, retain_graph=True)[0]
        slopes = torch.sqrt(torch.mean(torch.square(gradients), 1))
        grad_penalty = torch.mean((slopes - 1) ** 2)

        return grad_penalty

    def getD(self):
        return  self.D

    def getG(self):
        return self.G

    def saveG(self,dirpath="./model/"):
        torch.save(self.G, dirpath+self.__name__)

    def createData(self,z):
        return self.G(z)


class Medtrainer(trainer):

    def trainG(self,z):
        fake_X = self.AE.onlydecoder(self.G(z))
        pred_fake_X = self.D(fake_X)
        return self.G.getloss( pred_fake_X)

    def trainD(self,z,X):
        tmp = self.G(z)
        fake_X = self.AE.onlydecoder(tmp)

        pred_X = self.D(X)
        pred_fake_X = self.D(fake_X)
        return self.D.getloss(pred_X, pred_fake_X)
        pass

    def createData(self,z):
        return  self.AE.onlydecoder(self.G(z))


class MedGantrainer(Medtrainer):

    pass

class MedBGantrainer(Medtrainer):

    pass

class EMRWGantrainer(Medtrainer):

    def trainD(self,z,X):
        tmp = self.G(z)
        fake_X = self.AE.onlydecoder(tmp)

        pred_fake_X = self.D(fake_X)
        pred_X = self.D(X)
        baseloss = self.D.getloss(pred_X, pred_fake_X)

        grad_penalty = super().getGrad(X, fake_X)

        return baseloss + grad_penalty
    pass


class MedWGantrainer(Medtrainer):

    def __init__(self,Gclassname,GinputDim,Glayer,Dclassname,DinputDim,Dlayer,Optionfunc,argparam,AEpath,isGrad=False,isClip=False,clamp=0.01):
        super().__init__(Gclassname,GinputDim,Glayer,Dclassname,DinputDim,Dlayer,Optionfunc,argparam,AEpath)
        self.isClip = isClip
        self.isGrad = isGrad
        self.clamp = clamp


    def trainD(self,z,X):
        if not self.isGrad:
            return super().trainD(z,X)
        else:
            fake_X = self.AE.onlydecoder(self.G(z))

            pred_fake_X = self.D(fake_X)
            pred_X = self.D(X)
            baseloss = self.D.getloss(pred_X, pred_fake_X)

            grad_penalty = super().getGrad(X, fake_X)

            return baseloss + grad_penalty

    def stepD(self,loss):
        super().stepD(loss)
        if self.isClip:
            for p in self.D.parameters():
                p.data.clamp_(-self.clamp, self.clamp)


    pass

class DPGantrainer(Medtrainer):

    def __init__(self,Gclassname,GinputDim,Glayer,Dclassname,DinputDim,Dlayer,Optionfunc,argparam,AEpath,isGrad=False,isClip=False,clamp=0.01):
        super().__init__(Gclassname,GinputDim,Glayer,Dclassname,DinputDim,Dlayer,Optionfunc,argparam,AEpath)
        self.isClip = isClip
        self.isGrad = isGrad
        self.clamp = clamp


    def trainD(self,z,X):
        if not self.isGrad:
            return super().trainD(z,X)
        else:
            fake_X = self.AE.onlydecoder(self.G(z))

            pred_fake_X = self.D(fake_X)
            pred_X = self.D(X)
            baseloss = self.D.getloss(pred_X, pred_fake_X)

            grad_penalty = super().getGrad(X, fake_X)

            return baseloss + grad_penalty

    def stepD(self,loss):
        self.D_optim.zero_grad()
        loss.backward(retain_graph=True)

        for group in self.D_optim.param_groups:
            for p in group['params']:
                p.grad = p.grad + (torch.randn(list(p.grad.size())) * 0.1)
        self.D_optim.step()

        if self.isClip:
            for p in self.D.parameters():
                p.data.clamp_(-self.clamp, self.clamp)
    pass

class GanTrainer(trainer):

    pass

class WGantrainer(trainer):

    pass

class AEtrainer(trainer):

    pass

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

def train(model:trainer,dataloader,epochs=100,createsize=3000):

    Gl = []
    Dl = []
    epochl = []
    for epoch in range(epochs):
        Gloss = []
        Dloss = []
        for X in dataloader:
            X = X[0]

            #训练D
            z = torch.randn(size=[X.shape[0],model.getGinputDim()])
            loss = model.trainD(z,X)
            Dloss.append(loss.item())
            model.stepD(loss)

            #训练G
            loss = model.trainG(z)
            Gloss.append(loss.item())
            model.stepG(loss)
        if epoch %20 == 19:
            print(f"epoch = {epoch},Dloss = {np.mean(Dloss)},Gloss={np.mean(Gloss)}")
            epochl.append(epoch)
            Gl.append(np.mean(Gloss))
            Dl.append(np.mean(Dloss))

    #做损失图
    getChart(epochl,Dl,model.__class__.__name__+"_D")
    getChart(epochl, Gl, model.__class__.__name__ + "_G")

    z = torch.randn(size=(createsize,model.getGinputDim()))
    model.eval()
    data = model.createData(z)
    data = data.cpu().detach().numpy()
    np.save(STS["dataSavePath"]+model.__class__.__name__,data)
    print(STS["dataSavePath"]+model.__class__.__name__)

def trainAE(filename,lr=1e-3,epochs=300,Elayer = None ,Dlayer=None):
    data = np.load(filename)
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
    np.save(STS["dataSavePath"]+"AE",ae(data).cpu().detach().numpy())

def main():
    data = np.load(STS["filename"])
    dSize = data.shape[0]
    data = torch.tensor(data, dtype=torch.float)
    torch_dataset = Data.TensorDataset(data)
    loader = Data.DataLoader(dataset=torch_dataset, shuffle=True, batch_size=200)
    for trainname in [MedGantrainer, MedBGantrainer, EMRWGantrainer,MedWGantrainer, DPGantrainer,GanTrainer, WGantrainer]:
#     for trainname in [GanTrainer, WGantrainer]:
        # print(trainname.__name__)
        # print(STS["models"][trainname.__name__])
        model = trainname(**STS["models"][trainname.__name__])
        train(model=model,dataloader=loader,epochs=STS["epoch"],createsize=dSize)
        # print(model.__class__.__name__)




    pass
if __name__=="__main__":
    trainAE(**STS["models"]["AE"])
    main()
