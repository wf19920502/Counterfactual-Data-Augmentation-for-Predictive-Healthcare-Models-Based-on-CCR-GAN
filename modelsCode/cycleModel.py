import numpy as np
import  torch
from torch import nn
import os, re
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import f1_score

__all__=["cyclemedganGenerator","cyclemedganDiscriminator","ClassifierModel","cycleRGenerator","DualVAE","SingleVAE"]

class cyclemedganGenerator(nn.Module):
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
            nn.BatchNorm1d(genDim,momentum=0.009),
            nn.ReLU()
            )
            tempDim = genDim
            self.layer.append(layer)

        layer = nn.Sequential(
            nn.Linear(tempDim,generatorDims[-1],bias=False),
            nn.BatchNorm1d(generatorDims[-1],momentum=0.01),
            nn.Tanh()
            )
        self.layer.append(layer)

    def forward(self,X):

        return self.layer(X)

    def getloss(self,fake_Y):
        '''
        :param fake_Y: fake_Y = D(G(x))
        :return: 损失值
        '''
        return torch.mean(torch.log(1-fake_Y + 1e-12))

        pass



class cyclemedganDiscriminator(nn.Module):
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
#             self.module.append(nn.BatchNorm1d(disDim,momentum=0.009),)
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

#残差分类器
class cycleRGenerator(nn.Module):
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
            nn.BatchNorm1d(genDim,momentum=0.009),
            nn.ReLU()
            )
            tempDim = genDim
            self.layer.append(layer)

        layer = nn.Sequential(
            nn.Linear(tempDim,generatorDims[-1],bias=False),
            nn.BatchNorm1d(generatorDims[-1],momentum=0.01),
            nn.Tanh()
            )
        self.layer.append(layer)

    def forward(self,X):
        temp = X

        for layer in self.layer:
            temp = temp + layer(temp)

        return temp

    def getloss(self,fake_Y):
        '''
        :param fake_Y: fake_Y = D(G(x))
        :return: 损失值
        '''
        return torch.mean(-torch.log(fake_Y + 1e-12))

        pass


#分类器模型架构
class ClassifierModel(nn.Module):
    def __init__(self,inputDim,Layer):
        super().__init__()
        tempDim = inputDim

        self.layer = []
        for Dim in Layer[:-1]:
            self.layer.append(nn.Linear(tempDim,Dim))
#             self.layer.append(nn.BatchNorm1d(Dim,momentum=0.009))
            self.layer.append(nn.Tanh())
            tempDim = Dim

        self.layer.append(nn.Linear(tempDim,Layer[-1]))
        # self.layer.append(nn.Softmax())
        self.layer.append(nn.Sigmoid())

        self.layer = nn.Sequential(*self.layer)

    def forward(self,X):
        return torch.squeeze(self.layer(X))

#     def getloss(self,y_hat,y):
#         return torch.mean(-y*torch.log(y_hat)+(1-y)*torch.log(1-y_hat))

    def getloss(self,y_hat,y,alph=0.957,gamm=2):
        alph = alph
        gamm = gamm
        return torch.mean(- alph *((1-y_hat)**gamm)*y*torch.log(y_hat) -(1-alph)*y_hat**gamm *(1-y)*torch.log(1-y_hat ))

    # def getloss(self,y_hat,y):
    #
    #     P = y_hat[range(len(y_hat)), y]
    #     alph = 0.2
    #     gamm = 2
    #     return  torch.mean(-alph*((1-P)**gamm) * torch.log(P))
    #
    def accuracy(self,y_hat, y):  # @save
        """计算预测正确的数量"""
        y_hat[y_hat> 0.5] = 1
        y_hat[y_hat < 1] = 0

        F1 = f1_score(y_hat, y, average='macro')

        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(axis=1)
        cmp = y_hat.type(y.dtype) == y
        cmp0 = float(cmp[y ==0].type(y.dtype).sum())/float((y == 0).sum()+0.1)

        cmp1 = float(cmp[y == 1].type(y.dtype).sum()) / float((y == 1).sum() +0.1)
        return float(cmp.type(y.dtype).sum())/len(y),cmp0,cmp1,F1


class VAE(nn.Module):

    def __init__(self,in_features,latent_size,in_layers,out_layer):
        super().__init__()
        self.latent_size = latent_size

        self.encoder_forward = nn.Sequential()

        tempDim = in_features

        for Dim in in_layers:
            layer = nn.Sequential(
                nn.Linear(tempDim,Dim),
                nn.ReLU()
            )
            tempDim = Dim
            self.encoder_forward.append(layer)
        self.encoder_forward.append(nn.Sequential(nn.Linear(tempDim,2 * latent_size)))

        self.decoder_forward = nn.Sequential()

        tempDim = latent_size
        for Dim in out_layer[:-1]:
            self.decoder_forward.append(nn.Sequential(
                nn.Linear(tempDim,Dim),
                nn.ReLU()
            ))

            tempDim = Dim

        self.decoder_forward.append(nn.Sequential(
            nn.Linear(tempDim,out_layer[-1]),
            nn.Sigmoid()
        ))

    def encoder(self, X):
        out = self.encoder_forward(X)
        mu = out[:, :self.latent_size]
        log_var = out[:, self.latent_size:]
        return mu, log_var

    def decoder(self, z):
        mu_prime = self.decoder_forward(z)
        return mu_prime

    def reparameterization(self, mu, log_var):
        epsilon = torch.randn_like(log_var)
        z = nn.Sigmoid()(mu + epsilon * torch.sqrt(log_var.exp()))
        return z

    def forward(self, X, *args, **kwargs):
        mu, log_var = self.encoder(X)
        z = self.reparameterization(mu, log_var)
        mu_prime = self.decoder(z)
        return mu_prime, mu, log_var

    def loss(self, X, mu_prime, mu, log_var):
        # reconstruction_loss = F.mse_loss(mu_prime, X, reduction='mean') is wrong!
        reconstruction_loss = torch.mean(torch.square(X - mu_prime).sum(dim=1))

        latent_loss = torch.mean(0.5 * (log_var.exp() + torch.square(mu) - log_var).sum(dim=1))
        # print("reconstruction_loss=",reconstruction_loss,"  latent_loss=",latent_loss)
        return reconstruction_loss + latent_loss


class SingleVAE(nn.Module):

    def __init__(self,C_infeature,latent_size,C_encoderlayer,C_decoderlayer):
        super().__init__()
        self.C_vae = VAE(C_infeature,latent_size,C_encoderlayer,C_decoderlayer)
        self.C_infeature = C_infeature

    def encoder(self,Xc):

        c_mu,c_log_var = self.C_vae.encoder(Xc)
        zc = self.C_vae.reparameterization(c_mu,c_log_var)

        return c_mu,c_log_var,zc

    def decoder(self,Z):
        Xc_hat = self.C_vae.decoder(Z)

        return Xc_hat

    def forward(self,Xc):
        c_mu, c_log_var, zc  = self.encoder(Xc)

        Xc_hat = self.decoder(zc)

        return  c_mu,c_log_var,zc, Xc_hat

    def getloss(self,Xc,c_mu,c_log_var, Xc_hat):
        L_c = self.C_vae.loss(Xc,Xc_hat,c_mu,c_log_var)

        return L_c
        pass


class DualVAE(nn.Module):

    def __init__(self,D_infeature,C_infeature,latent_size,D_encoderlayer,C_encoderlayer,D_decoderlayer,C_decoderlayer):
        super().__init__()
        self.D_vae = VAE(D_infeature,latent_size,D_encoderlayer,D_decoderlayer)
        self.C_vae = VAE(C_infeature,latent_size,C_encoderlayer,C_decoderlayer)
        self.D_infeature = D_infeature
        self.C_infeature = C_infeature

    def encoder(self,Xd,Xc):
        d_mu,d_log_var = self.D_vae.encoder(Xd)
        zd = self.D_vae.reparameterization(d_mu,d_log_var)

        c_mu,c_log_var = self.C_vae.encoder(Xc)
        zc = self.C_vae.reparameterization(c_mu,c_log_var)

        Z = (zd +zc)/2.0

        return d_mu,d_log_var,zd,c_mu,c_log_var,zc,Z

    def decoder(self,Z):
        Xd_hat = self.D_vae.decoder(Z)
        Xc_hat = self.C_vae.decoder(Z)
        X_hat = torch.concat((Xd_hat,Xc_hat),dim=1)

        return Xd_hat,Xc_hat,X_hat

    def forward(self,Xd,Xc):
        d_mu, d_log_var, zd, c_mu, c_log_var, zc, Z = self.encoder(Xd,Xc)

        Xd_hat, Xc_hat, X_hat = self.decoder(Z)

        return  d_mu,d_log_var,zd,c_mu,c_log_var,zc,Xd_hat, Xc_hat

    def getloss(self,Xd,Xc,d_mu,d_log_var,zd,c_mu,c_log_var,zc,Xd_hat, Xc_hat):
        L_d = self.D_vae.loss(Xd,Xd_hat,d_mu,d_log_var)
        L_c = self.C_vae.loss(Xc,Xc_hat,c_mu,c_log_var)

        L_match  = torch.mean(torch.square(zd-zc).sum(1))

        return L_d+L_c+L_match
        pass
