'''
AutoEncoder
'''

import torch
from torch import nn

class AE(nn.Module):
    def __init__(self, inputDim,compressDims=[16],decompressDims=[16]):
        '''

        :param inputDim: 输入样本特征数量
        :param compressDims: 编码器网络参数
        :param decompressDims: 解码器网络参数
        '''
        super().__init__()
        tempDim = inputDim

        # 编码层网络
        self.encoder = nn.Sequential()
        for compressDim in compressDims:
            self.encoder.append(nn.Linear(inputDim,compressDim))
            self.encoder.append(nn.Sigmoid())
            tempDim = compressDim

        #解码层网络
        self.decoder = nn.Sequential()
        for  decompressDim in decompressDims[:-1]:
            self.decoder.append(nn.Linear(tempDim,decompressDim))

        self.decoder.append(nn.Linear(tempDim,decompressDims[-1]))
        self.decoder.append(nn.Sigmoid())

    def forward(self,X):
        X_en = self.encoder(X)
        X_de = self.decoder(X_en)

        return X_de

    def onlydecoder(self,X):
        return self.decoder(X)

    def onlyencoder(self,X):
        return self.encoder(X)

    def getloss(self,Y,Y_hat,type="L2"):
        '''

        :param Y_hat: 预测值
        :param Y: 真实值
        :param continuous:
        :return:
        '''
        # print(Y.shape)
        if type == "L2":
            loss = torch.mean(torch.sum(torch.square(Y - Y_hat),dim=1),dim=0)

        elif type == "Entropy":
            loss = torch.mean(-torch.sum(Y*torch.log(Y_hat+1e-12)+(1-Y)*torch.log(1. - Y_hat +1e-12),dim=1),dim=0)
        else:
            raise "type must L2 or Entropy"

        return loss

# class DualAE(nn.Module):

#     def __init__(self,D_infeature,C_infeature,latent_size,D_encoderlayer,C_encoderlayer,D_decoderlayer,C_decoderlayer):
#         super().__init__()
#         self.D_vae = AE(D_infeature,latent_size,D_encoderlayer,D_decoderlayer)
#         self.C_vae = AE(C_infeature,latent_size,C_encoderlayer,C_decoderlayer)
#         self.D_infeature = D_infeature
#         self.C_infeature = C_infeature

#     def encoder(self,Xd,Xc):
#         d_mu,d_log_var = self.D_vae.encoder(Xd)
#         zd = self.D_vae.reparameterization(d_mu,d_log_var)

#         c_mu,c_log_var = self.C_vae.encoder(Xc)
#         zc = self.C_vae.reparameterization(c_mu,c_log_var)

#         Z = (zd +zc)/2.0

#         return d_mu,d_log_var,zd,c_mu,c_log_var,zc,Z

#     def decoder(self,Z):
#         Xd_hat = self.D_vae.decoder(Z)
#         Xc_hat = self.C_vae.decoder(Z)
#         X_hat = torch.concat((Xd_hat,Xc_hat),dim=1)

#         return Xd_hat,Xc_hat,X_hat

#     def forward(self,Xd,Xc):
#         d_mu, d_log_var, zd, c_mu, c_log_var, zc, Z = self.encoder(Xd,Xc)

#         Xd_hat, Xc_hat, X_hat = self.decoder(Z)

#         return  d_mu,d_log_var,zd,c_mu,c_log_var,zc,Xd_hat, Xc_hat

#     def getloss(self,Xd,Xc,d_mu,d_log_var,zd,c_mu,c_log_var,zc,Xd_hat, Xc_hat):
#         L_d = self.D_vae.loss(Xd,Xd_hat,d_mu,d_log_var)
#         L_c = self.C_vae.loss(Xc,Xc_hat,c_mu,c_log_var)

#         L_match  = torch.mean(torch.square(zd-zc).sum(1))

#         return L_d+L_c+L_match
#         pass