'''
采用gan模式进行训练
'''

import torch
from torch import nn
import numpy as np
import torch.utils.data as Data
import itertools
import os

from dadaptation.dadapt_adam import DAdaptAdam

device = "cpu"


class AE(nn.Module):
    def __init__(self, mindata, maxdata, inputDim, compressDims=[16], decompressDims=[16]):
        '''

        :param inputDim: 输入样本特征数量
        :param compressDims: 编码器网络参数
        :param decompressDims: 解码器网络参数
        '''
        super().__init__()
        tempDim = inputDim

        self.mindata = mindata
        self.maxdata = maxdata

        # 编码层网络
        self.encoder = nn.Sequential()
        print(compressDims, decompressDims)
        for compressDim in compressDims:
            self.encoder.append(nn.Linear(inputDim, compressDim))
            self.encoder.append(nn.Tanh())
            tempDim = compressDim

        # 解码层网络
        self.decoder = nn.Sequential()
        for decompressDim in decompressDims[:-1]:
            self.decoder.append(nn.Linear(tempDim, decompressDim))
            tempDim = decompressDim

        self.decoder.append(nn.Linear(tempDim, decompressDims[-1]))
        self.decoder.append(nn.Sigmoid())

    def forward(self, X):
        X_en = self.encoder(X)
        DecX = self.decoder(X_en)

        DecX = self.mindata + (self.maxdata - self.mindata) * DecX

        return DecX

    def onlydecoder(self, X):
        DecX = self.decoder(X)
        DecX = self.mindata + (self.maxdata - self.mindata) * DecX

        return DecX

    def getloss(self, Y_hat, Y):
        '''

        :param Y_hat: 预测值
        :param Y: 真实值
        :param continuous:
        :return:
        '''
        # print(Y.shape)
        loss_ae = torch.mean(torch.sum(torch.square(Y - Y_hat), 1))

        return loss_ae


class Generator(nn.Module):
    '''

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
        return torch.mean(-torch.log(fake_Y + 1e-12))

        pass

class Discriminator(nn.Module):
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

        for ind,disDim in enumerate(discriminatorDims[:-1]):
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


def trainAE(data, compressdim, lr=1e-3, epochs=20):
    mindata = torch.min(data, dim=0).values
    maxdata = torch.max(data, dim=0).values
    inputdim = data.shape[1]

    ae = AE(mindata, maxdata, inputdim, [compressdim], [inputdim])

    ae_optim = torch.optim.RMSprop(ae.parameters(), lr=lr, weight_decay=0.001, eps=1e-07)
    print(ae)
    torch_dataset = Data.TensorDataset(data)
    loader = Data.DataLoader(dataset=torch_dataset, shuffle=True, batch_size=200)

    for epoch in range(epochs):
        lossvec = []
        for step, batch in enumerate(loader):
            x_hat = ae(batch[0])
            loss = ae.getloss(x_hat, batch[0])
            lossvec.append(loss.item())

            # 梯度下降
            ae_optim.zero_grad()
            loss.backward()
            ae_optim.step()
        if epoch % 10 == 9:
            print(epoch, np.mean(lossvec))

    torch.save(ae, "./model/ae.pth")
    return ae


def getChart(epoch, dloss, name="default"):
    y = []
    if isinstance(dloss[0], list):
        for i in range(len(dloss[0])):
            tmp = [j[i] for j in dloss]
            y.append(tmp)
    else:
        y.append(dloss)

    savedir = "./"
    import matplotlib.pyplot as plt
    plt.clf()
    plt.title(name)
    for lable, i in enumerate(y):
        plt.plot(epoch, i, label=str(lable))
    plt.legend()
    if os.path.exists(savedir) == False:
        os.makedirs(savedir)
    if os.path.exists(os.path.join(savedir, name.replace(".", "_") + ".png")):
        os.remove(os.path.join(savedir, name.replace(".", "_") + ".png"))
    plt.savefig(os.path.join(savedir, name.replace(".", "_")))


def trainGD(data0, data1, compressdim, varysize=30, lr=1e-3, epochs=100,batch_size=200):
    print(data0[:, :-varysize].shape)
    # aedata = torch.concat((data0[:, -varysize:], data1[:, -varysize:]), dim=0)
    aedata = torch.concat((data0, data1), dim=0)
    print(aedata.shape)
    ae = trainAE(data=aedata, compressdim=compressdim, lr=1e-3, epochs=200)

    datadim = data0.shape[1]

    print(compressdim)
    # datacompressdim = np.max(int(2 ** np.floor(np.log2(data0.shape[1]) - 2)),4)
    datacompressdim = 4
    netG_01 = Generator(datadim, [ compressdim, compressdim])  # 01
    netG_10 = Generator(datadim, [compressdim, compressdim])  # 10
    print(netG_10)
    print(netG_01)

    optim_G = torch.optim.RMSprop(itertools.chain(netG_01.parameters(), netG_10.parameters()), lr=lr,
                                  weight_decay=0.001, eps=1e-07)

    netD_0 = Discriminator(datadim, [compressdim, compressdim, compressdim, 1])  # 0
    netD_1 = Discriminator(datadim, [compressdim, compressdim, compressdim, 1])  # 1
    print(netD_1)
    print(netD_0)

    optim_netD_0 = DAdaptAdam(netD_0.parameters(), lr=lr, weight_decay=0.001, eps=1e-07)
    optim_netD_1 = DAdaptAdam(netD_1.parameters(), lr=lr, weight_decay=0.001, eps=1e-07)

    # https://github.com/aitorzip/PyTorch-CycleGAN/blob/master/train
    lossvec = []
    for epoch in range(epochs):
        # 训练0到1
        np.random.shuffle(data0)
        np.random.shuffle(data1)
        torch_dataset = Data.TensorDataset(data0, data1)
        loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size)
        for i, (real0, real1) in enumerate(loader):

            con0 = real0[:, :-varysize]
            vary0 = real0[:, -varysize:]

            con1 = real1[:, :-varysize]
            vary1 = real1[:, -varysize:]

            # 计算D0 loss
            optim_netD_0.zero_grad()
            tmp = netG_10(real1)
            vary_fake0 = ae.onlydecoder(tmp)
            fake0 = torch.concat((con1, vary_fake0), dim=1)

            pred_real = netD_0(real0)
            pred_fake = netD_0(fake0.detach())

            loss_D0 = netD_0.getloss(pred_real,pred_fake)
            loss_D0.backward()
            optim_netD_0.step()

            # 计算D1 loss
            optim_netD_1.zero_grad()

            tmp = netG_01(real0)
            vary_fake1 = ae.onlydecoder(tmp)
            fake1 = torch.concat((con0, vary_fake1), dim=1)

            pred_real = netD_1(real1)
            pred_fake = netD_1(fake1.detach())
            loss_D1 = netD_1.getloss(pred_real,pred_fake)
            loss_D1.backward()

            optim_netD_1.step()

            # 训练G
            optim_G.zero_grad()
            # 计算循环一致性
            tmp = netG_01(real0)
            vary_fake1 = ae.onlydecoder(tmp)
            fake1 = torch.concat((con0, vary_fake1), dim=1)

            tmp = netG_10(real1)
            vary_fake0 = ae.onlydecoder(tmp)
            fake0 = torch.concat((con1, vary_fake0), dim=1)

            tmp = netG_01(fake0)
            cycle1 = ae.onlydecoder(tmp)
            L_con1 = torch.mean(torch.sum(torch.abs(vary1 - cycle1), 1))

            tmp = netG_10(fake1)
            cycle0 = ae.onlydecoder(tmp)
            L_con0 = torch.mean(torch.sum(torch.abs(vary0 - cycle0), 1))

            # 计算GAN loss
            pred_fake = netD_0(fake0)
            loss_GAN_0 = netG_10.getloss(pred_fake)

            pred_fake = netD_1(fake1)
            loss_GAN_1 = netG_01.getloss(pred_fake)

            loss_G = L_con0 + L_con1 + loss_GAN_0 + loss_GAN_1
            loss_G.backward()
            optim_G.step()

        if epoch % 10 == 9:
            print("epoch={}, train_G={} ,train_D0 = {},train_D1 = {}".format(epoch, loss_G, loss_D0, loss_D1))
            lossvec.append([epoch, loss_G.item(), loss_D0.item(), loss_D1.item()])

    torch.save(netG_01, "./model/netG_01.pth")
    torch.save(netG_10, "./model/netG_10.pth")
    getChart([i[0] for i in lossvec], [i[1:] for i in lossvec])



def sysdata(data, varysize=30, type=0):
    ae = torch.load("./model/ae.pth")
    if type == 0:
        G = torch.load("./model/netG_10.pth")
    else:
        G = torch.load("./model/netG_01.pth")
    tmp = G(data)
    # for paramater in G.parameters():
    #     print(paramater.data)
    vary_fake = ae.onlydecoder(tmp)

    # vary_fake = ae(data[:,-varysize:])

    print(vary_fake.shape)
    fake = torch.concat((data[:, : -varysize], vary_fake), dim=1)

    return fake


def test():
    if not os.path.exists("./model"):
        os.makedirs("./model/")

    data0 = np.random.uniform(1, 100, size=(1000, 60))
    data0 = torch.from_numpy(data0).float()

    data1 = np.random.uniform(1, 100, size=(1000, 60))
    data1 = torch.from_numpy(data1).float()

    # exit()
    varysize = 30
    trainGD(data0, data1, varysize)

    data = sysdata(data0, varysize=30, type=0)
    print(data.shape)


def maintrain():
    from imblearn.over_sampling import SMOTE
    data = np.load(filepath)

    Label = -4
    X = np.delete(data,Label,axis=1)
    y = data[:, -4]

    y = y.astype(np.int)

    smote = SMOTE(random_state=1337)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    y_resampled = y_resampled.reshape(-1, 1)

    print(X_resampled.shape, y_resampled.shape)
    data = np.concatenate((X_resampled, y_resampled), axis=1)
    print(data.shape)

    data0 = data[data[:, -1] == 0]
    data1 = data[data[:, -1] == 1]

    data0 = torch.from_numpy(data0).float()
    data1 = torch.from_numpy(data1).float()

    # data0,data1,compressdim,varysize=30,lr = 1e-3,epochs =100

    paramss = {
        "data0": data0,
        "data1": data1,
        "compressdim": 8,
        "varysize": varysize,
        "lr": 1e-3,
        "epochs": 1000,
        "batch_size":200
    }

    trainGD(**paramss)

    # trainGD(data0=data0,data1=data1,compressdim=8,varysize=32,lr=1e-3,epochs=2000)

filepath = "./stroke_train.npy"
varysize=35

if __name__ == "__main__":
    maintrain()
    test = np.load(filepath)
    data0 = test
    data1 = test

    data0 = torch.from_numpy(data0).float()
    data1 = torch.from_numpy(data1).float()


    # data = sysdata(data=data1,varysize=varysize,type=0)
    # data = data.detach().numpy()
    # np.save("./data/syn0",data)
    print(varysize,data0.shape)
    print(data0[:, -varysize:].shape)

    sysdata1 = sysdata(data=data0, varysize=varysize, type=1)
    sysdata1 = sysdata1.detach().numpy()

    sysdata0 = sysdata(data=data1, varysize=varysize, type=0)
    sysdata0 = sysdata0.detach().numpy()

    data = np.concatenate((sysdata1, sysdata0), axis=0)
    print(data.shape)
    print(sysdata0.shape)

    np.save("./onehotstroke/cycmedgan", data)
    np.save("./onehotstroke/cycmedgan0", sysdata0)
    np.save("./onehotstroke/cycmedgan1", sysdata1)






