from MedModel import *
import torch
import GAN
import WGAN


# #lung
# filename ="/home/amax/Documents/Seagate4T/GAN/fanshishi/data/Data_preprocess/lung/train.npy"
# GinputDim = 45
# Glayer= [32,32,32]
# DinputDim= 45
# compressDim = 32
# Dlayer=[8,8]
# argparam={"lr":1e-4,"weight_decay":0.001,"eps":1e-07}
# AEpath="../model/lung/AE.pth"
# classifierpath = "/home/amax/Documents/Seagate4T/GAN/fanshishi/model/lung/classifier.pth"
# dataSavePath = "/home/amax/Documents/Seagate4T/GAN/fanshishi/data/SynData/onehot_Binary_lung/"
# modelpath="/home/amax/Documents/Seagate4T/GAN/fanshishi/model/lung/"
# Label = -1

# #crc
# filename ="/home/amax/Documents/Seagate4T/GAN/fanshishi/data/Data_preprocess/crc/train.npy"
# GinputDim = 64
# Glayer= [32,32,32]
# DinputDim= 64
# compressDim = 32
# Dlayer=[8,8]
# argparam={"lr":1e-4,"weight_decay":0.001,"eps":1e-07}
# AEpath="../model/crc/AE.pth"
# classifierpath = "/home/amax/Documents/Seagate4T/GAN/fanshishi/model/crc/classifier.pth"
# dataSavePath = "/home/amax/Documents/Seagate4T/GAN/fanshishi/data/SynData/onehot_Binary_crc/"
# modelpath="/home/amax/Documents/Seagate4T/GAN/fanshishi/model/crc/"
# Label = -1

# #breast
# filename ="/home/amax/Documents/Seagate4T/GAN/fanshishi/data/Data_preprocess/breast/train.npy"
# GinputDim = 86
# Glayer= [32,32,32]
# DinputDim= 86
# compressDim = 32
# Dlayer=[8,8]
# argparam={"lr":1e-4,"weight_decay":0.001,"eps":1e-07}
# AEpath="../model/breast/AE.pth"
# classifierpath = "/home/amax/Documents/Seagate4T/GAN/fanshishi/model/breast/classifier.pth"
# dataSavePath = "/home/amax/Documents/Seagate4T/GAN/fanshishi/data/SynData/onehot_Binary_breast/"
# modelpath="/home/amax/Documents/Seagate4T/GAN/fanshishi/model/breast/"
# Label = -1

# #stroke
# filename ="/home/amax/Documents/Seagate4T/GAN/fanshishi/data/Data_preprocess/Stroke/stroke_train.npy"
# GinputDim = 22
# Glayer= [16,16,16]
# DinputDim= 22
# compressDim = 16
# Dlayer=[8,8]
# argparam={"lr":1e-4,"weight_decay":0.001,"eps":1e-07}
# AEpath="../model/stroke/AE.pth"
# classifierpath = "/home/amax/Documents/Seagate4T/GAN/fanshishi/model/stroke/classifier.pth"
# dataSavePath = "/home/amax/Documents/Seagate4T/GAN/fanshishi/data/SynData/onehot_Binary_stroke/"
# modelpath="/home/amax/Documents/Seagate4T/GAN/fanshishi/model/stroke/"
# Label = -1

# AKI
# filename ="/home/amax/Documents/Seagate4T/GAN/fanshishi/data/Data_preprocess/Aki/train.npy"
# GinputDim = 50
# Glayer= [16,16,16]
# DinputDim=50
# compressDim = 16
# Dlayer=[16,16]
# argparam={"lr":1e-4,"weight_decay":0.001,"eps":1e-07}
# AEpath="../model/aki/AE.pth"
# classifierpath = "../model/aki/classifier.pth"
# dataSavePath = "/home/amax/Documents/Seagate4T/GAN/fanshishi/data/SynData/onehot_Binary_aki/"
# modelpath="../model/aki/"
# Label = -1

# AKI_mimic
# filename ="/home/amax/Documents/Seagate4T/GAN/fanshishi/data/Data_preprocess/Aki_mimic/train.npy"
# GinputDim = 50
# Glayer= [32,32,32]
# DinputDim=50
# compressDim = 32
# Dlayer=[16,16]
# argparam={"lr":1e-4,"weight_decay":0.001,"eps":1e-07}
# AEpath="../model/aki_mimic/AE.pth"
# classifierpath = "../model/aki_mimic/classifier.pth"
# dataSavePath = "/home/amax/Documents/Seagate4T/GAN/fanshishi/data/SynData/onehot_Binary_aki_mimic/"
# modelpath="../model/aki_mimic/"
# Label = -1

#HD
filename ="/home/amax/Documents/Seagate4T/GAN/fanshishi/data/Data_preprocess/Hd/filter_train.npy"
GinputDim = 79
Glayer= [64,64,64]
DinputDim=79
compressDim = 64
Dlayer=[16,16]
argparam={"lr":1e-4,"weight_decay":0.001,"eps":1e-07}
AEpath="../model/hd/AE.pth"
classifierpath = "../model/hd/classifier.pth"
dataSavePath = "/home/amax/Documents/Seagate4T/GAN/fanshishi/data/SynData/onehot_Binary_hd/"
modelpath="../model/hd/"
Label = -1


__all__=["STS"]
#(Gclassname=medgan_Generator,GinputDim=16,Glayer=[16,16],Dclassname=medgan_Discriminator,DinputDim=33,Dlayer=[16,8],Optionfunc=torch.optim.Adam,argparam={"lr":1e-3,"weight_decay":0.001,"eps":1e-07},AEpath="./model/AE.pth")


STS = {
    "dataSavePath":dataSavePath,
    "filename":filename,
    "AEpath":AEpath,
    "epoch":2000,
    "createsize":3000,
    "Label":Label,
    "modelpath":modelpath,
    "models":{
        "AE":{
            "filename":filename,
            "lr":1e-3,
            "epochs":300,
            "Elayer":[compressDim],
            "Dlayer":[DinputDim]
        },
        "MedGantrainer":{
            "Gclassname":medgan_Generator,
            "GinputDim":GinputDim,
            "Glayer":Glayer,
            "Dclassname":medgan_Discriminator,
            "DinputDim":DinputDim,
            "Dlayer":Dlayer,
            "Optionfunc":torch.optim.Adam,
            "argparam":argparam,
            "AEpath":AEpath
        },
        "MedBGantrainer": {
            "Gclassname": medbgan_Generator,
            "GinputDim": GinputDim,
            "Glayer": Glayer,
            "Dclassname": medbgan_Discriminator,
            "DinputDim": DinputDim,
            "Dlayer": Dlayer,
            "Optionfunc": torch.optim.Adam,
            "argparam": argparam,
            "AEpath": AEpath
        },
        "EMRWGantrainer": {
            "Gclassname": emrwgan_Generator,
            "GinputDim": GinputDim,
            "Glayer": Glayer,
            "Dclassname": emrwgan_Discriminator,
            "DinputDim": DinputDim,
            "Dlayer": Dlayer,
            "Optionfunc": torch.optim.Adam,
            "argparam": argparam,
            "AEpath": AEpath
        },
        "MedWGantrainer": {
            "Gclassname": emrwgan_Generator,
            "GinputDim": GinputDim,
            "Glayer": Glayer,
            "Dclassname": emrwgan_Discriminator,
            "DinputDim": DinputDim,
            "Dlayer": Dlayer,
            "Optionfunc": torch.optim.Adam,
            "argparam": argparam,
            "AEpath": AEpath,
            "isGrad":False,
            "isClip":False,
            "clamp":0.01
        },
        "DPGantrainer": {
            "Gclassname": emrwgan_Generator,
            "GinputDim": GinputDim,
            "Glayer": Glayer,
            "Dclassname": emrwgan_Discriminator,
            "DinputDim": DinputDim,
            "Dlayer": Dlayer,
            "Optionfunc": torch.optim.Adam,
            "argparam": argparam,
            "AEpath": AEpath,
            "isGrad": False,
            "isClip": False,
            "clamp": 0.01
        },
        "GanTrainer": {
            "Gclassname": GAN.Generator,
            "GinputDim": GinputDim,
            "Glayer": [8,DinputDim],
            "Dclassname": GAN.Discriminator,
            "DinputDim": DinputDim,
            "Dlayer": Dlayer,
            "Optionfunc": torch.optim.Adam,
            "argparam": argparam,
            "AEpath": AEpath,
        },
        "WGantrainer": {
            "Gclassname": WGAN.Generator,
            "GinputDim": GinputDim,
            "Glayer": [8,DinputDim],
            "Dclassname": WGAN.Discriminator,
            "DinputDim": DinputDim,
            "Dlayer": Dlayer,
            "Optionfunc": torch.optim.Adam,
            "argparam": argparam,
            "AEpath": AEpath,
        },
        "cycleMedGANtrainer":{
            "Gclassname": medgan_Generator,
            "GinputDim": GinputDim,
            "Glayer": Glayer,
            "Dclassname": medgan_Discriminator,
            "DinputDim": DinputDim,
            "Dlayer": Dlayer,
            "Optionfunc": torch.optim.Adam,
            "argparam": argparam,
            "AEpath": AEpath
        },
        "cycleGANtrainer":{
            "Gclassname": medgan_Generator,
            "GinputDim": GinputDim,
            "Glayer": Glayer,
            "Dclassname": medgan_Discriminator,
            "DinputDim": DinputDim,
            "Dlayer": Dlayer,
            "Optionfunc": torch.optim.Adam,
            "argparam": argparam,
            "AEpath": AEpath
        }
    }
}

