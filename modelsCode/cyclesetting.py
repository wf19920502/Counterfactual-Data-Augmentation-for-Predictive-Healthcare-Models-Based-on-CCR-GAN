from cycleModel import *
from EMRRGAN import  *
import torch
import GAN
import WGAN
from dadaptation import dadapt_adam


# # #lung
# filename ="/home/amax/Documents/Seagate4T/GAN/fanshishi/data/Data_preprocess/lung/train.npy"
# testname ="/home/amax/Documents/Seagate4T/GAN/fanshishi/data/Data_preprocess/lung/test.npy"
# GinputDim = 44
# Glayer= [32,32,32]
# DinputDim= 44
# compressDim = 32
# Dlayer=[8,8]
# argparam={"lr":1e-4,"weight_decay":0.001,"eps":1e-07}
# AEpath="/home/amax/Documents/Seagate4T/GAN/fanshishi//model/lung/CycleAE.pth"
# EMRAEpath="/home/amax/Documents/Seagate4T/GAN/fanshishi//model/lung/EMRAE.pth"
# classifierpath = "/home/amax/Documents/Seagate4T/GAN/fanshishi/model/lung/classifier.pth"
# DualVAE = "/home/amax/Documents/Seagate4T/GAN/fanshishi/model/lung/DualVAE.pth"
# dataSavePath = "/home/amax/Documents/Seagate4T/GAN/fanshishi/data/SynData/onehot_Binary_lung/"
# modelpath="/home/amax/Documents/Seagate4T/GAN/fanshishi/model/lung/"
# Label = -1
# D_size = 43
# C_size = 1
# latent_size=32
# # active = 'gumbel' #active== False;active== gumbel;active== onehot;
# # active = False
# active = 'onehot'
# Cols = [['D', 3], ['D', 8], ['D', 5], ['D', 7], ['D', 4], ['D', 2], ['D', 6], ['D', 6], ['D', 2], ['C', 1]]


#crc
# filename ="/home/amax/Documents/Seagate4T/GAN/fanshishi/data/Data_preprocess/crc/train.npy"
# testname ="/home/amax/Documents/Seagate4T/GAN/fanshishi/data/Data_preprocess/crc/test.npy"
# GinputDim = 63
# Glayer= [32,32,32]
# DinputDim= 63
# compressDim = 32
# Dlayer=[8,8]
# argparam={"lr":1e-4,"weight_decay":0.001,"eps":1e-07}
# AEpath="/home/amax/Documents/Seagate4T/GAN/fanshishi//model/crc/CycleAE.pth"
# EMRAEpath="/home/amax/Documents/Seagate4T/GAN/fanshishi//model/crc/EMRAE.pth"
# classifierpath = "/home/amax/Documents/Seagate4T/GAN/fanshishi/model/crc/classifier.pth"
# DualVAE = "/home/amax/Documents/Seagate4T/GAN/fanshishi/model/crc/DualVAE.pth"
# dataSavePath = "/home/amax/Documents/Seagate4T/GAN/fanshishi/data/SynData/onehot_Binary_crc/"
# modelpath="/home/amax/Documents/Seagate4T/GAN/fanshishi/model/crc/"
# Label = -1
# D_size = 59
# C_size = 4
# latent_size=32
# active = 'gumbel' #active== False;active== gumbel;active== onehot;
# Cols = [['D', 3], ['D', 11], ['D', 5], ['D', 5], ['D', 6], ['D', 5], ['D', 2], ['D', 9], ['D', 5], ['D', 6], ['D', 2], ['C', 4]]


# #breast
# filename ="/home/amax/Documents/Seagate4T/GAN/fanshishi/data/Data_preprocess/breast/train.npy"
# testname ="/home/amax/Documents/Seagate4T/GAN/fanshishi/data/Data_preprocess/breast/test.npy"
# GinputDim = 85
# Glayer= [32,32,32]
# DinputDim= 85
# compressDim = 32
# Dlayer=[8,8]
# argparam={"lr":1e-4,"weight_decay":0.001,"eps":1e-07}
# AEpath="/home/amax/Documents/Seagate4T/GAN/fanshishi//model/breast/CycleAE.pth"
# EMRAEpath="/home/amax/Documents/Seagate4T/GAN/fanshishi//model/breast/EMRAE.pth"
# classifierpath = "/home/amax/Documents/Seagate4T/GAN/fanshishi/model/breast/classifier.pth"
# DualVAE = "/home/amax/Documents/Seagate4T/GAN/fanshishi/model/breast/DualVAE.pth"
# dataSavePath = "/home/amax/Documents/Seagate4T/GAN/fanshishi/data/SynData/onehot_Binary_breast/"
# modelpath="/home/amax/Documents/Seagate4T/GAN/fanshishi/model/breast/"
# Label = -1
# D_size = 81
# C_size = 4
# latent_size=16
# active = 'gumbel' #active== False;active== gumbel;active== onehot;
# Cols = [['D', 3], ['D', 2], ['D', 9], ['D', 7], ['D', 5], ['D', 11], ['D', 15], ['D', 2], ['D', 4], ['D', 4], ['D', 6], ['D', 5], ['D', 6], ['D', 2], ['C', 4]]

#stroke
# filename ="/home/amax/Documents/Seagate4T/GAN/fanshishi/data/Data_preprocess/Stroke/stroke_train.npy"
# testname ="/home/amax/Documents/Seagate4T/GAN/fanshishi/data/Data_preprocess/Stroke/stroke_test.npy"
# GinputDim = 21
# Glayer= [16,16,16]
# DinputDim= 21
# compressDim = 16
# Dlayer=[8,8]
# argparam={"lr":1e-4,"weight_decay":0.001,"eps":1e-07}
# AEpath="/home/amax/Documents/Seagate4T/GAN/fanshishi//model/stroke/CycleAE.pth"
# EMRAEpath="/home/amax/Documents/Seagate4T/GAN/fanshishi//model/stroke/EMRAE.pth"
# classifierpath = "/home/amax/Documents/Seagate4T/GAN/fanshishi/model/stroke/classifier.pth"
# DualVAE = "/home/amax/Documents/Seagate4T/GAN/fanshishi/model/stroke/DualVAE.pth"
# dataSavePath = "/home/amax/Documents/Seagate4T/GAN/fanshishi/data/SynData/onehot_Binary_stroke/"
# modelpath="/home/amax/Documents/Seagate4T/GAN/fanshishi/model/stroke/"
# Label = -1
# D_size = 18
# C_size = 3
# latent_size=16
# active = 'gumbel' #active= False;active= gumbel;active= onehot;
# # active = False
# # active = 'onehot'
# Cols = [['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 4], ['D', 2], ['D', 4], ['C', 3]]

#AKI
# filename ="/home/amax/Documents/Seagate4T/GAN/fanshishi/data/Data_preprocess/Aki/train.npy"
# testname ="/home/amax/Documents/Seagate4T/GAN/fanshishi/data/Data_preprocess/Aki/test.npy"
# GinputDim = 49
# Glayer= [16,16,16]
# DinputDim=49
# compressDim = 16
# Dlayer=[16,16]
# argparam={"lr":1e-4,"weight_decay":0.001,"eps":1e-07}
# AEpath="/home/amax/Documents/Seagate4T/GAN/fanshishi//model/aki/CycleAE.pth"
# EMRAEpath="/home/amax/Documents/Seagate4T/GAN/fanshishi//model/aki/EMRAE.pth"
# classifierpath = "/home/amax/Documents/Seagate4T/GAN/fanshishi/model/aki/classifier.pth"
# DualVAE = "/home/amax/Documents/Seagate4T/GAN/fanshishi/model/aki/DualVAE.pth"
# dataSavePath = "/home/amax/Documents/Seagate4T/GAN/fanshishi/data/SynData/onehot_Binary_aki/"
# modelpath="/home/amax/Documents/Seagate4T/GAN/fanshishi/model/aki/"
# Label = -1
# D_size = 34
# C_size = 15
# latent_size=16
# active = 'gumbel' #active== False;active== gumbel;active== onehot;
# # active = False
# # active = 'onehot'
# Cols = [['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['C', 15]]


#AKI_mimic
# filename ="/home/amax/Documents/Seagate4T/GAN/fanshishi/data/Data_preprocess/Aki_mimic/train.npy"
# testname ="/home/amax/Documents/Seagate4T/GAN/fanshishi/data/Data_preprocess/Aki_mimic/test.npy"
# GinputDim = 49
# Glayer= [32,32,32]
# DinputDim=49
# compressDim = 32
# Dlayer=[16,16]
# argparam={"lr":1e-4,"weight_decay":0.001,"eps":1e-07}
# AEpath="/home/amax/Documents/Seagate4T/GAN/fanshishi//model/aki_mimic/CycleAE.pth"
# EMRAEpath="/home/amax/Documents/Seagate4T/GAN/fanshishi//model/aki_mimic/EMRAE.pth"
# classifierpath = "/home/amax/Documents/Seagate4T/GAN/fanshishi/model/aki_mimic/classifier.pth"
# DualVAE = "/home/amax/Documents/Seagate4T/GAN/fanshishi/model/aki_mimic/DualVAE.pth"
# dataSavePath = "/home/amax/Documents/Seagate4T/GAN/fanshishi/data/SynData/onehot_Binary_aki_mimic/"
# modelpath="/home/amax/Documents/Seagate4T/GAN/fanshishi/model/aki_mimic/"
# Label = -1
# D_size = 34
# C_size = 15
# latent_size=16
# active = 'gumbel' #active== False;active== gumbel;active== onehot;
# # active = False
# # active = 'onehot'
# Cols = [['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['C', 15]]

# HD
filename ="/home/amax/Documents/Seagate4T/GAN/fanshishi/data/Data_preprocess/Hd/filter_train.npy"
testname ="/home/amax/Documents/Seagate4T/GAN/fanshishi/data/Data_preprocess/Hd/filter_test.npy"
GinputDim = 78
Glayer= [64,64,64]
DinputDim=78
compressDim = 64
Dlayer=[16,16]
argparam={"lr":1e-4,"weight_decay":0.001,"eps":1e-07}
AEpath="/home/amax/Documents/Seagate4T/GAN/fanshishi//model/hd/CycleAE.pth"
EMRAEpath="/home/amax/Documents/Seagate4T/GAN/fanshishi//model/hd/EMRAE.pth"
classifierpath = "/home/amax/Documents/Seagate4T/GAN/fanshishi/model/hd/classifier.pth"
# DualVAE = "/home/amax/Documents/Seagate4T/GAN/fanshishi/model/aki/DualVAE.pth"
dataSavePath = "/home/amax/Documents/Seagate4T/GAN/fanshishi/data/SynData/onehot_Binary_hd/"
modelpath="/home/amax/Documents/Seagate4T/GAN/fanshishi/model/hd/"
Label = -1
D_size = 62
C_size = 16
latent_size=16
active = 'gumbel' #active== False;active== gumbel;active== onehot;
# active = False
# active = 'onehot'
Cols = [['D', 3], ['D', 7], ['D', 2], ['D', 3], ['D', 2], ['D', 3], ['D', 2], ['D', 3], ['D', 3], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['D', 2], ['C', 16]]
# '''Cols是与X中变量的类型和长度，其中D表示离散型，C表示连续型，
# 例如Cols=[["D",3],["D",5],["C",5],["D",7]]表示X的20列按顺序分别是离散型变量D1有3列、离散型变量D2有5列、连续型C1有5列，离散型D3有7列'''


emrlayer = compressDim

__all__=["STS"]
#(Gclassname=medgan_Generator,GinputDim=16,Glayer=[16,16],Dclassname=medgan_Discriminator,DinputDim=33,Dlayer=[16,8],Optionfunc=torch.optim.Adam,argparam={"lr":1e-3,"weight_decay":0.001,"eps":1e-07},AEpath="./model/AE.pth")


STS = {
    "dataSavePath":dataSavePath,
    "filename":filename,
    "AEpath":AEpath,
    "EMRAEpath":EMRAEpath,
    "epoch":2000,
    "createsize":3000,
    "Label":Label,
    "modelpath":modelpath,
    "Cpath":classifierpath,
    "D_size":D_size,
    "active":active,
    "Cols":Cols,
    "models":{
        "AE":{
            "filename":filename,
            "testname": testname,
            "lr":1e-3,
            "epochs":300,
            "Elayer":[compressDim],
            "Dlayer":[DinputDim]
        },
        "ClassifierModel":{
            "filename": filename,
            "testname": testname,
            "lr": 1e-3,
            "epochs": 1200,
#             "alph":0.1651165248639222,    #lung
#             "alph":0.601863232938025,    #crc
#             "alph":0.8801069032091458,    #breast
#             "alph":0.957,    #stroke
#             "alph":0.913,    #aki
#               "alph":0.5780918727915194,    #aki_mimic
            "alph":0.9707078925956062,    #hd
            "gamm":2,
            "inputDim": DinputDim,
            "layer": [DinputDim,DinputDim,DinputDim,1],
            "Optionfunc": torch.optim.Adam,
#             "Optionfunc": dadapt_adam.DAdaptAdam,
            "argparam": argparam,
            "classifierpath":classifierpath
        },
        "DualVAE":{
            "filename": filename,
            "testname": testname,
            "lr": 1e-3,
            "epochs": 1200,
            "modelarg":{
                "D_infeature":D_size,
                "C_infeature":DinputDim - D_size,
                "latent_size":latent_size,
                "D_encoderlayer":[D_size,D_size],
                "C_encoderlayer":[DinputDim - D_size,DinputDim - D_size],
                "D_decoderlayer":[D_size,D_size],
                "C_decoderlayer":[DinputDim - D_size,DinputDim - D_size],
            },
            "inputDim": DinputDim,
            "layer": [DinputDim,DinputDim,1],
            "Optionfunc": torch.optim.Adam,
#             "Optionfunc": dadapt_adam.DAdaptAdam,
            "argparam": argparam,
            "DualVAEpath":DualVAE
        },
        "SingleVAE":{
                    "filename": filename,
                    "testname": testname,
                    "lr": 1e-3,
                    "epochs": 1200,
                    "modelarg":{
                        "C_infeature":DinputDim+1,
                        "latent_size":latent_size,
                        "C_encoderlayer":[DinputDim+1,DinputDim+1],
                        "C_decoderlayer":[DinputDim+1,DinputDim+1],
                    },
                    # "inputDim": DinputDim,
                    # "layer": [DinputDim,DinputDim,1],
                    "Optionfunc": torch.optim.Adam,
                    "argparam": argparam,
                    "SingleVAEpath":DualVAE
        },
        "cycleMedGANtrainer":{
            "Gclassname": cyclemedganGenerator,
            "GinputDim": GinputDim,
            "Glayer": Glayer,
            "Dclassname": cyclemedganDiscriminator,
            "DinputDim": DinputDim,
            "Dlayer": Dlayer,
            "Optionfunc": torch.optim.Adam,
            "argparam": argparam,
            "AEpath": AEpath,
            "superarg":[0.0001,0.0001,0.0001]   #alph,bate,gamm
        },
        "cycleGANtrainer":{
            "Gclassname": cyclemedganGenerator,
            "GinputDim": GinputDim,
            "Glayer": Glayer+[DinputDim],
            "Dclassname": cyclemedganDiscriminator,
            "DinputDim": DinputDim,
            "Dlayer": Dlayer,
            "Optionfunc": torch.optim.Adam,
            "argparam": argparam,
            "AEpath": None,
            "superarg":[0.0001,0.0001,0.0001]   #alph,bate,gamm
        },
        "RGANtrainer": {
            "Gclassname": cyclemedganGenerator,
            "GinputDim": GinputDim,
            "Glayer": [GinputDim,GinputDim,GinputDim],
            "Dclassname": cyclemedganDiscriminator,
            "DinputDim": DinputDim,
            "Dlayer": Dlayer,
            "Optionfunc": torch.optim.Adam,
            "argparam": argparam,
            "AEpath": None,
            "Classifier":classifierpath,
            "superarg":[0.0001,0.0001,0.0001]   #alph,bate,gamm
        },
        "RGAN_ncTrainer": {
            "Gclassname": cyclemedganGenerator,
            "GinputDim": GinputDim,
            "Glayer": [GinputDim,GinputDim,GinputDim],
            "Dclassname": cyclemedganDiscriminator,
            "DinputDim": DinputDim,
            "Dlayer": Dlayer,
            "Optionfunc": torch.optim.Adam,
            "argparam": argparam,
            "AEpath": None,
            "Classifier":classifierpath,
            "superarg":[0.0001,0.0001,0.0001]   #alph,bate,gamm
        },
        "cycleRGANConsistTrainer": {
            "Gclassname": cyclemedganGenerator,
            "GinputDim": GinputDim,
            "Glayer": [GinputDim, GinputDim, GinputDim],
            "Dclassname": cyclemedganDiscriminator,
            "DinputDim": DinputDim,
            "Dlayer": Dlayer,
            "Optionfunc": torch.optim.Adam,
            "argparam": argparam,
            "AEpath": None,
            "Classifier": classifierpath,
            "superarg":[0.0001,0.0001,0.0001]   #alph,bate,gamm
        },
        "cycleRGANTrainer": {
            "Gclassname": cyclemedganGenerator,
            "GinputDim": GinputDim,
            "Glayer": [GinputDim, GinputDim, GinputDim],
            "Dclassname": cyclemedganDiscriminator,
            "DinputDim": DinputDim,
            "Dlayer": Dlayer,
            "Optionfunc": torch.optim.Adam,
            "argparam": argparam,
            "AEpath": None,
            "Classifier": classifierpath,
            "superarg": [0.0001, 0.0001, 0.0001]  # alph,bate,gamm
        },
        "cycleRGAN_ncTrainer": {
            "Gclassname": cyclemedganGenerator,
            "GinputDim": GinputDim,
            "Glayer": [GinputDim, GinputDim, GinputDim],
            "Dclassname": cyclemedganDiscriminator,
            "DinputDim": DinputDim,
            "Dlayer": Dlayer,
            "Optionfunc": torch.optim.Adam,
            "argparam": argparam,
            "AEpath": None,
            "Classifier": classifierpath,
            "superarg": [0.0001, 0.0001, 0.0001]  # alph,bate,gamm
        },
        "Dual_RGANTrainer": {
            "Gclassname": cyclemedganGenerator,
            "GinputDim": latent_size,
            "Glayer": [latent_size,latent_size,latent_size],
            "Dclassname": cyclemedganDiscriminator,
            "DinputDim": DinputDim,
            "Dlayer": Dlayer,
            "Optionfunc": torch.optim.Adam,
            "argparam": argparam,
            "AEpath": None,
            "Classifier":classifierpath,
            "superarg":[0.0001,0.0001,0.0001],   #alph,bate,gamm
            "DualVAEpath":DualVAE
        },
        "Dual_cycleRGANTrainer": {
            "Gclassname": cyclemedganGenerator,
            "GinputDim": latent_size,
            "Glayer": [latent_size,latent_size,latent_size],
            "Dclassname": cyclemedganDiscriminator,
            "DinputDim": DinputDim,
            "Dlayer": Dlayer,
            "Optionfunc": torch.optim.Adam,
            "argparam": argparam,
            "AEpath": None,
            "Classifier":classifierpath,
            "superarg":[0.0001,0.0001,0.0001],   #alph,bate,gamm
            "DualVAEpath":DualVAE
        },
        "Dual_cycleRGANConsistTrainer": {
            "Gclassname": cyclemedganGenerator,
            "GinputDim": latent_size,
            "Glayer": [latent_size,latent_size,latent_size],
            "Dclassname": cyclemedganDiscriminator,
            "DinputDim": DinputDim,
            "Dlayer": Dlayer,
            "Optionfunc": torch.optim.Adam,
            "argparam": argparam,
            "AEpath": None,
            "Classifier":classifierpath,
            "superarg":[0.0001,0.0001,0.0001],   #alph,bate,gamm
            "DualVAEpath":DualVAE
        },
        "EMRAE": {
            "filename": filename,
            "testname": testname,
            "lr": 1e-3,
            "epochs": 300,
            "Elayer": [compressDim],
            "Dlayer": [DinputDim]
        },
        "EMR_RGANTrainer": {
            "Gclassname": emrn_Generator,
            "GinputDim": emrlayer,
            "Glayer": Glayer,
#             "Glayer": [emrlayer, emrlayer, emrlayer],
#             "Dclassname": cyclemedganDiscriminator,
            "Dclassname": emr_Discriminator,
            "DinputDim": DinputDim,
            "Dlayer": Dlayer,
            "Optionfunc": torch.optim.Adam,
            "argparam": argparam,
            "AEpath": None,
            "Classifier": classifierpath,
            "superarg": [0.0001,0.0001, 0.0001],  # alph,bate,gamm
            "EMRAEpath": EMRAEpath
        },
        "EMR_cycleRGANTrainer": {
            "Gclassname": emrn_Generator,
            "GinputDim": emrlayer,
            "Glayer": Glayer,
#             "Glayer": [emrlayer, emrlayer, emrlayer],
#             "Dclassname": cyclemedganDiscriminator,
            "Dclassname": emr_Discriminator,
            "DinputDim": DinputDim,
            "Dlayer": Dlayer,
            "Optionfunc": torch.optim.Adam,
            "argparam": argparam,
            "AEpath": None,
            "Classifier": classifierpath,
            "superarg": [0.0001, 0.0001, 0.0001],  # alph,bate,gamm
            "EMRAEpath": EMRAEpath
        },
        "EMR_cycleRGANConsistTrainer": {
            "Gclassname": emrn_Generator,
            "GinputDim": emrlayer,
            "Glayer": Glayer,
#             "Glayer": [emrlayer, emrlayer, emrlayer],
#             "Dclassname": cyclemedganDiscriminator,
            "Dclassname": emr_Discriminator,
            "DinputDim": DinputDim,
            "Dlayer": Dlayer,
            "Optionfunc": torch.optim.Adam,
            "argparam": argparam,
            "AEpath": None,
            "Classifier": classifierpath,
            "superarg": [0.0001, 0.0001, 0.0001],  # alph,bate,gamm
            "EMRAEpath": EMRAEpath
        },
    }
}

