#%% Packages
import numpy as np
import tensorflow as tf
from components.Encoder import Encoder
from components.Decoder import Decoder
from components.NetworkComponents import compositLayers
from DataModel import DataModel
from tqdm import tqdm
from os import getcwd
from VAE import VAE
from pathlib import Path

#%% Data Model
file='measureGenerator'
data_path="{}\\data\\{}.csv".format(Path(getcwd()).parent,file)
Data=DataModel(data_path)

#%% Models
Dim=Data.Dim
encoder=Encoder(compositLayers([Dim*12,Dim*6,Dim*3,Dim*2,Dim*3,Dim*6,Dim*12,Dim],0))
decoder=Decoder(compositLayers([Dim*12,Dim*6,Dim*3,Dim*2,Dim*3,Dim*6,Dim*12,Dim],0))
Model=VAE(hyperParams={'G_train_step':1})

#%% Run
counter=0
for dat_train,dat_test in tqdm(Data.getPipeLine(train_rate=0.8,batch_ratio=0.2,repeat=500)):
    Model.trainWithBatch(dat_train,encoder,decoder)

# %%
