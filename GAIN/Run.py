#%% Packages
import numpy as np
import tensorflow as tf
from components.Generator import myGenerator
from components.Discriminator import myDiscriminator
from components.NetworkComponents import compositLayers
from DataModel import DataModel
from tqdm import tqdm
from os import getcwd
from GAN import GAN
from pathlib import Path

#%% Data Model
file='measureGenerator'
data_path="{}\\data\\{}.csv".format(Path(getcwd()).parent,file)
Data=DataModel(data_path)
#%% Models
Dim=Data.Dim
randomGenerator=myGenerator()
Generator=myGenerator(compositLayers([Dim*12,Dim*6,Dim*3,Dim*2,Dim*3,Dim*6,Dim*12,Dim],0))
Discriminator=myDiscriminator(compositLayers([Dim*12,Dim*6,Dim*3,Dim*2,Dim*3,Dim*6,Dim*12,Dim],0))
# Due to existing limitation of tensorflow api, 
# each GAN model could not be reused for another adversarial pair: 
# https://github.com/tensorflow/tensorflow/issues/27120
# Should use one single model once the issue is resolved.
Model1=GAN(hyperParams={'G_train_step':1})
Model2=GAN(logdir=Model1.logdir,hyperParams={'G_train_step':1})
#%% Run
# First train the discriminator against a random generator to increase its stability
counter=0
for dat_train,dat_test in tqdm(Data.getPipeLine(train_rate=0.8,batch_ratio=0.2,repeat=500)):
    Model1.trainWithBatch(dat_train,randomGenerator,Discriminator)
    # Model.trainWithBatch(dat_train,Generator,Discriminator)
    if(counter%20==0):
        Model1.performanceLog('<Random Generator>(train)',dat_train,randomGenerator,Discriminator)
        Model1.performanceLog('<Random Generator>(test)',dat_test,randomGenerator,Discriminator)
#%%
# Then train the discriminator against a train-able generator model.
counter=0
for dat_train,dat_test in tqdm(Data.getPipeLine(train_rate=0.8,batch_ratio=0.2,repeat=10)):
    Model2.trainWithBatch(dat_train,Generator,Discriminator)
    if(counter%20==0):
        Model2.performanceLog('<Generator>(train)',dat_train,Generator,Discriminator)
        Model2.performanceLog('<Generator>(test)',dat_test,Generator,Discriminator)