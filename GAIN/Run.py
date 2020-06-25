#%% Packages
import numpy as np
import tensorflow as tf
from components.Generator import myGenerator
from components.Discriminator import myDiscriminator
from components.NetworkComponents import compositLayers
from components.Memorisor import Memorise
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
Generator=myGenerator(compositLayers([Dim*10,Dim*10,Dim*10,Dim*10,Dim*20,Dim*20,Dim*20,Dim*50,Dim*200,Dim],0))
Discriminator=myDiscriminator(compositLayers([Dim*10,Dim*10,Dim*10,Dim*20,Dim*20,Dim*20,Dim*50,Dim*200,Dim],0))
Model1=GAN(hyperParams={'p_miss':0.5},optimizer=tf.keras.optimizers.Adam())
#%% Run - 1 : 
counter=0
train,test=Data.getPipeLine(p_miss=0.5,p_hints=0.5,train_rate=0.8,batch_ratio=0.05,repeat=200)
test=iter(test)
for dat_train,[mask,hint_mask,hints] in tqdm(train):
    Model1.train(dat_train,mask,hints,Generator,Discriminator,steps=5)
    if counter%5:
        Model1.tensorboard_log('Step 1.',dat_train,mask,hints,hint_mask,randomGenerator,Discriminator)
    counter+=1
    

#%% Run - Step 2
# First train the discriminator against a random generator to increase its stability
counter=0
train,test=Data.getPipeLine(p_miss=0.5,p_hints=0.5,train_rate=0.8,batch_ratio=0.05,repeat=1)
test=iter(test)
Discriminator.intiateUnrolling(Data.Dim)
for dat_train,[mask,hint_mask,hints] in tqdm(train):
    Model1.train_with_unrolling(dat_train,mask,hints,Generator,Discriminator)
    if counter%5:
        Model1.tensorboard_log('Step 2.',dat_train,mask,hints,hint_mask,Generator,Discriminator)
    counter+=1


# %%
