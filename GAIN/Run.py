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
file='measureGenerator_noRand'
data_path="{}\\data\\{}.csv".format(Path(getcwd()).parent,file)
Data=DataModel(data_path)

#%% Models
Dim=Data.Dim
Generator=myGenerator(compositLayers([Dim*10,Dim*10,Dim*10,Dim*10,Dim*20,Dim*20,Dim*20,Dim*50,Dim*200,Dim],0))
Discriminator=myDiscriminator(compositLayers([Dim*10,Dim*10,Dim*10,Dim*20,Dim*20,Dim*20,Dim*50,Dim*200,Dim],0))
Model1=GAN(Generator,Discriminator,hyperParams={'p_miss':0.5},optimizer=tf.keras.optimizers.Adam())
Model2=GAN(Generator,Discriminator,summary_writer=Model1.summary_writer,hyperParams={'p_miss':0.5},optimizer=tf.keras.optimizers.Adam())
Model3=GAN(Generator,Discriminator,summary_writer=Model1.summary_writer,hyperParams={'p_miss':0.5},optimizer=tf.keras.optimizers.Adam())
#%% Run - 1 : 
# First train the discriminator against a random generator to increase its stability
counter=0
train,test=Data.getPipeLine(p_miss=0.5,p_hints=0.5,train_rate=0.8,batch_ratio=0.2,repeat=1000)
test=iter(test)
for dat_train,[mask,hint_mask,hints] in tqdm(train):
    Model1.train_discriminator_with_random(dat_train,mask,hints)
    if counter%5:
        Model1.tensorboard_log_with_random('Step 1.',dat_train,mask,hints,hint_mask)
    counter+=1
#%% Run - 2 : 
# First train the discriminator against a proper generator
counter=0
train,test=Data.getPipeLine(p_miss=0.5,p_hints=0.5,train_rate=0.8,batch_ratio=0.2,repeat=1000)
test=iter(test)
for dat_train,[mask,hint_mask,hints] in tqdm(train):
    Model2.train_discriminator(dat_train,mask,hints,steps=5)
    Model2.train_generator(dat_train,mask,hints)
    if counter%5:
        Model2.tensorboard_log('Step 2.',dat_train,mask,hints,hint_mask)
    counter+=1
    

#%% Run - Step 3
# Training with discriminator unrolling
counter=0
train,test=Data.getPipeLine(p_miss=0.5,p_hints=0.5,train_rate=0.8,batch_ratio=0.2,repeat=10)
test=iter(test)
Discriminator.intiateUnrolling(Data.Dim)
for dat_train,[mask,hint_mask,hints] in tqdm(train):
    Model3.train_discriminator(dat_train,mask,hints)
    Model3.unroll_discriminator(dat_train,mask,hints)
    Model3.train_generator_with_discriminator_unrolled(dat_train,mask,hints)
    if counter%5:
        Model3.tensorboard_log('Step 3.',dat_train,mask,hints,hint_mask)
    counter+=1


# %%
