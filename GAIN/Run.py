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
Generator=myGenerator(compositLayers([Dim*48,Dim*48,Dim*48,Dim*15,Dim*15,Dim*10,Dim*15,Dim*15,Dim*48,Dim],0.2))
Discriminator=myDiscriminator(compositLayers([Dim*48,Dim*15,Dim*15,Dim*10,Dim*15,Dim*15,Dim*48,Dim],0.2))
# Due to existing limitation of tensorflow api, 
# each GAN model could not be reused for another adversarial pair: 
# https://github.com/tensorflow/tensorflow/issues/27120
# Should use one single model once the issue is resolved.
Model1=GAN(hyperParams={'p_miss':1/6, 'p_hint':1/6},optimizer=tf.keras.optimizers.Adam())
Model2=GAN(logdir=Model1.logdir,hyperParams={'p_miss':1/6, 'p_miss':1/6},optimizer=tf.keras.optimizers.Adam(1e-4))
#%% Run - Step 1
# First train the discriminator against a random generator to increase its stability
counter=0
train,test=Data.getPipeLine(train_rate=0.8,batch_ratio=1,repeat=2000)
test=iter(test)
for dat_train in tqdm(train):
    Model1.trainWithSteps(dat_train,randomGenerator,Discriminator)
    if(counter%20==0):
        Model1.performanceLog('<Random Generator>(train)',dat_train,randomGenerator,Discriminator)
        Model1.performanceLog('<Random Generator>(test)',test.next(),randomGenerator,Discriminator)
#%% Run - Step 2 (unrolled)
# Then train the generator and discriminator with discriminator unrolling. 
counter=0
train,test=Data.getPipeLine(train_rate=0.8,batch_ratio=1,repeat=400)
test=iter(test)
Model2.initialiseEpisodes(Discriminator,myDiscriminator)
for dat_train in tqdm(train):
    Model2.unrollDiscriminator(dat_train,Generator,Discriminator)
    Model2.trainGeneratorWithEpisodes(dat_train,Generator,Discriminator)
    if(counter%20==0):
        Model2.performanceLog('<Generator>(train)',dat_train,Generator,Discriminator)
        Model2.performanceLog('<Generator>(test)',test.next(),Generator,Discriminator)


#%% Run - Step 2
# Then train the discriminator against a train-able generator model without unrolling
# counter=0
# train,test=Data.getPipeLine(train_rate=0.8,batch_ratio=1,repeat=1)
# test=iter(test)
# Model2=GAN(logdir=Model1.logdir,hyperParams={'p_miss':1/6},optimizer=tf.keras.optimizers.Adam(1e-4))
# for dat_train in tqdm(train):
#     Model1.trainWithSteps(dat_train,randomGenerator,Discriminator)
#     if(counter%20==0):
#         Model2.performanceLog('<Generator>(train)',dat_train,Generator,Discriminator)
#         Model2.performanceLog('<Generator>(test)',test.next(),Generator,Discriminator)
