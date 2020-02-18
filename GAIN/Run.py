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
Generator=myGenerator(compositLayers([Dim*100,Dim*100,Dim*100,Dim*100,Dim*100,Dim*100,Dim*100,Dim*100,Dim*200,Dim],0.2))
Discriminator=myDiscriminator(compositLayers([Dim*100,Dim*100,Dim*100,Dim*100,Dim*100,Dim*100,Dim*100,Dim],0.2))
Discriminator1=myDiscriminator(compositLayers([Dim*100,Dim*100,Dim*100,Dim*100,Dim*100,Dim*100,Dim*100,Dim],0.2))
Discriminator2=myDiscriminator(compositLayers([Dim*100,Dim*100,Dim*100,Dim*100,Dim*100,Dim*100,Dim*100,Dim],0.2))
Discriminator3=myDiscriminator(compositLayers([Dim*100,Dim*100,Dim*100,Dim*100,Dim*100,Dim*100,Dim*100,Dim],0.2))
Discriminator4=myDiscriminator(compositLayers([Dim*100,Dim*100,Dim*100,Dim*100,Dim*100,Dim*100,Dim*100,Dim],0.2))
# Due to existing limitation of tensorflow api, 
# each GAN model could not be reused for another adversarial pair: 
# https://github.com/tensorflow/tensorflow/issues/27120
# Should use one single model once the issue is resolved.
Model1=GAN(hyperParams={'p_miss':1/6, 'p_hint':1/6},optimizer=tf.keras.optimizers.Adam())
Model2=GAN(logdir=Model1.logdir,hyperParams={'p_miss':1/6, 'p_miss':1/6},optimizer=tf.keras.optimizers.Adam())

#%% Run - Step 1
# First train the discriminator against a random generator to increase its stability
counter=0
train,test=Data.getPipeLine(train_rate=0.8,batch_ratio=1,repeat=10)
test=iter(test)
for dat_train in tqdm(train):
    Model1.trainWithSteps(dat_train,randomGenerator,Discriminator)
    if(counter>0 and counter%20==0):
        Model1.performanceLog('<Random Generator>(train)',dat_train,randomGenerator,Discriminator)
        Model1.performanceLog('<Random Generator>(test)',test.next(),randomGenerator,Discriminator)    
    if(counter>0 and counter%100==0):
        Discriminator.save(Model1.logdir+'\DiscriminatorS1E{}'.format(Model1.epoch))
        
#%% Run - Step 2 (multi generators and discriminators)
## Then train the generators and discriminators
counter=0
train,test=Data.getPipeLine(train_rate=0.8,batch_ratio=1,repeat=10)
test=iter(test)
Discriminators=[Discriminator,Discriminator1,Discriminator2,Discriminator3,Discriminator4]
for dat_train in tqdm(train):
    Model2.trainGeneratorWithDiscriminators(dat_train,Generator,Discriminators)
    Model2.trainDiscriminators(dat_train,Generator,Discriminators)
    # if(counter>0 and counter%20==0):
    #     Model2.performanceLog('<Generator>(train)',dat_train,Generator,Discriminator)
    #     Model2.performanceLog('<Generator>(test)',test.next(),Generator,Discriminator)
    # if(counter>0 and counter%100==0):
    #     Generator.save(Model2.logdir+'\GeneratorS2E{}'.format(Model2.epoch))
    #     Discriminator.save(Model2.logdir+'\DiscriminatorS2E{}'.format(Model2.epoch))
        
#%% Run - Step 2 (unrolled)
## Then train the generator and discriminator with discriminator unrolling. 
# counter=0
# train,test=Data.getPipeLine(train_rate=0.8,batch_ratio=1,repeat=3000)
# test=iter(test)
# Model2.initialiseEpisodes(Discriminator,myDiscriminator)
# for dat_train in tqdm(train):
#     Model2.unrollDiscriminator(dat_train,Generator,Discriminator)
#     Model2.trainGeneratorWithEpisodes(dat_train,Generator)
#     if(counter>0 and counter%20==0):
#         Model2.performanceLog('<Generator>(train)',dat_train,Generator,Discriminator)
#         Model2.performanceLog('<Generator>(test)',test.next(),Generator,Discriminator)
#     if(counter>0 and counter%100==0):
#         Generator.save(Model2.logdir+'\GeneratorS2E{}'.format(Model2.epoch))
#         Discriminator.save(Model2.logdir+'\DiscriminatorS2E{}'.format(Model2.epoch))

#%% Run - Step 2 (vanilla)
## Then train the discriminator against a train-able generator model without unrolling
# counter=0
# train,test=Data.getPipeLine(train_rate=0.8,batch_ratio=1,repeat=1)
# test=iter(test)
# Model2=GAN(logdir=Model1.logdir,hyperParams={'p_miss':1/6},optimizer=tf.keras.optimizers.Adam(1e-4))
# for dat_train in tqdm(train):
#     Model1.trainWithSteps(dat_train,randomGenerator,Discriminator)
#     if(counter>0 and counter%20==0):
#         Model2.performanceLog('<Generator>(train)',dat_train,Generator,Discriminator)
#         Model2.performanceLog('<Generator>(test)',test.next(),Generator,Discriminator)
