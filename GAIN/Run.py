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
Generator=myGenerator(compositLayers([Dim*10,Dim*10,Dim*10,Dim*10,Dim*20,Dim*20,Dim*20,Dim*50,Dim*200,Dim],0))
Discriminator=myDiscriminator(compositLayers([Dim*10,Dim*10,Dim*10,Dim*20,Dim*20,Dim*20,Dim*50,Dim*200,Dim],0))
# Due to existing limitation of tensorflow api, 
# each GAN model could not be reused for another adversarial pair: 
# https://github.com/tensorflow/tensorflow/issues/27120
# Should use one single model once the issue is resolved.
Model1=GAN(hyperParams={'p_miss':0.5, 'p_hint':0.5},optimizer=tf.keras.optimizers.Adam())
Model2=GAN(summary_writer=Model1.summary_writer,hyperParams={'p_miss':0.5, 'p_hint':0.5},optimizer=tf.keras.optimizers.Adam())
Model3=GAN(summary_writer=Model1.summary_writer,hyperParams={'p_miss':0.5, 'p_miss':0.5, 'alpha':0.1, 'episode_num':10 },optimizer=tf.keras.optimizers.Adam(1e-4))
#%% Run - Step 1
# First train the discriminator against a random generator to increase its stability
counter=0
train,test=Data.getPipeLine(train_rate=0.8,batch_ratio=1,repeat=50)
test=iter(test)
for dat_train in tqdm(train):
    Model1.trainWithSteps(dat_train,randomGenerator,Discriminator,steps=False)
    if(counter%20==0):
        Model1.performanceLog('<Random Generator>(train)',dat_train,randomGenerator,Discriminator)
        Model1.performanceLog('<Random Generator>(test)',test.next(),randomGenerator,Discriminator)    
    counter+=1    
    
Discriminator.save(Model1.logdir+'Models\DiscriminatorS1E{}'.format(1))
#%% Run - Step 2
# Then train the discriminator against a train-able generator model without unrolling
counter=0
model_counter=0
train,test=Data.getPipeLine(train_rate=0.8,batch_ratio=1,repeat=5)
test=iter(test)
for dat_train in tqdm(train):
    Model2.trainWithSteps(dat_train,Generator,Discriminator)
    if(counter%20==0):
        Model2.performanceLog('<Generator>(train)',dat_train,Generator,Discriminator)
        Model2.performanceLog('<Generator>(test)',test.next(),Generator,Discriminator)
    counter+=1
Generator.save(Model1.logdir+'\Models\GeneratorS2E{}'.format(1))
Discriminator.save(Model1.logdir+'\Models\DiscriminatorS2E{}'.format(1))

#%% Run - Step 3 (unrolled)
# Then train the generator and discriminator with discriminator unrolling. 
counter=0
model_counter=0
train,test=Data.getPipeLine(train_rate=0.8,batch_ratio=1,repeat=100)
test=iter(test)
episodes=Model3.initialiseEpisodes(Discriminator,myDiscriminator)
for dat_train in tqdm(train):
    Model3.unrollDiscriminator(dat_train,Generator,Discriminator,leap=1)
    Model3.trainGeneratorWithDiscriminators(dat_train,Generator,episodes)
    if(counter%20==0):
        Model3.performanceLog('<Generator>(train)',dat_train,Generator,Discriminator)
        Model3.performanceLog('<Generator>(test)',test.next(),Generator,Discriminator)
    counter+=1
Generator.save(Model1.logdir+'\GeneratorS3E{}'.format(1))
Discriminator.save(Model1.logdir+'\DiscriminatorS3E{}'.format(1))



# %%
