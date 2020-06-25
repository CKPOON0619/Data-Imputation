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
# Due to existing limitation of tensorflow api, 
# each GAN model could not be reused for another adversarial pair: 
# https://github.com/tensorflow/tensorflow/issues/27120
# Should use one single model once the issue is resolved.
Model1=GAN(hyperParams={'p_miss':0.5},optimizer=tf.keras.optimizers.Adam())
Model2=GAN(summary_writer=Model1.summary_writer,hyperParams={'p_miss':0.5},optimizer=tf.keras.optimizers.Adam())
Model3=GAN(summary_writer=Model1.summary_writer,hyperParams={'p_miss':0.5, 'alpha':0.1, 'episode_num':10 },optimizer=tf.keras.optimizers.Adam(1e-3))
#%% Run - Step 1
# First train the discriminator against a random generator to increase its stability
counter=0
train,test=Data.getPipeLine(p_miss=0.5,p_hints=0.5,train_rate=0.8,batch_ratio=0.05,repeat=1)
test=iter(test)
for dat_train,[mask,hint_mask,hints] in tqdm(train):
    Model1.trainWithSteps(dat_train,mask,hints,randomGenerator,Discriminator,steps=False)
    if(counter%20==0):
        Model1.performanceLog('<Random Generator>(train)',dat_train,mask,hint_mask,hints,randomGenerator,Discriminator)
        dat_test,[test_mask,test_hint_mask,test_hints]=test.next()
        Model1.performanceLog('<Random Generator>(test)',dat_test,test_mask,test_hint_mask,test_hints,randomGenerator,Discriminator)    
    counter+=1    
    
#%% Run - Step 2
# Then train the discriminator against a train-able generator model without unrolling
counter=0
model_counter=0
train,test=Data.getPipeLine(p_miss=0.5,p_hints=0.5,train_rate=0.8,batch_ratio=0.05,repeat=2)
test=iter(test)
mem_Discriminator=Memorise(Discriminator,100, Data.train_batch_size,[Data.Dim,Data.Dim])
mask_cache=Memorise(lambda x:x,100,Data.train_batch_size,[Data.Dim])
for dat_train,[mask,hint_mask,hints] in tqdm(train):
    mask_cache(mask)
    mask_recalled=mask_cache.recall_memory(0)
    generated_x,x_hat=Generator(dat_train,mask)
    Model2.trainDiscriminator(dat_train,x_hat,mask_recalled,hints,mem_Discriminator)
    if(counter%20==0):
        Model2.performanceLog('<Generator>(train)',dat_train,mask,hint_mask,hints,Generator,Discriminator)
        dat_test,[test_mask,test_hint_mask,test_hints]=test.next()
        Model2.performanceLog('<Generator>(test)',dat_test,test_mask,test_hint_mask,test_hints,Generator,Discriminator)
    counter+=1

#%% Run - Step 2
# Then train the generator with discriminator unrolling. 
# Discriminator trained with memory replay
counter=0
model_counter=0
train,test=Data.getPipeLine(p_miss=0.5,p_hints=0.5,train_rate=0.8,batch_ratio=0.05,repeat=2)
test=iter(test)
episodes=Model3.initialiseUnRolling(Discriminator,myDiscriminator,Data.Dim)
mem_Discriminator=Memorise(Discriminator,20, Data.train_batch_size,[Data.Dim,Data.Dim])
mask_cache=Memorise(lambda x:x,20,Data.train_batch_size,[Data.Dim])
for dat_train,[mask,hint_mask,hints] in tqdm(train):    
    generated_x,x_hat=Model3.generate(Generator,dat_train,mask)
    Model3.trainDiscriminator(dat_train,x_hat,mask_cache(mask),hints,mem_Discriminator)
    Model3.unrollDiscriminator(dat_train,mask,hints,x_hat,Discriminator)
    Model3.trainGeneratorWithDiscriminators(dat_train,mask,hints,Generator,episodes)
    if(counter%20==0):
        Model3.performanceLog('<Generator>(train)',dat_train,mask,hint_mask,hints,Generator,Discriminator)
        dat_test,[test_mask,test_hint_mask,test_hints]=test.next()
        Model3.performanceLog('<Generator>(test)',dat_test,test_mask,test_hint_mask,test_hints,Generator,Discriminator)
    counter+=1



# %%
