#%% Packages
import numpy as np
import tensorflow as tf
from components.Generator import myGenerator
from components.Discriminator import myDiscriminator
from components.Critic import myCritic
from components.CompositLayers import CompositLayers
from components.AttentionLayers import CompositeAttentionLayers
from components.Memorisor import Memorise
from DataModel import DataModel
from tqdm import tqdm
from os import getcwd
from WGAN import WGAN
from GAN import GAN
from pathlib import Path

#%% Data Model
file='measureGenerator'
data_path="{}\\data\\{}.csv".format(Path(getcwd()).parent,file)
Data=DataModel(data_path,rangeBoost=0)

#%% Models
Dim=Data.Dim
# Generator=myGenerator(CompositLayers([Dim*5,Dim*5,Dim*10,Dim*10,Dim*15,Dim*15,Dim*15,Dim*10,Dim*10,Dim],0))
# Critic=myCritic(CompositLayers([Dim*2,Dim*2,Dim*2,Dim*3,Dim*3,Dim*2,Dim*2,Dim],0))
# Discriminator=myDiscriminator(CompositLayers([Dim*2,Dim*3,Dim*3,Dim*3,Dim*2,Dim*2,Dim*2,Dim],0))

Generator=myGenerator(CompositeAttentionLayers(6,4,4,6))
Critic=myCritic(CompositeAttentionLayers(6,4,4,6))
Discriminator=myDiscriminator(CompositeAttentionLayers(6,4,4,6))

Model1=WGAN(Generator,Critic,hyperParams={'p_miss':0.5},optimizer=tf.keras.optimizers.Adam())
Model2=WGAN(Generator,Critic,summary_writer=Model1.summary_writer,hyperParams={'p_miss':0.5},optimizer=tf.keras.optimizers.Adam())
Model3=GAN(Generator,Discriminator,summary_writer=Model1.summary_writer,hyperParams={'p_miss':0.5},optimizer=tf.keras.optimizers.Adam())
Model4=GAN(Generator,Discriminator,summary_writer=Model1.summary_writer,hyperParams={'p_miss':0.5},optimizer=tf.keras.optimizers.Adam())

# %% Run with random
counter=0
train,test=Data.getPipeLine(p_miss=0.5,p_hints=0.5,train_rate=0.8,batch_ratio=1,repeat=2000)
test=iter(test)
for dat_train,[fix_mask,mask,hint_mask,hints] in tqdm(train):
    Model1.train_critic_with_random(dat_train,fix_mask,mask,hints)
    Model3.train_discriminator_with_random(dat_train,mask,hints)
    if counter%5==0:
        Model1.tensorboard_log_with_random("Step 1.",dat_train,fix_mask,mask,hints,hint_mask)
        Model3.tensorboard_log_with_random("step 1.",dat_train,mask,hints,hint_mask)
    counter+=1
#%%
Model1.critic_scan(dat_train)
Model3.discriminator_scan(dat_train)

#%% Run - 1 : 
counter=0
train,test=Data.getPipeLine(p_miss=0.5,p_hints=0.5,train_rate=0.8,batch_ratio=1,repeat=150)
test=iter(test)
for dat_train,[fix_mask,mask,hint_mask,hints] in tqdm(train):
    Model2.train_critic(dat_train,fix_mask,mask,hints,steps=5)
    Model4.train_discriminator(dat_train,mask,hints,steps=5)
    
    Model2.train_generator(dat_train,mask,hints)
    Model4.train_generator(dat_train,mask,hints)
    if counter%5:
        Model2.tensorboard_log('Step 2. Critic:',dat_train,fix_mask,mask,hints,hint_mask)
        Model4.tensorboard_log("step 2. Discriminator",dat_train,mask,hints,hint_mask)
    counter+=1

# %%
