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

#%% Helpers
# def cloneDiscriminator(model,myModel):
#     newModel=myModel()
#     newModel.body=tf.keras.models.clone_model(model.body)
#     return newModel
#%% Data Model
file='measureGenerator'
data_path="{}\\data\\{}.csv".format(Path(getcwd()).parent,file)
Data=DataModel(data_path)
#%% Models
Dim=Data.Dim
randomGenerator=myGenerator()
Generator=myGenerator(compositLayers([Dim*48,Dim*48,Dim*48,Dim*15,Dim*15,Dim*10,Dim*15,Dim*15,Dim*48,Dim],0.2))
Discriminator=myDiscriminator(compositLayers([Dim*48,Dim*15,Dim*15,Dim*10,Dim*15,Dim*15,Dim*48,Dim],0.2))
Discriminator2=myDiscriminator(compositLayers([Dim*48,Dim*15,Dim*15,Dim*10,Dim*15,Dim*15,Dim*48,Dim],0.2))
Discriminator3=myDiscriminator(compositLayers([Dim*48,Dim*15,Dim*15,Dim*10,Dim*15,Dim*15,Dim*48,Dim],0.2))
# Due to existing limitation of tensorflow api, 
# each GAN model could not be reused for another adversarial pair: 
# https://github.com/tensorflow/tensorflow/issues/27120
# Should use one single model once the issue is resolved.
#%% Run
# First train the discriminator against a random generator to increase its stability
counter=0
train,test=Data.getPipeLine(train_rate=0.8,batch_ratio=1,repeat=10)
test=iter(test)
Model1=GAN(hyperParams={'p_miss':1/6},optimizer=tf.keras.optimizers.Adam())
Model2=GAN(hyperParams={'p_miss':1/6},optimizer=tf.keras.optimizers.Adam())
Model3=GAN(hyperParams={'p_miss':1/6},optimizer=tf.keras.optimizers.Adam())
for dat_train in tqdm(train):
    Model1.trainWithSteps(dat_train,randomGenerator,Discriminator)
    Model2.trainWithSteps(dat_train,randomGenerator,Discriminator2)
    # Model.trainWithBatch(dat_train,Generator,Discriminator)
    if(counter%20==0):
        Model1.performanceLog('<Random Generator>(train)',dat_train,randomGenerator,Discriminator)
        Model2.performanceLog('<Random Generator>(test)',test.next(),randomGenerator,Discriminator)
#%%
# Then train the discriminator against a train-able generator model.
counter=0
train,test=Data.getPipeLine(train_rate=0.8,batch_ratio=1,repeat=2000)
test=iter(test)
Model2=GAN(logdir=Model1.logdir,hyperParams={'p_miss':1/6},optimizer=tf.keras.optimizers.Adam(1e-4))
for dat_train in tqdm(train):
    if(counter%20==0):
        Model2.performanceLog('<Generator>(train)',dat_train,Generator,Discriminator)
        Model2.performanceLog('<Generator>(test)',test.next(),Generator,Discriminator)

# %%

def cloneDiscriminator(model,myModel):
    newModel=myModel()
    newModel.body=tf.keras.models.clone_model(model.body)
    # newModel.trainable_variables
    return newModel

# %%
D3=clonePlayer(Discriminator,myDiscriminator)

# %%
D3.trainable_variables=[tf.Variable(var,shape=tf.shape(var)).assign(var) for var in Discriminator.trainable_variables]


# %%
