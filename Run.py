#%% Packages
import numpy as np
import tensorflow as tf
from components.Generator import myGenerator
from components.Discriminator import myDiscriminator
from DataModel import DataModel
from os import getcwd
from GAN import GAN

#%% Data Model
file='measureGenerator'
data_path="{}\\data\\{}.csv".format(getcwd(),file)
Data=DataModel(data_path)
data_pipeline=Data.getPipeLine(train_rate=0.8,batch_num=10,repeat=20)
#%% Models
Generator=myGenerator(Data.Dim,0)
Discriminator=myDiscriminator(Data.Dim,0)
Model=GAN()

#%% Run
epoch=0
for dat_train,dat_test in data_pipeline:
    epoch+=1
    print(epoch)
    Model.trainWithBatch(dat_train,Generator,Discriminator)
    Model.performanceLog('Test',dat_test,Generator,Discriminator)

# %%
