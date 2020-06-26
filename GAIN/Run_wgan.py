#%% Packages
import numpy as np
import tensorflow as tf
from components.Generator import myGenerator
from components.Critic import myCritic
from components.NetworkComponents import compositLayers
from components.Memorisor import Memorise
from DataModel import DataModel
from tqdm import tqdm
from os import getcwd
from WGAN import WGAN
from pathlib import Path

#%% Data Model
file='measureGenerator'
data_path="{}\\data\\{}.csv".format(Path(getcwd()).parent,file)
Data=DataModel(data_path)

#%% Models
Dim=Data.Dim
randomGenerator=myGenerator()
Generator=myGenerator(compositLayers([Dim*10,Dim*10,Dim*10,Dim*10,Dim*20,Dim*20,Dim*20,Dim*50,Dim*200,Dim],0))
Critic=myCritic(compositLayers([Dim,Dim*2,Dim*2,Dim*5,Dim*2,Dim*2,Dim],0))
Model1=WGAN(hyperParams={'p_miss':0.5},optimizer=tf.keras.optimizers.Adam())

# %%
counter=0
train,test=Data.getPipeLine(p_miss=0.5,p_hints=0.5,train_rate=1,batch_ratio=1,repeat=1)
# test=iter(test)
backup=tf.identity_n(Critic.body.trainable_variables)
for dat_train,[mask,hint_mask,hints] in tqdm(train):
    generated_data=Generator.generate(dat_train,mask)
    adjusted_generated_data=mask*dat_train+generated_data*(1-mask)
    
    
    #Critic
    tau=tf.random.uniform([tf.shape(dat_train)[0],1], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=None, name=None)
    interpolated_data=dat_train+tau*(adjusted_generated_data-dat_train)
    with tf.GradientTape(persistent=True) as tape:
        mean_critics_diff=Critic.calc_critic_diff(dat_train,adjusted_generated_data,hints)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(interpolated_data)           
            interpolated_critics=Critic.criticise(interpolated_data,hints)
        interpolated_critic_gradients = tape.gradient(interpolated_critics,interpolated_data)
        normed_gradients = tf.sqrt(tf.reduce_sum(tf.square(interpolated_critic_gradients), axis=1))
        gradient_penalty = tf.reduce_mean(tf.square(normed_gradients-1.))
        
        critic_loss=mean_critics_diff+Model1.alpha*gradient_penalty
    critic_loss_gradient = tape.gradient(critic_loss,Critic.body.trainable_variables)
    Model1.optimizer.apply_gradients(zip(critic_loss_gradient, Critic.body.trainable_variables))

    #Generator
    with tf.GradientTape(persistent=True) as tape:
        generated_data=Generator.generate(dat_train,mask)
        adjusted_generated_data=generated_data*(1-mask)+mask*dat_train
        generated_data_critics=Critic.criticise(adjusted_generated_data,hints)
        genenerated_critic_loss=-tf.reduce_mean(generated_data_critics)      
    loss_gradients = tape.gradient(critic_loss,Generator.body.trainable_variables)
Critic.body.trainable_variables

# %%
#%% Run - 1 : 
generated_data=Generator.generate(dat_train,mask)
adjusted_generated_data=mask*dat_train+generated_data*(1-mask)

#Critic
tau=tf.random.uniform([tf.shape(dat_train)[0],1], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=None, name=None)
interpolated_data=dat_train+tau*(adjusted_generated_data-dat_train)
with tf.GradientTape(persistent=True) as tape:
    mean_critics_diff=Critic.calc_critic_diff(dat_train,adjusted_generated_data,hints)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(interpolated_data)           
        interpolated_critics=Critic.criticise(interpolated_data,hints)
    interpolated_critic_gradients = tape.gradient(interpolated_critics,interpolated_data)
    normed_gradients = tf.sqrt(tf.reduce_sum(tf.square(interpolated_critic_gradients), axis=1))
    gradient_penalty = tf.reduce_mean(tf.square(normed_gradients-1.))
    


# %%
for i in range(0,len(backup)):
    Critic.body.trainable_variables[i].assign(backup[i])
    

# %%
