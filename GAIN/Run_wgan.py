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
Generator=myGenerator(compositLayers([Dim*10,Dim*10,Dim*10,Dim*10,Dim*20,Dim*20,Dim*20,Dim*50,Dim*200,Dim],0))
Critic=myCritic(compositLayers([Dim,Dim*2,Dim*2,Dim*5,Dim*2,Dim*2,Dim],0,layer_activation=tf.nn.sigmoid))
Model1=WGAN(Generator,Critic,hyperParams={'p_miss':0.5},optimizer=tf.keras.optimizers.Adam())

# %%
counter=0
train,test=Data.getPipeLine(p_miss=0.5,p_hints=0.5,train_rate=0.8,batch_ratio=1,repeat=200)
test=iter(test)
for dat_train,[mask,hint_mask,hints] in tqdm(train):
    Model1.train_critic_with_random(dat_train,mask,hints)
    # track=tf.reduce_sum([tf.reduce_sum(tf.math.is_nan(i)) for i in Critic.body.trainable_variables])
    if counter%5:
        Model1.tensorboard_log_with_random('Step 1.',dat_train,mask,hints,hint_mask)
    counter+=1

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
x=tf.constant(1,dtype=tf.float32)
z=tf.Variable(3,dtype=tf.float32)



# %%

with tf.GradientTape(persistent=True) as tape:
    with tf.GradientTape(watch_accessed_variables=False) as tape2:
        tape2.watch(x)  
        y=x*z
    dydx=tape2.gradient(y,x)
    l=dydx*dydx
g=tape.gradient(l,z)

tf.keras.optimizers.Adam().apply_gradients(zip([g],[z]))
    

# %%
