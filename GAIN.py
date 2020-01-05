#%% Packages
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from datetime import datetime

#%% Tensorboard logging
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = './tf_logs' + current_time + "/"

#%% System Parameters 

# Training Data 
file='measureGenerator'

# 1. Missing rate
p_miss = 0.5
# 3. Hint rate
p_hint = 0.5
# 4. Loss Hyperparameters
alpha = 1
# 5. Train Rate
train_rate = 0.8
# 6. iteration
iteration=4000
# 7. Dropout Rate
drop_rate=0

#%% Data

# Data generation
file='measureGenerator'
rawData = np.genfromtxt("../test inputs/{}.csv".format(file), delimiter=",",skip_header=1)

# Parameters
sample_size = len(rawData)
Dim = len(rawData[0,:])
train_size=int(sample_size*train_rate)