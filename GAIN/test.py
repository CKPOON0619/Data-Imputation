#%%
import numpy as np
import tensorflow as tf

#%%
x=[[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]
m=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]

xs=tf.data.Dataset.from_tensor_slices(x).shuffle(buffer_size=3).repeat(1)
ms=tf.data.Dataset.from_tensor_slices(m).shuffle(buffer_size=3).repeat(1)
xms=tf.data.Dataset.zip((xs,ms))


# %%
for i in xms:
    print(i)
# %%
