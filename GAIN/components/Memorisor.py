#%%
import tensorflow as tf

class Memorise(tf.Module):
    def __init__(self,model,memory_size,stack_size,arg_sizes,name=None):
        super(Memorise, self).__init__(name=name)
        memorySize=tf.TensorShape([memory_size,stack_size,tf.reduce_sum(arg_sizes)])
        self.memory = tf.Variable(tf.zeros(memorySize),trainable=False)
        self.model=model
        self.arg_sizes=arg_sizes
        self.memory_size=memory_size
        self.pointer=tf.Variable(0,trainable=False,dtype=tf.int32)

    def __call__(self, *args):
        self._update(self.pointer,tf.concat(args,axis=1))
        self.pointer.assign(tf.math.floormod(self.pointer+1,self.memory_size))
        return self.model(*args)
    
    def _update(self,entry,data):
        self.indice=tf.reshape(self.pointer,[1,1])
        self.newEntry=tf.expand_dims(data,0)
        self.memory.assign(tf.tensor_scatter_nd_update(self.memory,self.indice,self.newEntry))
        
        
    def replay(self):
        inputs=tf.split(self.memory,axis=2,num_or_size_splits=self.arg_sizes)
        return self.model(*[tf.reshape(x,[-1,self.arg_sizes[idx]]) for idx,x in enumerate(inputs)])
    
    def recall(self):
        inputs=tf.split(self.memory,axis=2,num_or_size_splits=self.arg_sizes)
        return [tf.reshape(x,[-1,self.arg_sizes[idx]]) for idx,x in enumerate(inputs)]



# %%
