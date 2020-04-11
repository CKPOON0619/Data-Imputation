#%%
import tensorflow as tf

#%% 
# Memoriser as wrapper using cusom pointer
class Memorise(tf.Module):
    def __init__(self,model,memory_size,stack_size,arg_sizes,initial=None,name=None):
        super(Memorise, self).__init__(name=name)
        if initial:
            self.memory=tf.Variable(initial,trainable=False)
        else:
            memorySize=tf.TensorShape([memory_size,stack_size,tf.reduce_sum(arg_sizes)])
            self.memory = tf.Variable(tf.zeros(memorySize),trainable=False)
        self.model=model
        self.arg_sizes=arg_sizes
        self.memory_size=memory_size
        self.pointer=tf.Variable(0,trainable=False,dtype=tf.int32)
    def _update(self,entry,data):
        self.indice=tf.reshape(self.pointer,[1,1])
        self.newEntry=tf.expand_dims(data,0)
        self.memory.assign(tf.tensor_scatter_nd_update(self.memory,self.indice,self.newEntry))
    def __call__(self, *args):
        self._update(self.pointer,tf.concat(args,axis=1))
        self.pointer.assign(tf.math.floormod(self.pointer+1,self.memory_size))
        inputs=tf.split(self.memory,axis=2,num_or_size_splits=self.arg_sizes)
        return self.model(*[tf.reshape(x,[-1,self.arg_sizes[idx]]) for idx,x in enumerate(inputs)]) 
    def recall(self,entry):
        inputs=tf.split(self.memory,axis=2,num_or_size_splits=self.arg_sizes)
        args=[tf.reshape(x,[-1,self.arg_sizes[idx]]) for idx,x in enumerate(inputs)]
        if entry!=None:
            return args[0]
        return args
    
        

