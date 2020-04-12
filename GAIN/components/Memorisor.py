#%%
import tensorflow as tf

#%% 
# Memoriser as wrapper using cusom pointer
class Memorise(tf.Module):
    '''
    A tf.Module subclass that works as a function wrapper for a tf.Model instances. 
    It create new instance that wrap around a given function with a memory unit. 
    Each time it is called, it will update its memory with the new input and execute the function with the memory.
    '''
    def __init__(self,model,memory_size,batch_size,arg_sizes,initial_memory=None,name=None):
        '''
        Creates an instance of the function wrapped with memory.
            Args:
                model: A model or function to be wrapped with Memory. It should be fixed on the stack and argument size.
                memory_size: The number of inputs to be remembered before the oldest memory be covered.
                batch_size: batch size of data input. i.e. The length of the first dimension of the input.
                arg_sizes: The arguments size of the input. i.e. The length of the second dimension of the input.
                initial_memory(Optional): Initial memory, needed if input_shape is not provided. 
                name(Optional): The name of the tf.Module created. 
            
            Return:
                The function wrapped with memories. 
                Each time it is called, it will update its memory with the new input and execute the function with the memory.
                 
        '''
        super(Memorise, self).__init__(name=name)
        if initial_memory:
            self.memory=tf.Variable(initial_memory,trainable=False)
        else:
            memorySize=tf.TensorShape([memory_size,batch_size,tf.reduce_sum(arg_sizes)])
            self.memory = tf.Variable(tf.zeros(memorySize),trainable=False)
        self.model=model
        self.arg_sizes=arg_sizes
        self.memory_size=memory_size
        self.pointer=tf.Variable(0,trainable=False,dtype=tf.int32)
        
    def _update_memory(self,data):
        '''
        This updates the memory entry at the pointer and moves the pointer accordingly. 
        The pointer will keep iterating through the memory in cycles.
            Args: 
                data: Data to be added to the memory.
        '''
        self.indice=tf.reshape(self.pointer,[1,1])
        self.newEntry=tf.expand_dims(data,0)
        self.memory.assign(tf.tensor_scatter_nd_update(self.memory,self.indice,self.newEntry))
        self.pointer.assign(tf.math.floormod(self.pointer+1,self.memory_size))
        
    def _process_memory(self):
        '''
        This processes the memory with the model provided during initiation.
        '''
        inputs=tf.split(self.memory,axis=2,num_or_size_splits=self.arg_sizes)
        return self.model(*[tf.reshape(x,[-1,self.arg_sizes[idx]]) for idx,x in enumerate(inputs)])
    
    def __call__(self, *args):
        '''
        When being called, this instance will update the memory and process the memory with the model provided.
        '''
        self._update_memory(tf.concat(args,axis=1))
        return self._process_memory()
    
    def recall_memory(self,entry=None):
        '''
        This is a method to retrieve the memory in a flattened format.
            Args:
                entry(optional): The argument entry to be returned. If it is `None`, it returns all arguments as an array.
            
            Return:
                Corresponding argument(s) of the model.
        '''
        inputs=tf.split(self.memory,axis=2,num_or_size_splits=self.arg_sizes)
        args=[tf.reshape(x,[-1,self.arg_sizes[idx]]) for idx,x in enumerate(inputs)]
        if entry is not None:
            return args[entry]
        return args