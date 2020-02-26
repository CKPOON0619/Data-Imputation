#%%
from tensorflow import concat, split, identity_n, squeeze, TensorShape,Variable,float32

class MemoryReplay():
    def __init__(self,func,size):
        self.size=size
        self.func=func
        self.pointer=0
    def initiateMemory(self,*inputs):
        self.memory=[[identity_n(input) for i in range(0,len(inputs))] for j in range(0,self.size)]
    def __pointNext(self):
        self.pointer=(self.pointer+1)%self.size
    
    def __call__(self,*args):
        self.memory[self.pointer]=identity_n(args)
        self.__pointNext()
        inputs=split(self.memory,axis=1,num_or_size_splits=len(args))
        return self.func(*[squeeze(x) for x in inputs])
#%%
class Memory_Output():
    def __init__(self,func,size):
        self.size=size
        self.func=func
        self.pointer=0
    
    def initiateMemory(self,*inputs):
        outputs=self.func(*inputs)
        self.memory_out=[identity_n(outputs) for i in range(self.size)]
        return outputs
    
    def __pointNext(self):
        self.pointer=(self.pointer+1)%self.size
    
    def __call__(self,*args):
        self.memory[self.pointer]=identity_n(args)
        self.__pointNext()
        inputs=split(self.memory,axis=1,num_or_size_splits=len(args))
        output=self.func(*[squeeze(x) for x in inputs])
        self.memory_out[self.pointer]=identity_n(output)
        return output

# %%
