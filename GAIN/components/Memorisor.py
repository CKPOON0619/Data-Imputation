#%%
from tensorflow import concat, split, identity_n, squeeze, TensorShape,Variable,float32

class MemoiseIn():
    def __init__(self,func,size):
        self.size=size
        self.func=func
        self.pointer=0
    def initiateMemory(self,*inputs):
        self.memory=[identity_n(inputs) for j in range(0,self.size)]
    def __pointNext(self):
        self.pointer=(self.pointer+1)%self.size
    
    def __call__(self,*args):
        self.memory[self.pointer]=identity_n(args)
        self.__pointNext()
        inputs=split(self.memory,axis=1,num_or_size_splits=len(args))
        return self.func(*[squeeze(x) for x in inputs])
#%%
class MemoiseOut():
    def __init__(self,func,size):
        self.size=size
        self.func=func
        self.pointer=0
    
    def initiateMemory(self,*inputs):
        outputs=self.func(*inputs)
        self.memory=[identity_n(outputs) for i in range(self.size)]
    
    def __pointNext(self):
        self.pointer=(self.pointer+1)%self.size
    
    def __call__(self,*args):
        output=self.func(*args)
        self.memory[self.pointer]=identity_n(output)
        self.__pointNext()
        return output

# %%
