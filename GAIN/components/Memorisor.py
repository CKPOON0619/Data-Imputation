#%%
from tensorflow import concat, split, identity_n, squeeze, TensorShape,Variable,float32
from tensorflow.keras import Model

class MemoiseIn(Model):
    def __init__(self,func,size):
        super(MemoiseIn, self).__init__()
        self.size=size
        self.func=func
        self.pointer=0
    def initiateMemory(self,*inputs):
        self.memory=[identity_n(inputs) for j in range(0,self.size)]
    def __pointNext(self):
        self.pointer=(self.pointer+1)%self.size
    
    def call(self,*args):
        self.memory[self.pointer]=identity_n(args)
        self.__pointNext()
        inputs=split(self.memory,axis=1,num_or_size_splits=len(args))
        return self.func(*[squeeze(x) for x in inputs])
#%%
class MemoiseOut(Model):
    def __init__(self,model,size):
        super(MemoiseOut, self).__init__()
        self.size=size
        self.model=model
        self.pointer=0
    
    def initiateMemory(self,*inputs):
        outputs=self.model(*inputs)
        self.memory=[identity_n(outputs) for i in range(self.size)]
    
    def __pointNext(self):
        self.pointer=(self.pointer+1)%self.size
    
    def call(self,*args):
        output=self.model(*args)
        self.memory[self.pointer]=identity_n(output)
        self.__pointNext()
        return output
