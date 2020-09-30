#%%
# Create generator/discriminator bodies
import tensorflow as tf

def compositLayers(layer_sizes,dropout_rate=0.2,batch_normalised=True,output_activation=tf.nn.sigmoid):
    layers = []
    for idx in range(len(layer_sizes)-2):
        layers+=[tf.keras.layers.Dense(layer_sizes[idx], kernel_initializer="glorot_normal"),tf.keras.layers.LeakyReLU()]
        if(dropout_rate>0):
            layers+=[tf.keras.layers.Dropout(dropout_rate)]
    
    layers+=[tf.keras.layers.Dense(layer_sizes[-2], kernel_initializer="glorot_normal"),tf.keras.layers.LeakyReLU()]
    layers=layers+[tf.keras.layers.Dense(layer_sizes[-1], activation=output_activation)]
    return tf.keras.Sequential(layers)

class CompositLayers(tf.keras.layers.Layer):
    def __init__(self, layer_sizes,dropout_rate=0.2,output_activation=tf.nn.sigmoid):
        super(CompositLayers, self).__init__()
        self.compositLayers= compositLayers(layer_sizes,dropout_rate=dropout_rate,output_activation=output_activation)
        
    def call(self,inputs):
        return self.compositLayers(tf.concat(axis = 1, values = inputs))


