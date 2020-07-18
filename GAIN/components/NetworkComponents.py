# Create generator/discriminator bodies
import tensorflow as tf

def compositLayers(layer_sizes,dropout_rate=0.2,output_activation=tf.nn.sigmoid):
    layers = []
    for idx in range(len(layer_sizes)-2):
        layers+=[tf.keras.layers.Dense(layer_sizes[idx], kernel_initializer="glorot_normal"),tf.keras.layers.LeakyReLU()]
        if(dropout_rate>0):
            layers+=[tf.keras.layers.Dropout(dropout_rate)]
    
    layers+=[tf.keras.layers.Dense(layer_sizes[-2], kernel_initializer="glorot_normal"),tf.keras.layers.LeakyReLU()]
    layers=layers+[tf.keras.layers.Dense(layer_sizes[-1], activation=output_activation)]
    return tf.keras.Sequential(layers)