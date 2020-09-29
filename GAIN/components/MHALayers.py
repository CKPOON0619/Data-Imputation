#%%
import tensorflow as tf
#%%
def scaled_dot_product_attention(q, k, v, mask):
    '''
    Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    The depth being the length of the attention vector.
    seq_len_q is the number of queries.
    seq_len_k is the number of keys.
    seq_len_v is the number of values for attention.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition. 
        
    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable 
            to (..., seq_len_q, seq_len_k). Defaults to None.
        
    Returns:
        output, attention_weights
    '''
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32) # =depth
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

# %%
class MultiHeadAttention(tf.keras.layers.Layer):
    '''
        Ref:https://www.tensorflow.org/tutorials/text/transformer#multi-head_attention
        parameters:
            num_heads: number of attentions
    '''
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
            
    def split_heads(self, x, batch_size):
        """
        num_heads being the number of layers and attention and depth being the length of the vector for dot product attention.
        Split the last dimension into (num_heads, depth). 
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
        
    def call(self, v, k, q, mask):
        """
        v: (batch_size, seq_len, depth_v)
        k: (batch_size, seq_len, depth_k)
        q: (batch_size, seq_len, depth_q)
        """
        batch_size = tf.shape(q)[0]
        
        # Applying linear transformation to the last dim of qkv vector, extending to dimension d_model
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        
        # Keeping sequence length while spliting d_model into (num_heads x depth) and then transpose 
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
            
        return output, attention_weights
    
# %%
class MakeSequence(tf.keras.layers.Layer):
    '''
        parameters:
     
    '''
    def __init__(self, d_model):
        super(MakeSequence, self).__init__()
        self.wl=tf.keras.layers.Dense(d_model)
        
    def call(self,inputs):
        inputStack=tf.stack(inputs,axis=2)
        return self.wl(inputStack)


#%%
def point_wise_feed_forward_network(d_model, dff):
      return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])
      
#%%
class ModuleLayer(tf.keras.layers.Layer):
    """
        Ref:https://www.tensorflow.org/tutorials/text/transformer#multi-head_attention
    """
    def __init__(self, d_model, num_heads, dff,training=True, rate=0.1):
        super(ModuleLayer, self).__init__()
        self.training=True

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        
    def set(self,params):
        self.__dict__.update(params)
        return self
        
    def call(self, x, mask_mha=None):
        # x = (batch_size, input_seq_len, d_model)
        attn_output, _ = self.mha(x, x, x, mask_mha)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=self.training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model), x must have the same dim as attn_output
        
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=self.training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        
        return out2
    
class CompositeLayers(tf.keras.layers.Layer):
    """
        Ref:https://www.tensorflow.org/tutorials/text/transformer#multi-head_attention
    """
    def __init__(self, layer_num, d_model, num_heads, dff,activation="sigmoid",training=True, rate=0.1):
        super(CompositeLayers, self).__init__()
        self.training=True
        self.inputLayer=tf.keras.layers.Dense(d_model)
        self.modules = [ModuleLayer(d_model, num_heads, dff,training, rate) for i in range(layer_num)]
        self.outputLayer=tf.keras.layers.Dense(1,activation=activation)
        
    def setModules(self,params):
        for module in self.modules:
            module.set(params)
        
    def call(self, inputs):
        x=self.inputLayer(tf.stack(inputs,axis=2))
        for module in self.modules:
            x=module(x)
        y=self.outputLayer(x)
        shape=tf.shape(y)
        return tf.reshape(y,(shape[0],-1))

