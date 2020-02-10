import tensorflow as tf
from tensorflow.keras import Model

class Encoder(Model):
    """
    A Encoder class for the VAE model.

    Args:
        Dim: Dimension of data point.
        body: A kera Model that return a matrix of the same shape as data input. 
    """
    def __init__(self,body):
        super(Encoder,self).__init__()
        self.body = body

    def call(self,x,mask):
        """
        encoder model call for VAE which returns a mean and log variance pair of the inferred distribution.

        Args: 
            x: Data input scaled to have range [0,1].
            mask: mask for data. 1 = reveal, 0 = hidden

        Returns:
            a mean and log variance pair evaluated by the encoder body.
        """
        masked_x=mask*x
        mean, logvar = tf.split(self.body(tf.concat(axis = 1, values = [masked_x,mask])), num_or_size_splits=2, axis=1)
        return mean, logvar

