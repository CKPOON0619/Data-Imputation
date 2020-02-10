import tensorflow as tf
from tensorflow.keras import Model

class Decoder(Model):
    """
    A Decoder class for the VAE model.

    Args:
        Dim: Dimension of data point.
        body: A kera Model that return a matrix of the same shape as data input. 
    """
    def __init__(self, body):
        super(Decoder, self).__init__()
        self.body = body

    def call(self, z, apply_sigmoid=False):
        """
        decoder model call for VAE.

        Args: 
            x: Data input scaled to have range [0,1].
            mask: mask for data. 1 = reveal, 0 = hidden

        Returns:
            Samples from encoded distribution.
        """
        logits = self.body(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits