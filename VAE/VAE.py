#%% Packages
import tensorflow as tf
from os import getcwd, makedirs
from datetime import datetime
import time
import numpy as np
import glob

#%% Helpers
def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

def reparameterize(mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

# Create Mask
def createMask(data,maskRatio):
    '''
    Args:
        data: tensor to be masked
        maskRatio: proportion of entries to be marked as 1
    Returns: 
        0,1 matrix of the same shape as data
    '''
    return tf.dtypes.cast((tf.random.uniform(tf.shape(data),minval=0,maxval=1)>(1-maskRatio)),dtype=tf.float32)


#%% Body
class VAE(tf.keras.Model):
    def __init__(self, logdir= getcwd()+'\\logs\\tf_logs' + datetime.now().strftime("%Y%m%d-%H%M%S"), hyperParams={}, optimizer=tf.keras.optimizers.Adam()):
        super(VAE, self).__init__()
        self.iter=0
        # self.__dict__.update(defaultParams)
        self.__dict__.update(hyperParams)
        self.optimizer = optimizer
        self.reset(logdir)
        
    def reset(self,logdir= getcwd()+'\\logs\\tf_logs' + datetime.now().strftime("%Y%m%d-%H%M%S")):
        '''
        A function to reset logging directory and training epoch.
        Args: 
            logdir: logging directory for tensorboard
        '''
        self.logdir=logdir
        self.epoch = tf.Variable(0,dtype=tf.int64)
        makedirs(logdir, exist_ok=True)
        self.summary_writer = tf.summary.create_file_writer(logdir)
        print('tensorboard --logdir {}'.format(logdir)+' --host localhost')

    def calcLoss(self, encoder, decoder, x):
        mask = createMask(x, 0.2)
        [mean, logvar] = encoder(x, mask)
        z = reparameterize(mean, logvar)
        x_logit = decoder(z)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1])
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)
    
    # @tf.function
    def trainWithBatch(self, x, encoder, decoder):
        with tf.GradientTape() as tape:
            loss = self.calcLoss(encoder, decoder, x)
        gradients = tape.gradient(loss, encoder.trainable_variables+decoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, encoder.trainable_variables+decoder.trainable_variables))

#%% Run
# epochs = 100
# latent_dim = 50
# num_examples_to_generate = 16

# # keeping the random vector constant for generation (prediction) so
# # it will be easier to see the improvement.
# random_vector_for_generation = tf.random.normal(
#     shape=[num_examples_to_generate, latent_dim])
# model = CVAE(latent_dim)

# for epoch in range(1, epochs + 1):
#   start_time = time.time()
#   for train_x in train_dataset:
#     compute_apply_gradients(model, train_x, optimizer)
#   end_time = time.time()

#   if epoch % 1 == 0:
#     loss = tf.keras.metrics.Mean()
#     for test_x in test_dataset:
#       loss(compute_loss(model, test_x))
#     elbo = -loss.result()
#     display.clear_output(wait=False)
#     print('Epoch: {}, Test set ELBO: {}, '
#           'time elapse for current epoch {}'.format(epoch,
#                                                     elbo,
#                                                     end_time - start_time))

# %%
