import tensorflow as tf

import os
import time
import numpy as np
import glob


#%% Helpers
def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)


#%% Body
class VAE(tf.keras.Model):
    def __init__(self, latent_dim,logdir= getcwd()+'\\logs\\tf_logs' + datetime.now().strftime("%Y%m%d-%H%M%S"), optimizer = tf.keras.optimizers.Adam(1e-4)):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential([
            tf.keras.layers.Dense(latent_dim*8, activation=tf.nn.relu, kernel_initializer='glorot_normal'),
            tf.keras.layers.Dense(latent_dim*6, activation=tf.nn.relu, kernel_initializer='glorot_normal'),
            tf.keras.layers.Dense(latent_dim*4, activation=tf.nn.relu, kernel_initializer='glorot_normal'),
            # No activation
            tf.keras.layers.Dense(latent_dim*2, activation=tf.nn.relu, kernel_initializer='glorot_normal'),
        ])

        self.generative_net = tf.keras.Sequential(
            [
            tf.keras.layers.Dense(latent_dim*2, activation=tf.nn.relu, kernel_initializer='glorot_normal'),
            tf.keras.layers.Dense(latent_dim*4, activation=tf.nn.relu, kernel_initializer='glorot_normal'),
            tf.keras.layers.Dense(latent_dim*6, activation=tf.nn.relu, kernel_initializer='glorot_normal'),
            tf.keras.layers.Dense(latent_dim*8, activation=tf.nn.relu, kernel_initializer='glorot_normal'),
            ]
        )

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def calcLoss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = model.decode(z)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)
    
    @tf.function
    def trainWithBatch(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss = self.calc_loss(self, x)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

#%% Run

epochs = 100
latent_dim = 50
num_examples_to_generate = 16

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim)

for epoch in range(1, epochs + 1):
  start_time = time.time()
  for train_x in train_dataset:
    compute_apply_gradients(model, train_x, optimizer)
  end_time = time.time()

  if epoch % 1 == 0:
    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
      loss(compute_loss(model, test_x))
    elbo = -loss.result()
    display.clear_output(wait=False)
    print('Epoch: {}, Test set ELBO: {}, '
          'time elapse for current epoch {}'.format(epoch,
                                                    elbo,
                                                    end_time - start_time))