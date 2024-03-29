import tensorflow as tf
from tensorflow.keras import layers

#Neural Network
class Net( tf.keras.Model):
    def __init__(self, cond_lambd, lambda_lim, dim, nbNeurons, activation= tf.nn.tanh):
        super().__init__()
        self.nbNeurons = nbNeurons
        self.dim = dim
        self.ListOfDense = [layers.Dense( nbNeurons[i],activation= activation, kernel_initializer= tf.keras.initializers.GlorotNormal())  for i in range(len(nbNeurons)) ] +[layers.Dense(self.dim, activation= None, kernel_initializer= tf.keras.initializers.GlorotNormal())]
        if cond_lambd:
          self.lambd= tf.Variable(tf.keras.initializers.GlorotNormal()([1]),  trainable = True, dtype=tf.float32, constraint=lambda t: tf.clip_by_value(t, -lambda_lim, lambda_lim))

    def call(self,inputs):
        x = inputs
        for layer in self.ListOfDense:
            x = layer(x)
        return x