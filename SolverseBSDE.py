import tensorflow as tf
from ErgodicFactorModel import ErgodicFactorModel
import time
from tensorflow.keras import optimizers

class SolverBase:
    # mathModel          Math model
    # modelKeras         Keras model
    # lRate              Learning rate
    def __init__(self, ErgodicFactorModel, lRate):
        self.ErgodicFactorModel = ErgodicFactorModel
        self.StochasticFactor = ErgodicFactorModel.stochastic_factor
        self.lRate = lRate

    @tf.function
    def proj(self, Z, Zlim):
        norms = tf.norm(Z, axis=1, keepdims=True)
        # Calculate scaling factors for rows where the norm exceeds Zlim
        scaling_factors = tf.where(norms > Zlim, Zlim / norms, 1)
        Z_proj = Z * scaling_factors

        return Z_proj

# Global solver
class SolverGlobaleBSDE(SolverBase):
    def __init__(self, ErgodicFactorModel, modelKerasUZ , lRate):
        super().__init__(ErgodicFactorModel, lRate)
        self.modelKerasUZ = modelKerasUZ

    def train(self, batchSize, batchSizeVal, num_epoch, num_epochExt):
        @tf.function
        def optimizeBSDE(nbSimul):
            T_A, dW, V = self.StochasticFactor.sample(self.StochasticFactor.mu, self.StochasticFactor.kappa, nbSimul)
            dW, V = tf.convert_to_tensor(dW, dtype=tf.float32), tf.convert_to_tensor(V, dtype=tf.float32)
            Y = []
            maxind = max(T_A)
            Y_trans = self.ErgodicFactorModel.Y0 * tf.ones([nbSimul], dtype=tf.float32)
            for iStep in range(int(maxind)):
              input_tensor = tf.expand_dims(V[iStep, :], axis=-1)
              Z = self.modelKerasUZ(input_tensor)
              Y_trans = Y_trans - self.StochasticFactor.dt * self.ErgodicFactorModel.f(V[iStep, :], Z) + self.modelKerasUZ.lambd * tf.ones([nbSimul], dtype=tf.float32) * self.StochasticFactor.dt + tf.reduce_sum(Z * dW[iStep, :], axis=1)
              indices = tf.where(T_A == iStep)[:, 0]
              Y_trans_selected = tf.gather(Y_trans, indices)
              Y.append(Y_trans_selected)

            Y = tf.concat(Y, axis=0)

            return tf.reduce_mean(tf.square(Y - self.ErgodicFactorModel.Y0))

        # train to optimize control
        @tf.function
        def trainOptNN(nbSimul, optimizer):
            with tf.GradientTape() as tape:
                objFunc_Z = optimizeBSDE(nbSimul)
            gradients = tape.gradient(objFunc_Z, self.modelKerasUZ.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.modelKerasUZ.trainable_variables))
            return objFunc_Z

        optimizerN = optimizers.Adam(learning_rate = self.lRate)

        self.listlambd = []
        self.lossList = []
        self.duration = 0
        for iout in range(num_epochExt):
            start_time = time.time()
            for epoch in range(num_epoch):
                # un pas de gradient stochastique
                trainOptNN(batchSize, optimizerN)
            end_time = time.time()
            rtime = end_time-start_time
            self.duration += rtime
            objError_Yterm = optimizeBSDE(batchSizeVal)
            lambd = self.modelKerasUZ.lambd.numpy()
            print(" Error",objError_Yterm.numpy(),  " elapsed time %5.3f s" % self.duration, "lambda so far ",lambd, 'epoch', iout)
            self.listlambd.append(lambd)
            self.lossList.append(objError_Yterm)

        return self.listlambd, self.lossList

# Local solver
class SolverLocaleBSDE(SolverBase):
    def __init__(self, ErgodicFactorModel, modelKerasY, modelKerasZ, lRate):
        super().__init__(ErgodicFactorModel, lRate)
        self.modelKerasY = modelKerasY
        self.modelKerasZ = modelKerasZ

    def train(self, batchSize, batchSizeVal, num_epoch, num_epochExt):
        @tf.function
        def optimizeBSDE(nbSimul):
            T_A, dW, V = self.StochasticFactor.sample(self.StochasticFactor.mu, self.StochasticFactor.kappa, nbSimul)
            dW, V = tf.convert_to_tensor(dW, dtype=tf.float32), tf.convert_to_tensor(V, dtype=tf.float32)
            maxind = max(T_A)
            Loss = []
            loss_term = tf.zeros([nbSimul], dtype=tf.float32)
            Y_trans = self.ErgodicFactorModel.Y0 * tf.ones([nbSimul], dtype=tf.float32)
            for iStep in range(int(maxind)):
              input_tensor = tf.expand_dims(V[iStep, :], axis=-1)
              Z = self.modelKerasZ(input_tensor)
              Y = self.modelKerasY(input_tensor)
              toAdd = self.StochasticFactor.dt * self.ErgodicFactorModel.f(V[iStep, :], Z) - self.modelKerasZ.lambd * tf.ones([nbSimul], dtype=tf.float32) * self.StochasticFactor.dt - tf.reduce_sum(Z * dW[iStep, :], axis=1)
              loss_term = loss_term + toAdd

              indices = tf.where(T_A >= iStep)[:, 0]
              loss_term_selec = tf.expand_dims(tf.gather(loss_term, indices), axis=-1)
              Y_selec = tf.gather(Y, indices)

              Loss.append(Y_selec  + loss_term_selec - self.ErgodicFactorModel.Y0*tf.ones_like(loss_term_selec))

            Loss = tf.concat(Loss, axis=0)

            return tf.reduce_mean(tf.square(Loss))

        # train to optimize control
        @tf.function
        def trainOptNN(nbSimul, optimizerZ, optimizerY):
            with tf.GradientTape(persistent=True) as tape:
                objFunc = optimizeBSDE(nbSimul)
            gradientsZ = tape.gradient(objFunc, self.modelKerasZ.trainable_variables)
            gradientsY = tape.gradient(objFunc, self.modelKerasY.trainable_variables)

            optimizerZ.apply_gradients(zip(gradientsZ, self.modelKerasZ.trainable_variables))
            optimizerY.apply_gradients(zip(gradientsY, self.modelKerasY.trainable_variables))

            del tape
            return objFunc

        optimizerZ = optimizers.Adam(learning_rate=self.lRate)
        optimizerY = optimizers.Adam(learning_rate=self.lRate)

        self.listlambd = []
        self.lossList = []
        self.duration = 0
        for iout in range(num_epochExt):
            start_time = time.time()
            for epoch in range(num_epoch):
                # un pas de gradient stochastique
                trainOptNN(batchSize, optimizerZ, optimizerY)
            end_time = time.time()
            rtime = end_time-start_time
            self.duration += rtime
            objError_Yterm = optimizeBSDE(batchSizeVal)
            lambd = self.modelKerasZ.lambd.numpy()
            print(" Error",objError_Yterm.numpy(),  " elapsed time %5.3f s" % self.duration, "lambda so far ",lambd, 'epoch', iout)
            self.listlambd.append(lambd)
            self.lossList.append(objError_Yterm)


        return self.listlambd, self.lossList