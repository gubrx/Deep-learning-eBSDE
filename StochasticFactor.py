import numpy as np
import tensorflow as tf

class StochasticFactor():
    def __init__(self, x0, dt, T_H, dim):
        self.x0 = x0
        self.dt = dt
        self.T_H = T_H
        self.dim = dim

    def one_step(self, V, mu, kappa):
        dw_sample = np.sqrt(self.dt)*np.random.normal(size=(len(V), self.dim))
        V = V + mu(V)*self.dt + np.sum(np.array(kappa)*dw_sample, axis=1)
        return dw_sample, V

    def sample(self, mu, kappa, num_sample):
        V = np.ones((1, num_sample)) * self.x0
        dW = np.empty((0, num_sample, self.dim))
        T_A = np.zeros(num_sample) + np.inf
        indT_H = int(np.ceil(self.T_H/self.dt))

        for i in range(indT_H):
            dw_sample, V_new = self.one_step(V[-1, :], mu, kappa)
            V = np.vstack([V, V_new[None, :]]) # Stack vertically
            dW = np.vstack([dW, dw_sample[None, :]]) # Adjusted for extra dimension

        Lstart = V[-1, :] - self.x0*np.ones(num_sample)
        i = indT_H+1  # Start from the index of T_H in the time grid
        while np.any(T_A == np.inf) and i < 20/self.dt:
            dw_sample, V_new = self.one_step(V[-1, :], mu, kappa)
            V = np.vstack([V, V_new[None, :]])
            dW = np.vstack([dW, dw_sample[None, :]])

            L = V_new - self.x0
            condition = L * Lstart <= 0
            # Update T_A for indices where condition is met and T_A not changed yet
            for j in np.where((condition) & (T_A == np.inf))[0]:
                T_A[j] = i

            i += 1

        if i == 20 * indT_H:
            print("The maximum return time over num_sample is large, greater than T=20. Consider changing parameters of the stochastic factor.")

        return np.array(T_A, int), dW, V


    def sol_SDE(self, X, dW, drift, vol, tinit, gammainit):
        x_sample = np.zeros([len(X[:,0]), len(X[0,:])])
        x_sample[tinit, :] = np.ones(len(X[0,:])) * gammainit
        for i in range(tinit, len(X[:,0])-1):
          x_sample[i+1, :] = x_sample[i, :] + x_sample[i, :]*drift(X[i, :])*self.dt \
            + x_sample[i,:] * tf.reduce_sum(vol(X[i, :]) * dW[i, :], axis=1)

        return x_sample


class OrnsteinUhlenbeck(StochasticFactor):
    def __init__(self, x0, dt, T_H, dim, muval, nu, kappa):
        super().__init__(x0, dt, T_H, dim)
        self.muval = muval
        self.kappa = kappa
        self.x0 = x0
        self.nu = nu

    def mu(self, x):
      return -self.muval * (x - self.nu)



