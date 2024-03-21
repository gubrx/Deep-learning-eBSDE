import tensorflow as tf
import numpy as np
from scipy.integrate import quad
import scipy.optimize as opt

class ErgodicFactorModel():
  def __init__(self, stochastic_factor):
    self.stochastic_factor = stochastic_factor
    self.dt = stochastic_factor.dt
    self.T_H = stochastic_factor.T_H
    self.dim = stochastic_factor.dim

  def proj_Pi(self, Pi, X):
      Pi = tf.convert_to_tensor(Pi, dtype=X.dtype)
      # Ensure Pi has shape (dim, 2)
      assert Pi.shape[1] == 2, "Pi should have shape (dim, 2)"

      Pi_expanded = tf.expand_dims(tf.expand_dims(Pi, 0), 0)

      X_expanded = tf.expand_dims(X, -1)
      lower_mask = X_expanded < Pi_expanded[..., 0:1]
      upper_mask = X_expanded > Pi_expanded[..., 1:2]

      X_proj = tf.where(lower_mask, Pi_expanded[..., 0:1], X_expanded)
      X_proj = tf.where(upper_mask, Pi_expanded[..., 1:2], X_proj)

      X_proj = tf.squeeze(X_proj, -1)

      return X_proj

  # forward scheme eBSDE
  def forward_BSDE(self, T_A, dW, V, ergodic_model, kerasModel):
      nbr_traj = len(V[0,:])
      maxind = max(T_A)
      dW, V = tf.convert_to_tensor(dW, dtype=tf.float32), tf.convert_to_tensor(V, dtype=tf.float32)
      Y_sim = np.zeros([maxind+1, nbr_traj])
      Z_sim = np.zeros((maxind, nbr_traj, self.dim))

      Y_trans = ergodic_model.Y0 * tf.ones([nbr_traj], dtype=tf.float32)
      Y_sim[0, :] = Y_trans.numpy()
      for iStep in range(maxind):
          Z = kerasModel(tf.expand_dims(V[iStep, :nbr_traj], axis=-1))
          Y_trans = Y_trans - self.dt * tf.transpose(ergodic_model.f(V[iStep, :nbr_traj], Z)) + kerasModel.lambd.numpy() * tf.ones([nbr_traj], dtype=tf.float32) * self.dt + tf.reduce_sum(Z * dW[iStep, :], axis=1)
          Y_sim[iStep+1, :] = Y_trans.numpy()
          Z_sim[iStep, :] = Z.numpy()

      return Y_sim, Z_sim

  def mean_abs_error(self, T_A, Y_ex, Y_sim):
      indTH = int(np.ceil(self.T_H / self.dt))
      nbr_traj = Y_ex.shape[1]
      Y_ex_mod = Y_ex[:indTH+1, :nbr_traj]
      mask_rel = tf.cast(tf.math.abs(Y_ex_mod) > 0, dtype=tf.float32)
      Y_sim_mod = Y_sim[:indTH+1, :nbr_traj]
      # Absolute error
      Etot = np.abs(Y_ex_mod - Y_sim_mod)
      # Relative error
      Etotrel = np.where(Y_ex_mod != 0, np.abs((Y_ex_mod - Y_sim_mod) / Y_ex_mod), 0)

      Emoy = np.sum(Etot, axis=1) / nbr_traj
      Emoy_rel = np.sum(Etotrel, axis=1) / np.sum(mask_rel, axis=1)

      return Emoy, Emoy_rel

  def integral_error(self, T_A, Y_ex, Y_sim, Z_ex, Z_sim):
      indTH = int(np.ceil(self.T_H / self.dt))
      nbr_traj = Y_ex.shape[1]
      interrY = tf.reduce_mean(self.dt * tf.reduce_sum(abs(Y_ex[:indTH, :nbr_traj] - Y_sim[:indTH, :nbr_traj]), axis=0))
      interrZ = tf.reduce_mean(self.dt * tf.reduce_sum(tf.norm(Z_ex[:indTH, :nbr_traj, :] - Z_sim[:indTH, :nbr_traj, :], axis=-1)**2, axis=0))

      return interrY, interrZ

class EV(ErgodicFactorModel):
  def __init__(self, stochastic_factor):
      super().__init__(stochastic_factor)

  # Approximation Monte Carlo lambda for generators depending only on the stochastic factor V
  def lambdapprox(self, V, T_A):
      T_A = tf.convert_to_tensor(T_A, dtype=tf.float32)
      V = tf.convert_to_tensor(V, dtype=tf.float32)
      T_Aint = tf.cast(T_A, dtype=tf.int32)

      timestep_range = tf.range(tf.shape(V)[0])[:, tf.newaxis]
      selec_index = timestep_range <= T_Aint

      V_selec = tf.where(selec_index, V, tf.zeros_like(V))
      f_values = self.f(V_selec, 0)
      appint_values = tf.reduce_sum(f_values * self.stochastic_factor.dt, axis=0)

      meanappint = tf.reduce_sum(appint_values)
      meanT_R = tf.reduce_sum(T_A * self.dt)

      return float(meanappint / meanT_R)

# The solutions for Example 1 and Example 2 are valid for a stochastic factor of type OrnsteinUhlenbeck and in dimension 1.
class Example1(EV):
  def __init__(self, ornstein_uhlenbeck, Cv):
      super().__init__(ornstein_uhlenbeck)
      self.Cv = Cv
      self.muval = ornstein_uhlenbeck.muval
      self.kappa = ornstein_uhlenbeck.kappa
      self.Y0 = (self.Cv/(self.muval + (1/2)*self.kappa[0]**2))*np.sqrt(2*np.pi)*scipy.stats.norm.cdf(ornstein_uhlenbeck.x0, 0, 1)
      self.Zlim = tf.norm(tf.constant(self.kappa, dtype=tf.float32))*(self.Cv / (self.muval - self.Cv))
      self.lambdlim = self.Cv*tf.exp(-1**2/2)
      self.lambd_ex = 0

  # Driver
  def f(self, v, z):
      return v*self.Cv*tf.exp(-v**2/2)

  # Exact solution
  def y_ex(self, V):
      return (self.Cv/(self.muval + (1/2)*self.kappa[0]**2))*np.sqrt(2*np.pi)*scipy.stats.norm.cdf(V, 0, 1)

  def z_ex(self, V):
      return (self.Cv/(self.muval + (1/2)*self.kappa[0]**2))*tf.exp(-V**2/2)


class Example2(EV):
  def __init__(self, ornstein_uhlenbeck, Cv):
      super().__init__(ornstein_uhlenbeck)
      self.Cv = Cv
      self.dt = ornstein_uhlenbeck.dt
      self.muval = ornstein_uhlenbeck.muval
      self.kappa = np.sqrt(2*self.muval) #kappa fixed from drift, see Annex A
      self.x0 = ornstein_uhlenbeck.x0

      self.A1 = self.Cv/(self.kappa**2)
      self.A2 = 2*self.A1
      self.Y0 = 1+quad(lambda y: np.exp(y**2/2)*(self.A1*np.exp(-y**2) + self.A2*(scipy.stats.norm.cdf(y, 0, 1) - 1)), 0, self.x0)[0]

      self.Zlim = tf.norm(self.kappa)*(self.Cv / (self.muval - self.Cv))
      self.lambd_ex = self.Cv / np.sqrt(2*np.pi)
      self.lambdlim = self.Cv*tf.exp(-1**2/2)

  def integrandpos(self, y):
      return np.exp(y**2/2)*(self.A1*np.exp(-y**2) + self.A2*(scipy.stats.norm.cdf(y, 0, 1) - 1))

  def integrandneg(self, y):
      return np.exp(y**2/2)*(-self.A1*np.exp(-y**2) + self.A2*scipy.stats.norm.cdf(y, 0, 1))

  # Exact solution
  def y_ex(self, T_A, X):
      nbr_time_step, nbr_traj = X.shape
      Y_ex = np.zeros((nbr_time_step, nbr_traj))
      for iStep in range(nbr_time_step):
          listM = tf.cast(iStep <= T_A, tf.float32)
          Neg = tf.cast(X[iStep, :] < 0, tf.float32)
          Pos = tf.cast(X[iStep, :] >= 0, tf.float32)
          Xpos = X[iStep, :] * Pos * listM
          Xneg = X[iStep, :] * Neg * listM
          for m in range(nbr_traj):
              if listM[m]:  # Only integrate for active trajectories
                  if Pos[m]:
                      Y_ex[iStep, m] = 1+ quad(self.integrandpos, 0, Xpos[m])[0]
                  if Neg[m]:
                      Y_ex[iStep, m] = 1+quad(self.integrandneg, 0, Xneg[m])[0]

      return Y_ex

  def z_ex(self, X):
    Pos = X >= 0
    Neg = X < 0
    return Neg*self.integrandneg(X) + Pos * self.integrandpos(X)

  # Driver
  def f(self, v, z):
      return self.Cv*abs(v)*tf.exp(-v**2/2)


class ErgodicPowerSE(ErgodicFactorModel):
  def __init__(self, stochastic_factor, market_price, p, b, delt):
      super().__init__(stochastic_factor)
      self.delt = delt #risk aversion
      self.gamma = 1 /(1 - self.delt)
      self.Cv = (1/2)*(delt/(1-delt))*np.linalg.norm(p)*np.max([2, b])
      self.lambdlim = (1/2)*(delt/(1-delt))*b**2
      self.dt = stochastic_factor.dt
      self.mu = stochastic_factor.mu
      self.kappa = stochastic_factor.kappa
      self.Zlim = tf.norm(tf.constant(self.kappa, dtype=tf.float32))*(self.Cv / (self.stochastic_factor.muval - self.Cv))
      self.Y0 = 0.
      self.market_price = market_price
      self.lambd_ex = 'Not known'

  def thet(self, x):
    return self.market_price(x)

  def beta(self, V, l):
    return (self.delt / (1 - self.delt)) * (self.gamma / 2) * tf.norm(self.thet(V), axis=-1)**2 - self.gamma * l

  def nu(self, V):
    return (self.delt / (1 - self.delt)) * self.thet(V)

  # Approximation Monte Carlo - Power utility example
  def loss(self, T_A, X, dW, l):
    Nbsimul = len(X[0, :])
    G = self.stochastic_factor.sol_SDE(X, dW, lambda x: self.beta(x, l), self.nu, 0, 1)
    selected_G = tf.gather_nd(G, tf.stack([T_A, tf.range(Nbsimul)], axis=1))
    lossval = tf.abs(tf.reduce_mean(selected_G) - 1)

    return lossval.numpy()

  def lambda_app_newt(self, num_sample):
    T_A, dW, V = self.stochastic_factor.sample(self.mu, self.kappa, num_sample)
    return scipy.optimize.fsolve(lambda l: self.loss(T_A, V, dW, l), 0)

  # Driver
  def f(self, v, z):
    return (1/2)*self.delt/(1 - self.delt)*tf.norm(z + self.thet(v), axis = -1)**2 + (1/2)*tf.norm(z, axis=-1)**2

  # Exact solution
  def yex(self, EC):
    return self.Y0 + np.reshape((1/self.gamma)*np.log(EC),  (len(EC[:,0]), len(EC[0, :])))

  # optimal portoflio
  def optimal_strategy(self, Pi, V, kerasModel):
    Ztot = np.zeros((len(V[:,0]), len(V[0,:]), len(Pi) ))
    thetV = np.zeros((len(V[:,0]), len(V[0,:]), len(Pi) ))
    for iStep in range(len(V[:,0])):
      Z = kerasModel(tf.expand_dims(V[iStep, :len(V[0,:])], axis=-1))
      Ztot[iStep, :] = Z.numpy()
      thetV[iStep,:] = self.market_price(V[iStep,:])

    return self.proj_Pi(Pi, (1/(1-self.delt))*(thetV + Ztot))


class ErgodicPowerGen(ErgodicFactorModel):
  def __init__(self, stochastic_factor, market_price, p, b, delt):
      super().__init__(stochastic_factor)
      self.delt = delt #risk aversion
      self.gamma = 1 /(1 - self.delt)
      self.Cv = (1/2)*(delt/(1-delt))*np.linalg.norm(p)*np.max([2, b])
      self.lambdlim = (1/2)*(delt/(1-delt))*b**2
      self.dt = stochastic_factor.dt
      self.mu = stochastic_factor.mu
      self.muval = self.stochastic_factor.muval
      self.kappa = stochastic_factor.kappa
      self.Zlim = tf.norm(tf.constant(self.kappa, dtype=tf.float32))*(self.Cv / (self.muval - self.Cv))
      self.market_price = market_price
      self.Y0 = 0
      self.lambd_ex = 'Not known'

  def thet(self, x):
    return self.market_price(x)

  # Driver
  def f(self, v, z):
    return (1/2)*self.delt/(1 - self.delt)*(z[:,0] + self.thet(v)[:,0])**2 + (1/2)*tf.norm(z, axis=-1)**2

  # Optimal portoflio
  def optimal_strategy(self, Pi, V, kerasModel):
    Ztot = np.zeros((len(V[:,0]), len(V[0,:]), len(Pi) ))
    thetV = np.zeros((len(V[:,0]), len(V[0,:]), len(Pi) ))
    for iStep in range(len(V[:,0])):
      Z = kerasModel(tf.expand_dims(V[iStep, :len(V[0,:])], axis=-1))
      Ztot[iStep, :] = Z.numpy()
      thetV[iStep,:] = self.market_price(V[iStep,:])

    return self.proj_Pi(Pi, (1/(1-self.delt))*(thetV + Ztot))





