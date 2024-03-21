import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from Networks import Net
from StochasticFactor import StochasticFactor
from ErgodicFactorModel import ErgodicFactorModel
from SolverseBSDE import SolverGlobaleBSDE, SolverLocaleBSDE

# Parameters
T_H = 1
h = 0.01
kappa = [0.8]
dim = len(kappa)
nu = 0
x0 = 0
Mlambd = 100000

# Power utility examples parameters
delt = 0.5  # risk aversion
p = [0.8]  # Lipschitz constant market price of risk - of dimension dim
b = 3  # born market price of risk
Pi = [[-np.inf, np.inf], [0, 0]]

def market_price(v):
    v = tf.convert_to_tensor(v, dtype=tf.float32)
    pa = tf.convert_to_tensor(p, dtype=tf.float32)
    v = tf.reshape(v, (-1, 1))
    pv = pa * v

    norm_pv = tf.norm(pv, axis=1, keepdims=True)
    condition = norm_pv > b
    pv = tf.where(condition, (pv / norm_pv) * b, pv)

    return pv

# NN Parameters
nbNeuron = 20 + dim
nbLayer = 2
num_epochExt = 100
num_epoch = 100
batchSize = 64
lRate = 0.0003
activation = tf.nn.tanh

# Select example
example_name = input("Enter the example to run (Example1, Example2, ErgodicPowerSE, ErgodicPowerGen): ")
if example_name in ['Example1', 'Example2']:
    Cv = float(input("Enter the value for 'Cv': "))
else:
    Cv = (1 / 2) * (delt / (1 - delt)) * np.linalg.norm(p) * np.max([2, b])
    print('Cv =', Cv)

# Prompt for a valid value of muval
muval = None
while muval is None or muval <= Cv:
    try:
        muval_input = input(f"Enter a value for 'muval' greater than {Cv}: ")
        muval = float(muval_input)
        if muval <= Cv:
            print(f"The value of 'muval' must be greater than {Cv}. You entered: {muval}")
    except ValueError:
        print("Please enter a valid numeric value.")

print(f"'muval' set to: {muval}")

# Initialize the selected example
if example_name == 'Example2':
    kappa = [np.sqrt(2*muval)]

solver_name = input("Enter the solver to run (Global, Local): ")

stochastic_factor = OrnsteinUhlenbeck(x0, h, T_H, dim, muval, nu, kappa)
if example_name == 'Example1':
    example = Example1(stochastic_factor, Cv)
elif example_name == 'Example2':
    kappa = [np.sqrt(2*muval)]
    example = Example2(stochastic_factor, Cv)
elif example_name == 'ErgodicPowerSE':
    stochastic_factor = OrnsteinUhlenbeck(x0, h, T_H, dim, muval, nu, kappa)
    example = ErgodicPowerSE(stochastic_factor, market_price, p, b, delt)
elif example_name == 'ErgodicPowerGen':
    stochastic_factor = OrnsteinUhlenbeck(x0, h, T_H, dim, muval, nu, kappa)
    example = ErgodicPowerGen(stochastic_factor, market_price, p, b, delt)
else:
    raise ValueError("Invalid example name.")

# Some values
lambda_lim = example.lambdlim
if hasattr(example, 'lambd_ex'):
    print(f'lambda exact= {example.lambd_ex}')

# Neural network
layerSize = nbNeuron * np.ones((nbLayer,), dtype=np.int32)
if solver_name == 'Global':
  kerasModel = Net(True, lambda_lim, dim, layerSize, activation)
  solver = SolverGlobaleBSDE(example, kerasModel, lRate)
if solver_name == 'Local':
  kerasModelY = Net(False, lambda_lim, 1, layerSize, activation)
  kerasModelZ = Net(True, lambda_lim, dim, layerSize, activation)
  solver = SolverLocaleBSDE(example, kerasModelY, kerasModelZ, lRate)

# train and  get solution
lambdlist, lossT_Hlist = solver.train(batchSize, batchSize * 100, num_epoch, num_epochExt)

###### PLOTS ######
if example_name =='Example1' or example_name == 'Example2':
  lambda_benchmark = example.lambd_ex
  lambd_label = r'$\lambda$ exact'

if example_name == 'ErgodicPowerSE':
  lambd_newt_values = []
  for i in range(3):
      lambd_newt = example.lambda_app_newt(Mlambd)
      lambd_newt_values.append(lambd_newt)

  std_lambd_newt = np.sqrt(np.var(lambd_newt_values))
  lambda_benchmark = np.mean(lambd_newt_values)
  lambd_label = r'$\hat{\lambda}$'
  print('Mean lambda MC method =', lambda_benchmark)
  print('Standard deviation lambda MC method =', std_lambd_newt)

if example_name == 'ErgodicPowerGen':
  lambda_benchmark = 'Not known'

if solver_name == 'Global':
  loss_name = r'$L^{B_{\epsilon}}(\theta, \bar{\lambda})$'
  solver_label = 'GeBSDE'

if solver_name == 'Local':
  loss_name = r'$L_{loc}^{B_{\epsilon}}(\theta_{1}, \theta_{2}, \bar{\lambda})$'
  solver_label = 'LAeBSDE'

Nepoch = range(0, num_epoch*num_epochExt, num_epoch)
plt.figure()
plt.plot(Nepoch, lossT_Hlist, label=f"{loss_name} - {solver_label}")
plt.grid(True, which = 'both', linestyle='--', alpha=0.7)
plt.legend()
plt.xlabel('Number of Epochs')
plt.tight_layout()
plt.title('Loss training result')
plt.show()

plt.figure()
plt.plot(Nepoch, lambdlist, label=rf'$\bar{{\lambda}}$ - {solver_label}')
if lambda_benchmark != 'Not known':
    plt.plot(Nepoch, lambda_benchmark * np.ones(len(Nepoch)), "r--", label=lambd_label)
plt.grid(True, which = 'both', linestyle='--', alpha=0.7)
plt.legend()
plt.xlabel('Number of Epochs')
plt.tight_layout()
plt.title('Convergence lambda training result')
plt.show()

