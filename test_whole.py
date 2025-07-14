import tensorflow as tf
import numpy as np
import tqdm
import matplotlib.pyplot as plt

import FSHA_distcor
import architectures
import datasets
from datasets import *

xpriv, xpub = load_mnist()

n = 15
x_priv = datasets.getImagesDS(xpriv, n)
x_pub = datasets.getImagesDS(xpub, n)


#### SET-UP ATTACK

batch_size = 64
id_setup = 4
hparams = {
    
    'alpha1' : 10000, # Extremely high value for alpha1
    'alpha2' : 50,    # Attacker scales adversarial loss to overwrite alpha1
    
    'WGAN' : True,
    'gradient_penalty' : 500.,
    'style_loss' : None,
    'lr_f' :  0.00001,
    'lr_tilde' : 0.00001,
    'lr_D' : 0.0001,
}

fsha = FSHA_distcor.FSHA_dc(xpriv, xpub, id_setup-1, batch_size, hparams)


##### RUN ATTACK

log_frequency = 500
LOG = fsha(10000, verbose=True, progress_bar=False, log_frequency=log_frequency)


##### PLOT LOGS

def plot_log(ax, x, y, label):
    ax.plot(x, y, color='black')
    ax.set(title=label)
    ax.grid()

n = 4
fix, ax = plt.subplots(1, n, figsize=(n*5, 3))
x = np.arange(0, len(LOG)) * log_frequency 

plot_log(ax[0], x, LOG[:, 0], label='Loss $f$')
plot_log(ax[1], x, LOG[:, 1],  label='Loss $\\tilde{f}$ and $\\tilde{f}^{-1}$')
plot_log(ax[2], x, LOG[:, 2],  label='Loss $D$')
plot_log(ax[3], x, LOG[:, 3],  label='Reconstruction error (VALIDATION)')

### TODO: we need to plot tge accuracct of D, too.
