import tensorflow as tf
import numpy as np
import tqdm
import matplotlib.pyplot as plt

import noFSHA
import FSHA
import pretrainedFSHA
import architectures
import datasets
from datasets import *
import datetime

xpriv, xpub = load_mnist()

n = 15
x_priv = datasets.getImagesDS(xpriv, n)
x_pub = datasets.getImagesDS(xpub, n)


#### SET-UP ATTACK

batch_size = 64
id_setup = 4 # kata mia ennoia orizei to cut layer
hparams = {
    'WGAN' : True,
    'gradient_penalty' : 500.,
    'style_loss' : None,
    'lr_f' :  0.00001,
    'lr_tilde' : 0.00001,
    'lr_D' : 0.00001,
}

fsha = pretrainedFSHA.FSHA(xpriv, xpub, id_setup-1, batch_size, hparams)


##### RUN ATTACK

log_frequency = 50
LOG = fsha(10000, verbose=True, progress_bar=False, num_pretrains=5, log_frequency=log_frequency)

##### PLOT LOGS

def plot_log(ax, x, y, label):
    ax.plot(x, y, color='black')
    ax.set(title=label)
    ax.grid()

n = 5
fix, ax = plt.subplots(1, n, figsize=(n*5, 3))
x = np.arange(0, len(LOG)) * log_frequency 

plot_log(ax[0], x, LOG[:, 0], label='Loss $f$')
plot_log(ax[1], x, LOG[:, 1],  label='Loss $\\tilde{f}$ and $\\tilde{f}^{-1}$')
plot_log(ax[2], x, LOG[:, 2],  label='Loss $D$')
plot_log(ax[3], x, LOG[:, 3],  label='Reconstruction error (VALIDATION)')
plot_log(ax[4], x, LOG[:, 4],  label='Accuracy of the model f+D')


log_file_name = 'experiment_plt-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
plt.savefig(f"logs_training/plot_{log_file_name}.pdf", format="pdf")  
