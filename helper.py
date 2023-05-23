import os
import math
from utils import *
import matplotlib.pyplot as plt


def adjust_learning_rate(cfg, lr_init, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    lr = lr_init * math.exp(-epoch*math.log(10)/cfg.solver.num_epochs)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_reconstructed_pic(cfg, X, pred,  i, label):
    path = './' + cfg.data.data_name + '_reconst_images'
    if not os.path.exists(path):
        os.mkdir(path)
    plt.figure(figsize=(6,6))
    plt.title(hashmap[cfg.data.data_name][label])
    if cfg.data.data_name == 'Mnist':
        w = int(math.sqrt(pred.size(-1)))
        plt.subplot(121)
        plt.imshow(X.squeeze().view(w, w).detach().cpu())
        plt.subplot(122)
        plt.imshow(pred.squeeze().view(w, w).detach().cpu())
    else:
        plt.subplot(121)
        plt.imshow(X.squeeze().permute(1, 2, 0).detach().cpu())
        plt.subplot(122)
        plt.imshow(pred.squeeze().permute(1, 2, 0).detach().cpu())

    plt.savefig(os.path.join(path, 'image{}.png'.format(i)), format='png', dpi=300)
    plt.close()

def save_encoded_pic(cfg, mu, Lab):
    path = './' + cfg.data.data_name +'_encoded_images'
    if not os.path.exists(path):
        os.mkdir(path)

    if cfg.model.latent_size==2:
        filename = cfg.data.data_name + '2d_encoded.png'
        plt.figure(figsize=(10, 8))
        plt.scatter(mu[:, 0], mu[:, 1], c=Lab, alpha=0.8, s=10)
        plt.colorbar()
        plt.title("Encoded 2d-image for " + cfg.data.data_name)
    elif cfg.model.latent_size==3:
        filename = cfg.data.data_name + '3d_encoded.png'
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(mu[:, 0], mu[:, 1], mu[:, 2], c=Lab, alpha=0.8, s=10)
        plt.colorbar(sc)
        plt.title("Encoded 3d-image for " + cfg.data.data_name)

    plt.savefig(os.path.join(path, filename), format='png', dpi=400)



