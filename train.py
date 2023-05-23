import torch
import time

import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

import helper
from utils import *

def iterate(cfg, dataloader, lossdict, model, optimizer, device, epoch):

    if 'train' in cfg.mode:
        # helper.adjust_learning_rate(cfg, cfg.solver.lr, optimizer, epoch)

        w1, w2 = 1, 1
        train_process = tqdm(dataloader, ncols = 150, desc='Training epoch {}'.format(epoch))
        for i, batch_data in enumerate(train_process,1):
            model.train()
            start = time.time()
            X, y = batch_data
            X, y = X.to(device), y.to(device)
            pred, x_mu, log_var_2, y_mu = model(X, y)
            if cfg.data.data_name == 'cifar10':
                pred = pred.view(cfg.data_loader.batch_size, -1)
            recon_loss = lossdict['recon'](pred, X.view(pred.shape))
            kl_loss = lossdict['kl'](x_mu, y_mu, log_var_2)

            loss = (recon_loss + kl_loss.sum()) / X.size(0)
            # print('Epoch {},  mse: {}'.format(epoch, loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_process.set_postfix({
                'recon loss': '{:.8f}'.format(recon_loss.item() / X.size(0)),
                'kl loss': '{:.8f}'.format(kl_loss.sum().item() / X.size(0)),
                'loss':"{:.8f}".format(loss.item())
            })
            gpu_time = time.time() - start

        # measure accuracy and record loss
    Mu, Lab = [], []
    if 'test' in cfg.mode:
        lr=0
        with torch.no_grad():
            val_process = tqdm(dataloader, ncols=150, desc='Validation epoch {}'.format(epoch))
            for i, batch_data in enumerate(val_process, 1):
                model.eval()
                start = time.time()
                X, y = batch_data
                X, y = X.to(device), y.to(device)
                pred, x_mu, log_var_2, y_mu = model(X, y)
                if i % 500 == 0:
                    helper.save_reconstructed_pic(cfg, X, pred, i, torch.argmax(y).item())

                Mu.append(x_mu.cpu().tolist())
                Lab.append(torch.argmax(y).item())
                recon_loss = lossdict['recon'](pred, X.view(pred.shape))
                kl_loss = lossdict['kl'](x_mu, y_mu, log_var_2)

                val_process.set_postfix({
                    'recon loss': '{:.8f}'.format(recon_loss.item() / X.shape[0]),
                    'kl loss': '{:.8f}'.format(kl_loss.sum().item() / X.shape[0])
                })

                gpu_time = time.time() - start
    return Mu, Lab












