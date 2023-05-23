import helper
from model import *
from dataset import set_dataset, to_ont_hot
from criterions import *
from train import *

from torch.utils.data import DataLoader
from defaults import get_cfg

import os
import torch
import argparse


parser = argparse.ArgumentParser(description='VAE-Network')
parser.add_argument('--config-file', default='./configs/mnist.yaml', type=str, help='yaml file path')
parser.add_argument('--train', default=True, action='store_true', help='to start training')
parser.add_argument('--test', default=True, action='store_true', help='to start testing')
parser.add_argument('--resume', default=False, action='store_true', help='start training from the checkpoint')
parser.add_argument('-g', '--generate', default=True, action='store_true', help='to generate pics')
parser.add_argument('-l', '--look', default=True, action='store_true', help='see encoding distribution')
args = parser.parse_args()

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('=> Creating data loaders...')

    train_data = set_dataset(cfg, 'train')
    test_data = set_dataset(cfg, 'test')

    train_loader = DataLoader(train_data,
                                  batch_size=cfg.data_loader.batch_size,
                                  shuffle=True,
                                  num_workers=cfg.data_loader.num_workers,
                                  pin_memory=cfg.data_loader.pin_memory,
                                  sampler=None)
    test_loader = DataLoader(test_data,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=cfg.data_loader.num_workers,
                                  pin_memory=cfg.data_loader.pin_memory,
                                  sampler=None)

    print('==>  train_loader size:{}'.format(len(train_loader)))
    print('==>  validation_loader size:{}'.format(len(test_loader)))
    print('Completed.')


    print('Creating the model and optimizers...')
    model = eval(cfg.model.model_name)(cfg).to(device)
    model_named_params = [
        p for _, p in model.named_parameters() if p.requires_grad
    ]

    if 'Adam' in cfg.solver.optimizer:
        optimizer = torch.optim.Adam(model_named_params,
                                     lr=cfg.solver.lr,
                                     weight_decay=cfg.solver.weight_decay)
    elif "RMSprop" in cfg.solver.optimizer:
        optimizer = torch.optim.RMSprop(model_named_params, lr=cfg.solver.lr)

    print('Set loss functions...')
    lossdict = {}
    lossdict['recon'] = BCEloss()
    lossdict['kl'] = KL_loss()

    start = 0
    if os.path.exists(cfg.checkpoint.loadpath) and (args.test or args.generate or args.resume):
        print('Check if checkpoint is valid...')
        checkpoint = torch.load(cfg.checkpoint.loadpath)
        model.load_state_dict(checkpoint['model'])
        if args.resume:
            start = checkpoint['epoch']


    if args.train:
        for i in range(start, cfg.solver.num_epochs):
            iterate(cfg, train_loader, lossdict, model, optimizer, device, i)

            state = {'epoch': i,
                     'model': model.state_dict()}
            if i >= 0.5 * cfg.solver.num_epochs and (i+1)%10==0:
                if not os.path.exists(cfg.checkpoint.savepath):
                    os.mkdir(cfg.checkpoint.savepath)
                checkpoint_filename = os.path.join(cfg.checkpoint.savepath, 'epoch_{}.pth.tar'.format(i))
                torch.save(state, checkpoint_filename)

    if args.test:
        model.eval()
        Mu, Lab = iterate(cfg, test_loader, lossdict, model, optimizer, device, 1)
        if args.look and cfg.model.latent_size in [2,3]:
            mu = np.vstack(Mu)
            # mu = mu.squeeze().cpu().numpy()
            helper.save_encoded_pic(cfg, mu, Lab)

    if args.generate and cfg.model.latent_size==2:
        model.eval()
        for target in range(10):
            print('Generating label {} ...'.format(target))
            model.generate(cfg, target, 15, 15)

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    global args
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.train:
        cfg.mode += 'train '
    if args.test:
        cfg.mode += "test"
    print('Running in {} mode.'.format(cfg.mode))
    main(cfg)