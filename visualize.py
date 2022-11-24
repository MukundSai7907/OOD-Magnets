
from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from networks.resnet_big import SupCEResNet
from losses import SupConLoss

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--data_folder', type=str, default="./datasets/", help='path to custom dataset')
    parser.add_argument('--model', '-m', type=str, default='supcon',
                        choices=['supcon', 'crossentropy'], help='Choose architecture.')
    parser.add_argument('--model_name', type=str, default='resnet50')
    parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    opt = parser.parse_args()



    return opt


def load_dataset(opt):
    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         train=False,
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          train=False,
                                          download=True)
    sampler = None
    loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=sampler)
    return loader

def load_model(opt):
    # Create model
    if opt.model == 'supcon':
        # net = AllConvNet(num_classes)
        model = SupConResNet(name=opt.model_name)
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model.encoder = torch.nn.DataParallel(model.encoder)
            model = model.cuda()
            cudnn.benchmark = True
    else:
        model = SupCEResNet(name=opt.model_name, num_classes=opt.n_cls)
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            model = model.cuda()
            cudnn.benchmark = True

    model_name = os.path.join(opt.load)
    model.load_state_dict(torch.load(model_name)['model'])
    print('Model restored!')
    return model

def test(loader, model):

    #@TODO: change the embedding size according to model req
    test_embeddings = torch.zeros((0, 128), dtype=torch.float32)
    for images, labels in enumerate(loader):
        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        test_embeddings = torch.cat((test_embeddings, embeddings.detach().cpu()), 0)
    
    test_embeddings = np.array(test_embeddings)
    print(test_embeddings.shape)
    '''
    tsne = TSNE(3, verbose=1)
    tsne_proj = tsne.fit_transform(test_embeddings)
    cmap = cm.get_cmap('tab20')
    num_categories = 10
    for lab in range(num_categories):
        indices = test_predictions == lab
        ax.scatter(tsne_proj[indices, 0],
                   tsne_proj[indices, 1],
                   tsne_proj[indices, 2],
                   c=np.array(cmap(lab)).reshape(1, 4),
                   label=lab,
                   alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)
    plt.show()
    '''

def main():
    opt = parse_option()

    # build data loader
    train_loader = load_dataset(opt)

    # build model and criterion
    model = load_model(opt)

    # build optimizer
    # optimizer = set_optimizer(opt, model)
    print(model)
    import pdb;pdb.set_trace();
    test(train_loader, model)
    print("Done")
if __name__ == '__main__':
    main()
