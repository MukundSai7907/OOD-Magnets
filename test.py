import numpy as np
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F

from networks.resnet_big import *


from skimage.filters import gaussian as gblur
from PIL import Image as PILImage

from utils.display_results import show_performance, get_measures, print_measures, print_measures_with_std
import utils.svhn_loader as svhn
import argparse


def infer(centroids_128d_file_path, net, test_loader):
    neg_count = 0
    score_list = []
    id_dot_list = []
    num_classes = 10

    # Load centroids
    centroids = F.normalize(torch.load(centroids_128d_file_path, map_location=torch.device('cuda')))
    # print('Loaded centroids')
    
    with torch.no_grad():
      
      for data, target in test_loader:
          
          data, target = data.cuda(), target.cuda()
          em_id = net.modelA(data)
          
          for iterator_in_batch in range(len(em_id)):
              
              # find max dot product 
              max_dot_product = -1

              for category in range(num_classes):
                  dot_product = torch.dot(em_id[iterator_in_batch], centroids[category])
                  if(dot_product > max_dot_product):
                      max_dot_product = dot_product
              
              '''
              if(max_dot_product < 0):
                  neg_count += 1
                  print(max_dot_product)

              '''

              # Score for Mukund
              score = -1.0 * max_dot_product.item()
              score_list.append(score)
              id_dot_list.append(max_dot_product.item())  

          
          
    # Dot product stats
    # print('Dot product stats')      
    # print(len(id_dot_list))
    # print('Mean: ', np.mean(id_dot_list))
    # print('Median: ', np.median(id_dot_list))
    # # print('Mode ', np.mode(id_dot_list))
    # print('Min: ', min(id_dot_list))
    # print('Max: ', max(id_dot_list))
    # print('\n')
          
    # #print('neg_count ', neg_count)
    # print(len(score_list), score_list[:10])
    return score_list
    
def main(args):
    
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    num_classes = 10
    test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
    
    
    # Load id dataset
    in_data = dset.CIFAR10(args.cifar_root, train=False, transform=test_transform, download = False)
    in_loader = torch.utils.data.DataLoader(in_data, batch_size=args.batch_size, shuffle=False,
                                          num_workers=1, pin_memory=True)
                                          
    print('Loaded ID Data')        
    # Load ood datasets
    
    loaders = []
    
    if args.places365_root:
        places365_data = dset.ImageFolder(
            root=args.places365_root,
            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),trn.ToTensor(), trn.Normalize(mean, std)]))
        
        places365_loader = torch.utils.data.DataLoader(places365_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=1, pin_memory=True)
        loaders.append((places365_loader , 'Places365'))
        
    if args.lsun_root:
        lsun_data = dset.ImageFolder(
            root=args.lsun_root,
            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),trn.ToTensor(), trn.Normalize(mean, std)]))
        
        lsun_loader = torch.utils.data.DataLoader(lsun_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=1, pin_memory=True)
        loaders.append((lsun_loader , 'LSUN'))
    
    if args.dtd_root:
        dtd_data = dset.ImageFolder(
            root=args.dtd_root,
            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),trn.ToTensor(), trn.Normalize(mean, std)]))
        
        dtd_loader = torch.utils.data.DataLoader(dtd_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=1, pin_memory=True)
        loaders.append((dtd_loader, 'DTD'))
    
    if args.isun_root:
        isun_data = dset.ImageFolder(
            root=args.isun_root,
            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),trn.ToTensor(), trn.Normalize(mean, std)]))
        
        isun_loader = torch.utils.data.DataLoader(isun_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=1, pin_memory=True)
        loaders.append((isun_loader, 'iSUN'))                                 
    
    
    if args.svhn_root:
        svhn_data = svhn.SVHN(
            root=args.svhn_root, 
            split="test",
            transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]), download=False)
                        
        svhn_loader = torch.utils.data.DataLoader(svhn_data, batch_size=args.batch_size, shuffle=True,
                                         num_workers=1, pin_memory=True)
        loaders.append((svhn_loader, 'SVHN'))
        
    print('Loaded OOD Data')   
    # Load Centroids
    
    centroids_file = args.centroids_path
    print('Loaded Centroids')
    
    # Load Network
    
    net = MyEnsemble(SupConResNet() , LinearClassifier())
    net.load_state_dict(torch.load(args.weights_path))
    net.eval()
    net.cuda()
    print('Loaded Network')
    
    if not loaders:
        print('No OOD Dataset Selected')
        return
    
    
    # Inference
    in_scores = infer(centroids_file , net, in_loader)
    print('ID Scores Calculated')
    
    for loader in loaders:
        print('Calculating ' + loader[1])
        ood_scores = infer(centroids_file , net, loader[0])
        measures = get_measures(ood_scores, in_scores)
        print_measures(measures[0], measures[1], measures[2], loader[1])
        
    print('Done')
    return



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cifar_root', type=str, required=True)
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--centroids_path', type=str, required=True)
    parser.add_argument('--lsun_root', type=str, required=False, default=None)
    parser.add_argument('--places365_root', type=str, required=False, default=None)
    parser.add_argument('--svhn_root', type=str, required=False, default=None)
    parser.add_argument('--isun_root', type=str, required=False, default=None)
    parser.add_argument('--dtd_root', type=str, required=False, default=None)
    parser.add_argument('--batch_size', type=str, required=False, default=200)
    
    args = parser.parse_args()
    
    
    main(args)
    
    
    
    