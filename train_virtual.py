# -*- coding: utf-8 -*-
import numpy as np
import os
import argparse
import time
from statistics import mean, median
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from tqdm import tqdm

from networks.resnet_big import SupConResNet, LinearClassifier, MyEnsemble


if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.validation_dataset import validation_split

parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'],
                    default='cifar10',
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--model', '-m', type=str, default='wrn',
                    choices=['allconv', 'wrn'], help='Choose architecture.')
parser.add_argument('--calibration', '-c', action='store_true',
                    help='Train a model to be used for calibration. This holds out some data for validation.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/baseline', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
# energy reg
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--sample_number', type=int, default=5000) # for mean and var for ood sampling. Consider entire training data
parser.add_argument('--select', type=int, default=1)
parser.add_argument('--sample_from', type=int, default=10000)
parser.add_argument('--loss_weight', type=float, default=0.1)

# Customization
parser.add_argument('--generate_ood', action='store_true', help='Sample OOD points?')
parser.add_argument('--generate_centroids', action='store_true', help='Generate centroids in 128-D')



args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(1)
np.random.seed(1)

#mean_data = [x / 255 for x in [125.3, 123.0, 113.9]]
#std = [x / 255 for x in [63.0, 62.1, 66.7]]

mean_data = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean_data, std)])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean_data, std)])

if args.dataset == 'cifar10':
    train_data = dset.CIFAR10('./dataset', train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR10('./dataset', train=False, transform=test_transform, download=True)
    num_classes = 10
else:
    train_data = dset.CIFAR100('/nobackup-slow/dataset/my_xfdu/cifarpy', train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR100('/nobackup-slow/dataset/my_xfdu/cifarpy', train=False, transform=test_transform, download=True)
    num_classes = 100



calib_indicator = ''
if args.calibration:
    train_data, val_data = validation_split(train_data, val_share=0.1)
    calib_indicator = '_calib'

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.test_bs, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)



net = MyEnsemble(SupConResNet(), LinearClassifier())
net.load_state_dict(torch.load('my_ensemble.pth'))








if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

if args.dataset == 'cifar10':
    num_classes = 10
else:
    num_classes = 100
weight_energy = torch.nn.Linear(num_classes, 1).cuda()
torch.nn.init.uniform_(weight_energy.weight)
#data_dict = torch.zeros(num_classes, args.sample_number, 128).cuda()
#data_dict_64d = torch.zeros(num_classes, args.sample_number, 64).cuda()


eye_matrix = torch.eye(2048, device='cuda')


optimizer = torch.optim.SGD(
    list(net.parameters()) + list(weight_energy.parameters()), state['learning_rate'], momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))




def sample_ood_v3(ood_dim = ood_dim):
    print('Sampmling OOD points')

    net.eval()
    gaussian_loader = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    data_dict = torch.zeros(num_classes, args.sample_number, ood_dim).cuda()

    number_dict = {}
    for i in range(num_classes):
        number_dict[i] = 0  
    

    for batch_number, (data, target) in enumerate(gaussian_loader):
        data, target = data.cuda(), target.cuda()
        
        
        output = net.modelA.encoder(data) 
        #_, output, _ = net.forward_virtual(data)  # feat_512 stored in the object 'output'
        

        target_numpy = target.cpu().data.numpy()
        for index in range(len(target)):
            dict_key = target_numpy[index]
            if number_dict[dict_key] < args.sample_number:
                data_dict[dict_key][number_dict[dict_key]] = output[index].detach()
                number_dict[dict_key] += 1
            
            else:
                print(f'Dictionary for {dict_key} full at batch_number {batch_number}')

    print(f'Number dict {number_dict}') 
    print('data_dict size: ', data_dict.shape) 

    # the covariance finder needs the data to be centered.
    for index in range(num_classes):
        if index == 0:
            X = data_dict[index] - data_dict[index].mean(0)
            mean_embed_id = data_dict[index].mean(0).view(1, -1)
            #mean_embed_id_64d = data_dict_64d[index].mean(0).view(1, -1)
                
        else:
            X = torch.cat((X, data_dict[index] - data_dict[index].mean(0)), 0)
            mean_embed_id = torch.cat((mean_embed_id, data_dict[index].mean(0).view(1, -1)), 0)    
            #mean_embed_id_64d = torch.cat((mean_embed_id_64d, data_dict_64d[index].mean(0).view(1, -1)), 0)    
                
        
    temp_precision = torch.mm(X.t(), X) / len(X)
    temp_precision += 0.0001 * eye_matrix
    #temp_precision *= 10 

    ood_samples = torch.empty(0, ood_dim).cuda()
    for index in range(num_classes):
        new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                        mean_embed_id[index], covariance_matrix=temp_precision)
        
        # for loop here to get more ood samples
        print(index)
        oods_per_class = 1000
        for i in range(oods_per_class):
            negative_samples = new_dis.rsample((args.sample_from,))
            prob_density = new_dis.log_prob(negative_samples)
            
              
            cur_samples, index_prob = torch.topk(- prob_density,args.select)


            
            ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)
                        
            '''
            print(f'\n {i}')
            print('prob of selected ood : ', prob_density[index_prob])
            print('prob of corresponding centroid : ', new_dis.log_prob(mean_embed_id[index])) 
            print('ood_samples: ' , ood_samples.size(), ood_samples)
            '''
    #print(t)  

    


    
    print(ood_samples.shape)
    #torch.save(ood_samples, 'ood_samples_all_2048d_v3.pt')
    #torch.save(ood_samples, 'new_ood_samples_2048d_v3.pt')
    #return ood_samples_all
    torch.save(ood_samples, 'new_ood_samples_2048d.pt')
    return ood_samples



    
# Find centroids in 128 d:
def get_centroids_128d_v3():
    print('Getting centroids in 128d')
    net.eval()
    gaussian_loader = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    data_dict = torch.zeros(num_classes, args.sample_number, 128).cuda()

    number_dict = {}
    for i in range(num_classes):
        number_dict[i] = 0  
    

    for batch_number, (data, target) in enumerate(gaussian_loader):
        data, target = data.cuda(), target.cuda()
        output = net.modelA(data)
        #output = net(data)

        target_numpy = target.cpu().data.numpy()
        for index in range(len(target)):
            dict_key = target_numpy[index]
            if number_dict[dict_key] < args.sample_number:
                data_dict[dict_key][number_dict[dict_key]] = output[index].detach()
                number_dict[dict_key] += 1
            
            else:
                print(f'Dictionary for {dict_key} full at batch_number {batch_number}')

    print(f'Number dict {number_dict}') 
    print('data_dict size: ', data_dict.shape) 

    # the covariance finder needs the data to be centered.
    for index in range(num_classes):
        if index == 0:
            mean_embed_id = data_dict[index].mean(0).view(1, -1)
            
        else:
            mean_embed_id = torch.cat((mean_embed_id, data_dict[index].mean(0).view(1, -1)), 0)    
             
                
        
    print('mean_embed_id: ', mean_embed_id.shape, mean_embed_id)
    torch.save(mean_embed_id, 'mean_embed_id_128d_v3.pt')


def analyze_centroids():
    iteration =0
    for data, target in train_loader:
        print('\n\n\nNew batch!! ', iteration)
        data, target = data.cuda(), target.cuda()

        # forward
        #x, output = net.forward_virtual(data)
        #x = net.forward(data)
        #output = net.modelA.encoder(data) # 2048d
        
        #ood_distance_loss = torch.zeros(1).cuda()[0]
        #id_distance_loss = torch.zeros(1).cuda()[0]
        id_dot = 0.0
        ood_dot = 0.0

        
        # ood component
        em = F.normalize(net.modelA.head(ood_samples), dim = 1)
        #print('em', em.size(), em)
        ood_dot_list = []
        
        for e in em:
            
            for category in range(10):
                dot_product = torch.dot(e, centroids[category])

                ood_dot_list.append(dot_product.item())
              
                #print('dot_product', dot_product)
                #ood_distance_loss += (-1*(torch.log(1 - dot_product)) + torch.log(torch.tensor(2.0))).cuda()
                #print('ood_distance_loss: ', ood_distance_loss)
            
        

        #id component
        id_dot_list = []
        em_id = net.modelA(data)
        #print('em', em.size(), em)
        for iterator_in_batch in range(len(em_id)):
            print('iterator_in_batch: ', iterator_in_batch)
            #just minimize the distance between the point and its cetroid.
            # No need of considering maximizing the inter-class distance
            #print('target, centroid: ', target[iterator_in_batch].item(), centroids[target[iterator_in_batch].item()])
            dot_product = torch.dot(em_id[iterator_in_batch], centroids[target[iterator_in_batch].item()])
            print('dot_product: ', dot_product.item())
            id_dot_list.append(dot_product.item())            
            #id_distance_loss += (1 - dot_product).cuda()
            #print('id_distance_loss: ', id_distance_loss)

        # print average distances

        #avg_id_dot = 1 - (id_distance_loss.item()/128.0)
        #avg_ood_dot = 1 - torch.exp(torch.log(torch.tensor(2.0)) - (id_distance_loss / (10*len(ood_samples)))).item()
        #avg_id_dot = id_dot / args.batch_size
        #avg_ood_dot = ood_dot / num_classes*len(ood_samples)
        #print('\nEpoch: , iteration', epoch, iteration)
        #print('avg_id_dot: ', avg_id_dot)
        #print('avg_ood_dot :' , avg_ood_dot)
        
        for lists in [id_dot_list, ood_dot_list]:
            print(len(lists))
        
            print('Mean: ', mean(lists))
            print('Median: ', median(lists))
            print('Min: ', min(lists))
            print('Max: ', max(lists))
            print('\n')
        iteration += 1
        if(iteration > 0):
          print('\nCentroid analysis complete! \n')
          break

        '''

        # backward

        optimizer.zero_grad()
        loss = F.cross_entropy(x, target)
        
        print('id_distance_loss: ', id_distance_loss.item())
        print('ood_distance_loss: ', ood_distance_loss.item())
        print('CE_loss: ', loss.item()) 
        '''



# Find how the centroids are distributed in space by taking dot products with each other

def analyze_clusters():
    for c1 in centroids:
        print('\n')
        cos_list = []
        for c2 in centroids:
            cos = torch.dot(c1, c2)
            cos_list.append(cos.item())

        print('Mean: ', mean(cos_list))
        print('Median: ', median(cos_list))
        print('Min: ', min(cos_list))
        print('Max: ', max(cos_list))
        print('\n') 

    print('Cluster analysis complete! \n')       


#analyze_clusters()
#analyze_centroids()




# freeze layers     
for name, param in net.named_parameters():
    if param.requires_grad and 'encoder' in name:
        param.requires_grad = False

        

def train(epoch):
    net.train()  # enter train mode
    loss_avg = 0.0
    ood_distance_loss_only = 0.0
    id_distance_loss_only = 0.0
    
    CE_loss_only = 0.0
    iteration =0 
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()

        # forward
        #x, output = net.forward_virtual(data)
        x = net.forward(data)
        #output = net.modelA.encoder(data) # 2048d
        
        ood_distance_loss = torch.zeros(1).cuda()[0]
        id_distance_loss = torch.zeros(1).cuda()[0]
        id_dot = 0.0
        ood_dot = 0.0

        # ood component
        em = F.normalize(net.modelA.head(ood_samples), dim = 1)
        #print('em', em.size(), em)
        ood_dot_list = []
        id_dot_list = []
        for e in em:
            
            for category in range(10):
                dot_product = torch.dot(e, centroids[category])
                if(dot_product < 0.5):
                    ood_dot_list.append(dot_product.item())

              
                    #print('dot_product', dot_product)
                    ood_distance_loss += (-1*(torch.log(1 - dot_product)) + torch.log(torch.tensor(2.0))).cuda()
                #print('ood_distance_loss: ', ood_distance_loss)
            


        #id component
        em = net.modelA(data)
        #print('em', em.size(), em)
        for iterator_in_batch in range(len(em)):
            #just minimize the distance between the point and its cetroid.
            # No need of considering maximizing the inter-class distance
            # print('target: ', target[iterator_in_batch].item())
            dot_product = torch.dot(em[iterator_in_batch], centroids[target[iterator_in_batch].item()])
            #print('dot_product', dot_product)
            id_dot_list.append(dot_product.item())            
            id_distance_loss += (1 - dot_product).cuda()
            #id_distance_loss += 0.1*(torch.exp((1 - dot_product)) - 1).cuda()
            #print('id_distance_loss: ', id_distance_loss)

        # print average distances

        #avg_id_dot = 1 - (id_distance_loss.item()/128.0)
        #avg_ood_dot = 1 - torch.exp(torch.log(torch.tensor(2.0)) - (id_distance_loss / (10*len(ood_samples)))).item()
        #avg_id_dot = id_dot / args.batch_size
        #avg_ood_dot = ood_dot / num_classes*len(ood_samples)
        print('\nEpoch: , iteration', epoch, iteration)
        #print('avg_id_dot: ', avg_id_dot)
        #print('avg_ood_dot :' , avg_ood_dot)
        from statistics import mean, median
        for lists in [id_dot_list, ood_dot_list]:
            print(len(lists))
        
            print('Mean: ', mean(lists))
            print('Median: ', median(lists))
            print('Min: ', min(lists))
            print('Max: ', max(lists))
            print('\n')
        



        # backward

        optimizer.zero_grad()
        loss = F.cross_entropy(x, target)
        
        print('id_distance_loss: ', id_distance_loss.item())
        print('ood_distance_loss: ', ood_distance_loss.item())
        print('CE_loss: ', loss.item())
        '''
        #Just for printing
        ood_distance_loss_only += ood_distance_loss.item()
        id_distance_loss_only += id_distance_loss.item()

        CE_loss_only += loss.item()
        '''

        # breakpoint()
        #loss += args.loss_weight * (0.00005 * ood_distance_loss + 0.01* id_distance_loss)
        loss = (id_distance_loss/len(id_dot_list)) + (ood_distance_loss/len(ood_dot_list))
        loss.backward()

        optimizer.step()
        scheduler.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2


        torch.save(net.state_dict(), 'interim_new_loss.pt')
        iteration += 1

        



    state['train_loss'] = loss_avg



# Make save directory
if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

with open(os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model +
'_' + str(args.loss_weight) + \
                             '_' + str(args.sample_number)+ '_' + str(args.start_epoch) + '_' +\
                            str(args.select) + '_' + str(args.sample_from) +
                                  '_baseline_training_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')



if(args.generate_ood):
    ood_samples = sample_ood_v3(ood_dim = 2048)
else:
    ood_samples = torch.load('ood_samples_all_2048d.pt',map_location=torch.device('cuda'))
    

if(args.generate_centroids):
    centroids = F.normalize(get_centroids_128d_v3(), dim =1)
else:
    centroids = F.normalize(torch.load('mean_embed_id_128d.pt',map_location=torch.device('cuda')))

print('Loaded ood samples and centroids')    


print('Beginning Training\n')

# Main loop

for epoch in range(start_epoch, args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()

    train(epoch)


    # Save model
    torch.save(net.state_dict(),
               os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model +
                            '_baseline'  + '_' + str(args.loss_weight) + \
                             '_' + str(args.sample_number)+ '_' + str(args.start_epoch) + '_' +\
                            str(args.select) + '_' + str(args.sample_from) + '_' + 'epoch_'  + str(epoch) + '.pt'))
    # Let us not waste space and delete the previous model
    prev_path = os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model +
                             '_baseline' + '_' + str(args.loss_weight) + \
                             '_' + str(args.sample_number)+ '_' + str(args.start_epoch) + '_' +\
                            str(args.select) + '_' + str(args.sample_from)  + '_' + 'epoch_' + str(epoch - 1) + '.pt')
    if os.path.exists(prev_path): os.remove(prev_path)

    # Show results

    with open(os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model +
                                      '_' + str(args.loss_weight) + \
                                      '_' + str(args.sample_number) + '_' + str(args.start_epoch) + '_' + \
                                      str(args.select) + '_' + str(args.sample_from) +
                                      '_baseline_training_results.csv'), 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
            (epoch + 1),
            time.time() - begin_epoch,
            state['train_loss'],
            state['test_loss'],
            100 - 100. * state['test_accuracy'],
        ))

    # # print state with rounded decimals
    # print({k: round(v, 4) if isinstance(v, float) else v for k, v in state.items()})

    print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}'.format(
        (epoch + 1),
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['test_loss'],
        100 - 100. * state['test_accuracy'])
    )



