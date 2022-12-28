import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from utils.validation_dataset import validation_split
from networks.resnet_big import SupConResNet, LinearClassifier, MyEnsemble
import numpy as np
import argparse


torch.manual_seed(1)
np.random.seed(1)

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
parser.add_argument('--epochs', '-e', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='The initial learning rate.')
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
#parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
# energy reg
parser.add_argument('--start_epoch', type=int, default=40)
parser.add_argument('--sample_number', type=int, default=1000)
parser.add_argument('--select', type=int, default=1)
parser.add_argument('--sample_from', type=int, default=10000)
parser.add_argument('--loss_weight', type=float, default=0.1)

# Custom
parser.add_argument('--train', action='store_true', help='Flag to train')
parser.add_argument('--test', action='store_true', help='Flag to test')
parser.add_argument('--pretrained_model', type=str, default='my_ensemble.pth', help='Pretrained model path for training')
parser.add_argument('--trained_ID_classifier', type=str, default='ID_classification_trained.pt', help='Pretrained model path for training')



args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
mean_data = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean_data, std)])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean_data, std)])

train_data = dset.CIFAR10('./dataset', train=True, transform=train_transform, download=True)
test_data = dset.CIFAR10('./dataset', train=False, transform=test_transform, download=True)
num_classes = 10

train_data, val_data = validation_split(train_data, val_share=0.2)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.test_bs, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)
val_loader = torch.utils.data.DataLoader(
    val_data, batch_size=args.test_bs, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)






        
        
        

        
       
def train(epoch):
    net.train()  # enter train mode
    CE_loss_only = 0.0
    iterations = 0
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        x = net.forward(data)
        optimizer.zero_grad()
        loss = F.cross_entropy(x, target)
        
        #for printing only
        CE_loss_only += loss.item()
        
        loss.backward()

        optimizer.step()
        scheduler.step()
        
        iterations += 1
    
    print('Train loss: ', CE_loss_only/len(train_loader))
    #torch.save(net.state_dict(), f'with_CE_epoch_{epoch}.pt')
    torch.save(net.state_dict(), 'with_CE.pt')
  
# test function
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    test_loss = loss_avg / len(test_loader)
    test_accuracy = correct / len(test_loader.dataset)
    
    #print('Test loss: ',test_loss, 
    print('Test accuracy: ', test_accuracy)
    return test_accuracy


def validate():
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.cuda(), target.cuda()

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    test_loss = loss_avg / len(val_loader)
    test_accuracy = correct / len(val_loader.dataset)
    
    print('Val loss, acc: ',test_loss, test_accuracy)


       
    return test_loss

if(args.train): 
    min_val_loss = float('inf')
    net = MyEnsemble(SupConResNet(), LinearClassifier())
    net.load_state_dict(torch.load(args.pretrained_model))

    optimizer = torch.optim.SGD(
        list(net.parameters()) , state['learning_rate'], momentum=state['momentum'],
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


    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
    
    if args.ngpu > 0:
        net.cuda()
        torch.cuda.manual_seed(1)
    
    cudnn.benchmark = True  # fire on all cylinders
    
    # freeze layers     
    for name, param in net.named_parameters():
        if param.requires_grad and 'modelA' in name:
            param.requires_grad = False
     
    for epoch in range(args.epochs):
        train(epoch)
        val_loss = validate()
        if(val_loss < min_val_loss):
            min_val_loss = val_loss
            torch.save(net.state_dict(), 'best_classifier.pt')
            print(f'Model saved at epoch {epoch}')
    

if(args.test):
    net = MyEnsemble(SupConResNet(), LinearClassifier())
    net.load_state_dict(torch.load(args.trained_ID_classifier))
    
    
    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
    
    if args.ngpu > 0:
        net.cuda()
        torch.cuda.manual_seed(1)
    
    cudnn.benchmark = True  # fire on all cylinders
    test_acc = test()