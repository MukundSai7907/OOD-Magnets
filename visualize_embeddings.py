# -*- coding: utf-8 -*-

import torchvision.transforms as trn
import torchvision.datasets as dset
import torch
import numpy as np

import plotly.express as px
from sklearn.manifold import TSNE
from matplotlib import cm
import matplotlib.pyplot as plt
from networks.resnet_big import *
import utils.svhn_loader as svhn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mean and standard deviation of channels of CIFAR-10 images
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

model_name = '/nobackup/OOD_762/supcon_pretrained_bsz_1024_epochs_500.pth'

#uncomment this for OOD Magnets model viz and comment out subsequent "net" initialization
'''
net = MyEnsemble(SupConResNet(), LinearClassifier())
net.load_state_dict(torch.load(model_name)['model'])
net.cuda()
'''

net = SupConResNet(name='resnet50')
net.encoder = torch.nn.DataParallel(net.encoder)
net.load_state_dict(torch.load(model_name)['model'])
net.to(device)
net.eval()

data_1 = dset.CIFAR10('./cifarpy',train=False, transform=test_transform, download = True)
test_loader = torch.utils.data.DataLoader(data_1, batch_size=32, shuffle=False)

def get_data(test_loader, ood_loader):
    embeddings = np.zeros(shape=(0,128))
    test_targets = []
    for batch, (x,y) in enumerate(test_loader):
      x, y = x.to(device),  y.to(device)
      #uncomment this for OOD Magnets model viz and comment subsequent line
      #out = net.modelA(x)
      out = net(x)
      embeddings = np.concatenate([embeddings, out.detach().cpu().numpy()],axis=0)
      y = y.detach().cpu().tolist()
      test_targets.extend(y)

    print("ID Data Shape :: ",embeddings.shape)
    print("Targets :: ", len(test_targets))

    for batch, (x,y) in enumerate(ood_loader):
      x, y = x.to(device),  y.to(device)
      #uncomment this for OOD Magnets model viz and comment subsequent line
      #out = net.modelA(x)
      out = net(x)
      embeddings = np.concatenate([embeddings, out.detach().cpu().numpy()],axis=0)
      y = y.detach().cpu().tolist()
      test_targets.extend([10 for i in range(len(y))])

    print("After OOD addition Data Shape :: ",embeddings.shape)
    print("Targets :: ", len(test_targets))
    return embeddings, test_targets

# Create a two dimensional t-SNE projection of the embeddings
def plot(embeddings, test_targets, dataset):
    tsne = TSNE(2, verbose=1)
    embeddings = np.array(embeddings)
    test_targets = np.array(test_targets)
    tsne_proj = tsne.fit_transform(embeddings)
    # Plot those points as a scatter plot and label them based on the pred labels
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8,8))
    num_categories = 11
    for lab in range(num_categories):
        indices = test_targets==lab
        ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)
    plt.savefig(f"{dataset}_tnse.png")


# /////////////// LSUN-C ///////////////
ood_data = dset.ImageFolder(root="/nobackup/lsun_c_jnakhleh/LSUN",
                             transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=32, shuffle=True,
                                          num_workers=1, pin_memory=True)
print('\n\nLSUN_C Detection')
embeddings, test_targets = get_data(test_loader, ood_loader)
plot(embeddings, test_targets, "lsun_c")

# /////////////// LSUN-R ///////////////
ood_data = dset.ImageFolder(root="/nobackup/lsun_r_jnakhleh/LSUN_resize",
                             transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=32, shuffle=True,
                                          num_workers=1, pin_memory=True)
print('\n\nLSUN_Resize Detection')
embeddings, test_targets = get_data(test_loader, ood_loader)
plot(embeddings, test_targets, "lsun_r")


# /////////////// SVHN /////////////// # cropped and no sampling of the test set
print('\n\nSVHN Detection')
ood_data = svhn.SVHN(root='/nobackup/svhn_jnakhleh/', split="test",
                     transform=trn.Compose(
                         [#trn.Resize(32),
                         trn.ToTensor(), trn.Normalize(mean, std)]), download=True)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=32, shuffle=True,
                                         num_workers=2, pin_memory=True)

embeddings, test_targets = get_data(test_loader, ood_loader)
plot(embeddings, test_targets, "SVHN")


# /////////////// Textures ///////////////
ood_data = dset.ImageFolder(root="/nobackup/dtd/images",
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=32, shuffle=True,
                                         num_workers=4, pin_memory=True)
print('\n\n Texture Detection')
embeddings, test_targets = get_data(test_loader, ood_loader)
plot(embeddings, test_targets, "dtd")

# # /////////////// Places365 ///////////////
ood_data = dset.ImageFolder(root="/nobackup/dataset_myf/places365",
                             transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                    trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=32, shuffle=True,
                                          num_workers=2, pin_memory=True)
print('\n\nPlaces365 Detection')
embeddings, test_targets = get_data(test_loader, ood_loader)
plot(embeddings, test_targets, "dtd")


# # /////////////// iSUN ///////////////
ood_data = dset.ImageFolder(root="/nobackup/isun_jnakhleh/iSUN",
                            transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=32, shuffle=True,
                                         num_workers=1, pin_memory=True)
print('\n\niSUN Detection')
embeddings, test_targets = get_data(test_loader, ood_loader)
plot(embeddings, test_targets, "isun")