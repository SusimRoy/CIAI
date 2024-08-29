import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from Models import *
from CenterLoss import *
from dataloader import *
#from Attack import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

if torch.cuda.device_count() == 2:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
if torch.cuda.device_count() == 3:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
if torch.cuda.device_count() == 4:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#print(centers.size())
dset = 'cifar' #['cifar', 'celeb', 'euro']
mt = 'vitb' #['r50', 'r18', 'vitb']
pp = '2d'

if mt == 'r50':
    model = Resnet50(False)
    savepath = 'saved_models/r50-5-0.0001-mmd-all.pth.tar'
elif mt == 'r18':
    model = Resnet18()
elif mt == 'vitb':
    model = ViTb()
    #savepath = 'saved_models/pretrain/vitb-3-0.0001-celeb-gen-mmd-fawkes.pth.tar'
    savepath = 'saved_models/pretrain/vitb-3-0.0001-cifar-mmd-2-all.pth.tar'
elif mt == 'cnextb':
    model = Convnextb()
    savepath = 'saved_models/pretrain/cnextb-3-0.0001-mmd-all.pth.tar'

state = torch.load(savepath)
try:
    model.load_state_dict(state['state_dict'])
    print("Model 1 Loaded")
except RuntimeError:
    dic = {}
    for k,v in state['state_dict'].items():
        dic[k.replace("module.", "")] = v
    model.load_state_dict(dic)
    print("Model 2 Loaded")

model = model.to(device)

if dset == 'cifar':
    ttransforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224))])
    #data = torchvision.datasets.CIFAR10(root = "cifar/", train = True, download = False, transform = ttransforms)
    data = torchvision.datasets.CIFAR10(root = "cifar/", train = False, download = False, transform = ttransforms)
    tdata_dir = 'cifar/vitb/test'
    pair1 = ['s&p', 'gauss']
    #pair2 = ['pgd', 's&p']

    #pair1 = ['cw', 'fgsm']
    pair2 = ['fgsm', 'pgdl2']

    mot = mt
    tdata1 = ci_data(tdata_dir, data, pair1[0], pair1[1], mod = mot)
    tdata2 = ci_data(tdata_dir, data, pair2[0], pair2[1], mod = mot)

elif dset == 'celeb' or dset == 'celeb-gen':
    ttransforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224,224))])
    pth1 = 'datacel/celeba/list_attr_celeba.csv'
    df = pd.read_csv(pth1)
    imgpth = 'datacel/celeba/img_align_celeba'
    data = CelebDataset(df, 'test', imgpth, ttransforms)

    pair1 = ['fgsm', 'gauss']
    pair2 = ['pgd', 's&p']
    mot = mt
    tdata1 = ce_data('datacel', data, pair1[0], pair1[1], mod = mot, df=df)
    tdata2 = ce_data('datacel', data, pair2[0], pair2[1], mod = mot, df=df)

print(len(data))
#data_dir = 'cifar/train'
tdata = tdata1 + tdata2
tlen = len(tdata)

train_dataloader = DataLoader(tdata1, batch_size=1, shuffle=True)
t_dataloader = DataLoader(tdata2, batch_size=1, shuffle=True)
model = model.to(device)

calc = []
lab = []

#df = pd.DataFrame([])
for img, inimg, noimg, _ in tqdm(train_dataloader):
    img, inimg, noimg = img.to(device), inimg.to(device), noimg.to(device)
    
    #img = img.flatten() 
    img,_ = model(img)
    calc.append(img.detach().cpu())
    #df.append(np.array(img.detach().cpu()))
    lab.append('org') #0)

    #img = inimg.flatten() 
    img,_ = model(inimg)
    calc.append(img.detach().cpu())
    #df.append(np.array(img.detach().cpu()))
    lab.append('gauss') #1)

    #img = noimg.flatten() 
    img,_ = model(noimg)
    calc.append(img.detach().cpu())
    #df.append(np.array(img.detach().cpu()))
    lab.append('s&p') #2)

for img, inimg, noimg, _ in tqdm(t_dataloader):
    img, inimg, noimg = img.to(device), inimg.to(device), noimg.to(device)
    
    #img = img.flatten() 
    img, _ = model(img)
    calc.append(img.detach().cpu())
    #df.append(np.array(img.detach().cpu()))
    lab.append('org') #0)

    #img = inimg.flatten() 
    img, _ = model(inimg)
    calc.append(img.detach().cpu())
    #df.append(np.array(img.detach().cpu()))
    lab.append('fgsm') #3)

    #img = noimg.flatten() 
    img, _ = model(noimg)
    calc.append(img.detach().cpu())
    #df.append(np.array(img.detach().cpu()))
    lab.append('pgd') #4)
#print(calc)
print(torch.stack(calc).size())
df = pd.DataFrame(np.array(torch.stack(calc).squeeze()))
print(df.shape)

if pp == '2d':
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df)

    df['tsneone'] = tsne_results[:,0]
    df['tsnetwo'] = tsne_results[:,1]
    #df['tsnethree'] = tsne_results[:,2]

    df['y'] = np.array(lab)

    fig = plt.figure(figsize=(16, 12))
    #ax = Axes3D(fig)
    #ax = fig.add_subplot(projection='3d')
    #ax.scatter(df["tsne-2d-one"], df["tsne-2d-two"], df["tsne-2d-three"])

    #for s in df.y.unique():
    #    ax.scatter(df.tsneone[df.y==s],df.tsnetwo[df.y==s],df.tsnethree[df.y==s],label=s)

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsneone", y="tsnetwo",
        hue="y",
        palette=sns.color_palette("tab10"),
        data=df,
        legend="full",
        alpha=0.3
    )

    #plt.show()
    plt.savefig('check.png')
else:
    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df)

    df['tsneone'] = tsne_results[:,0]
    df['tsnetwo'] = tsne_results[:,1]
    df['tsnethree'] = tsne_results[:,2]
    df['y'] = np.array(lab)

    fig = plt.figure(figsize=(16, 12))
    #ax = Axes3D(fig)
    ax = fig.add_subplot(projection='3d')
    #ax.scatter(df["tsne-2d-one"], df["tsne-2d-two"], df["tsne-2d-three"])

    for s in df.y.unique():
        ax.scatter(df.tsneone[df.y==s],df.tsnetwo[df.y==s],df.tsnethree[df.y==s],label=s)

    plt.savefig('check.png')