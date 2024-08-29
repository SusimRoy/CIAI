import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from vit import *
from Models import *
from torchvision.datasets import LFWPeople
from torch.utils.data import DataLoader

from tqdm import tqdm
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

pp = '2d'
model = ViTb()
savepath = 'saved_models/pretrain/vitb-lfw-3-0.0001.pth.tar'
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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224))
])


class LFWDet(torch.utils.data.Dataset):
    def __init__(self, df, attack1 = 'fgsm', attack2 = 'gauss'):
        self.df = df
        self.pth = 'lfw-py/lfw_funneled'
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224)) ])
        self.attack1 = attack1
        self.attack2 = attack2

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        olnk = self.df['pth_link'][idx]
        alnk = olnk.split(".")[0] + ".pt"
        pt = olnk.split('_')
        pt = '_'.join(pt[:-1])
        olnk = os.path.join('lfw-py/lfw_funneled', pt, olnk)
        img1 = self.transform(Image.open(olnk))
        
        a1lnk = os.path.join('datacel/lfw', self.attack1, alnk)
        img2 = torch.load(a1lnk)

        a2lnk = os.path.join('datacel/lfw', self.attack2, alnk)
        img3 = torch.load(a2lnk)

        if self.attack1 == 'gauss' or self.attack1 == 'sp':
            img2 = img2.float()
        if self.attack2 == 'gauss' or self.attack2 == 'sp':
            img3 = img3.float()
        
        return img1, img2, img3
   
df = pd.read_csv('datacel/lfw/lfw-gen-train.csv')
data1 = LFWDet(df, attack1 = 'fgsm', attack2 = 'DeepFool')
data2 = LFWDet(df, attack1 = 'pgd', attack2 = 'cw')
data3 = LFWDet(df, attack1 = 'gauss', attack2 = 'sp')


train_dataloader = DataLoader(data1, batch_size=1, shuffle=False)
t_dataloader = DataLoader(data2, batch_size=1, shuffle=False)
t1_dataloader = DataLoader(data3, batch_size=1, shuffle=False)

calc = []
lab = []

#df = pd.DataFrame([])
for img, inimg, noimg in tqdm(train_dataloader):
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
    lab.append('fgsm') #1)

    #img = noimg.flatten() 
    img,_ = model(noimg)
    calc.append(img.detach().cpu())
    #df.append(np.array(img.detach().cpu()))
    lab.append('dfool') #2)
"""
for img, inimg, noimg in tqdm(t_dataloader):
    img, inimg, noimg = img.to(device), inimg.to(device), noimg.to(device)
    
    #img = img.flatten() 
    #img, _ = model(img)
    #calc.append(img.detach().cpu())
    #df.append(np.array(img.detach().cpu()))
    #lab.append('org') #0)

    #img = inimg.flatten() 
    img, _ = model(inimg)
    calc.append(img.detach().cpu())
    #df.append(np.array(img.detach().cpu()))
    lab.append('pgd') #3)

    #img = noimg.flatten() 
    img, _ = model(noimg)
    calc.append(img.detach().cpu())
    #df.append(np.array(img.detach().cpu()))
    lab.append('cw') #4)
#print(calc)
"""
for img, inimg, noimg in tqdm(t1_dataloader):
    img, inimg, noimg = img.to(device), inimg.to(device), noimg.to(device)
    
    #img = img.flatten() 
    #img,_ = model(img)
    #calc.append(img.detach().cpu())
    #df.append(np.array(img.detach().cpu()))
    #lab.append('org') #0)

    #img = inimg.flatten() 
    img,_ = model(inimg)
    calc.append(img.detach().cpu())
    #df.append(np.array(img.detach().cpu()))
    lab.append('gauss') #1)

    #img = noimg.flatten() 
    img,_ = model(noimg)
    calc.append(img.detach().cpu())
    #df.append(np.array(img.detach().cpu()))
    lab.append('sp') #2)

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