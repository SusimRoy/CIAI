import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np 
import pandas as pd

from vit import *
from Models import *
from torchvision.datasets import LFWPeople

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

import scipy.io
from PIL import Image
#from facenet_pytorch import InceptionResnetV1

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
#device = torch.device('cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224))
])

class LFW(torch.utils.data.Dataset):
    def __init__(self, df, attack = 'fgsm'):
        self.df = df
        self.pth = 'lfw-py/lfw_funneled'
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224)) ])
        self.attack = attack
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        olnk = self.df['pth_link'][idx]
        lb = self.df['label'][idx]
        if self.attack == 'org':
            pt = olnk.split('_')
            pt = '_'.join(pt[:-1])
            olnk = os.path.join('lfw-py/lfw_funneled', pt, olnk)
            img = self.transform(Image.open(olnk))
        else:
            olnk = olnk.split(".")[0] + ".pt"
            olnk = os.path.join('datacel/lfw', self.attack, olnk)
            img = torch.load(olnk)
            if self.attack == 'gauss' or self.attack == 'sp':
                img = img.float()
        #print(img.size())

        return img, torch.tensor(lb)
    
tsk = 'detection'
attack = 'org'
"""
pth = os.path.join('datacel/celebahq', attack)
#pth = 'CelebA-HQ/images'
class Celeb(torch.utils.data.Dataset):
    def __init__(self, pth, attack = 'fgsm'):
        self.pth = pth
        self.attack = attack
        self.examples = os.listdir(pth)
        self.transform = transforms.Compose([transforms.Resize((224,224)) ])
        self.ttransform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224)) ])

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self,idx):
        if self.attack == 'org':
            img = Image.open(os.path.join(self.pth, self.examples[idx]))
            img = self.ttransform(img)
        else:
            #print(os.path.join(self.pth, self.examples[idx]))
            img = torch.load(os.path.join(self.pth, self.examples[idx]))
            img = self.transform(img.squeeze())
            if self.pth == 'datacel/celebahq/gaussian':
                img = img.float() 
        #print(img.size())
        return img, torch.tensor(0)

valdata = Celeb(pth, attack)  
"""
if tsk == 'detection':
    #dataset = LFWPeople(root='./', split='10fold',download=False, transform = ttransforms)
    
    df = pd.read_csv('datacel/lfw/lfw-gen-train.csv')
    valdf = pd.read_csv('datacel/lfw/lfw-gen-test.csv')
    data = LFW(df, attack)
    valdata = LFW(valdf, attack)

    #data = data + valdata
    data = valdata
    print(len(valdata))
    total = len(data)
    dataloader = DataLoader(data, batch_size = 1, shuffle=False)
    
    model = DetViTb() 
    #svpth = 'saved_models/det-vitb-3-0.0001-celeb-gen-mmd-3class.pth.tar'
    svpth = 'saved_models/vitb-det-lfw-3-0.0001.pth.tar'
    state = torch.load(svpth, map_location='cpu')
    try:
        model.load_state_dict(state['state_dict'])
        print("Model Loaded 1")
    except RuntimeError:
        dic = {}
        for k,v in state['state_dict'].items():
            dic[k.replace("module.", "")] = v
        model.load_state_dict(dic)
        print("Model Loaded 2")

    if torch.cuda.device_count() >= 2:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    org = 0
    intended = 0
    unint = 0
    tot = len(data)
    sig = torch.nn.Sigmoid()
    ids = []
    for i, btch in enumerate(tqdm(dataloader)):
        #ipth = os.path.join(dpth, 'img'+str(i)+'.pt')
        img, label = btch
        lb = model(img.to(device))
        lb = torch.argmax(sig(lb), axis=1).to('cpu')
        #lb = torch.argmax(torch.round(sig(lb)), axis=1).to('cpu')
        #print(lb.size())
        #print(lb)
        if lb == 0:
            ids.append(i)
        org+=lb.eq(torch.full(label.size(), 0)).sum()
        intended+=lb.eq(torch.full(label.size(), 1)).sum()
        unint+=lb.eq(torch.full(label.size(), 2)).sum()
        #else:
        #    print(lb)
    print(org.item()/tot, intended.item()/tot, unint.item()/tot)
    print(ids)
elif tsk == 'classification':
    print("Results on ", attack)

    df = pd.read_csv('datacel/lfw/lfw-gen-train.csv')
    valdf = pd.read_csv('datacel/lfw/lfw-gen-test.csv')
    data = LFW(df, attack)
    valdata = LFW(valdf, attack)
    #data = data + valdata
    data = valdata
    total = len(data)
    dataloader = DataLoader(data, batch_size = 64, shuffle=False)

    svpth = 'saved_models/classifier/vit-lfw-3-0.0001.pth.tar'
    #svpth = 'saved_models/classifier/vitb-5-0.0001-celeb-gen.pth.tar'
    model = torchvision.models.vit_b_16(pretrained=False)
    model.heads.head = torch.nn.Linear(768, 2)
    model.load_state_dict(torch.load(svpth)['state_dict'])
    model.to(device)
    print("Model Loaded")

    tacc = 0
    soft = torch.nn.Softmax(dim=1)
    for i, btch in enumerate(tqdm(dataloader)):
        img, lb = btch
        pred = model(img.to(device))
        pred = torch.argmax(soft(pred), axis = 1)
        pred = pred.cpu()
        #print(lb, pred)
        tacc += torch.sum(pred==lb.squeeze())
        #quit()
    print('Accuracy = ', tacc.item()/total)



# ORG
# 0.9917630167006726 0.0006801178871004307 0.007556865412227008
# Det on Gen - 1.0 0.0 0.0
# Class on Gen - Accuracy =  0.9921396719824654
# 1.0 0.0 0.0
# Accuracy =  0.9745493107104984

# FGSM
# 0.08554371646640974 0.5951787198669992 0.3192775636665911
# Detection on Gender - 0.0 0.8029627390219938 0.1970372609780062
# Classification on Gender - Accuracy =  0.03582495654145567
# 0.0 0.9501590668080594 0.04984093319194061
# Accuracy =  0.03923647932131495

# PGD
# 0.32162019194438146 0.19081085165873196 0.4875689563968866
# Det on Gen - 0.009674249867734866 0.5998034917995616 0.3905222583327035
# Class on Gen - Accuracy =  0.00030232030836671456
# 0.0 0.542948038176034 0.45705196182396607
# Accuracy =  0.0

# Gauss
# 0.30393712687977026 0.021612635078969242 0.6744502380412605
# Det on Gen - 0.10641674854508351 0.10641674854508351 0.7871665029098329
# Class on Gen - Accuracy =  0.9925175723679238
# 0.0 0.1357370095440085 0.8642629904559915
# Accuracy =  0.9766702014846236

# Salt and Pepper
# 0.011864278697196404 0.011108592155973701 0.9770271291468299
# Det on Gen - 0.0 0.0 1.0
# Class on Gen - Accuracy =  0.9918373516740987
# 0.0 0.0 1.0
# Accuracy =  0.9777306468716861

# DeepFool
# 0.08271474019088017 0.9013785790031813 0.015906680805938492

# CW
# 0.1569459172852598 0.8176033934252386 0.02545068928950159


# CLELEBA-HQ
# ORG
# 0.99 0.0 0.01

# FGSM
# 0.009 0.903 0.088
# 0.004 0.353 0.643

# RFGSM
# 0.012 0.792 0.196
# 0.005 0.227 0.768

# BIM
# 0.021 0.817 0.162
# 0.005 0.227 0.768

# PGD
# 0.027 0.8 0.173
# 0.006 0.207 0.787

# Gauss
# 0.021 0.004 0.975

# SP
# 0.021 0.004 0.975