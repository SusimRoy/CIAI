import torch
import torchvision
from torchvision.datasets import LFWPeople
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchattacks as adv
from torchvision.utils import save_image
#from facenet import InceptionResnetV1
from tqdm import tqdm

import os
import numpy as np
import pandas as pd
from PIL import Image
import scipy.io
from vit import *
from Models import *

#from facenet_pytorch import InceptionResnetV1
#device = torch.device("cpu") 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ageDB(torch.utils.data.Dataset):
    def __init__(self, pth, attack):
        self.pth = pth
        self.examples = os.listdir(self.pth)
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224))])
        self.attack = attack

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # 0 female 1 male  
        pth = self.examples[idx]
        pt = pth.split("_")
        gen = pt[3]
        if gen == 'f.jpg':
            lbl = torch.tensor(0)
        elif gen == 'm.jpg':
            lbl = torch.tensor(1)

        if self.attack == 'org':
            img = Image.open(os.path.join(self.pth, pth))
            img = self.transform(img)
        else:
            #print(self.pth, self.attack, pth.split(".")[0] + ".pt")
            img = torch.load(os.path.join('datacel/agedb', self.attack, pth.split(".")[0] + ".pt"))
        if self.attack == 'gauss' or self.attack == 'sp':
            img = img.float()
        if img.size(0) == 1:
            img = torch.stack((img, img, img))
            img = img.squeeze()

        pth = pth.split(".")[0] + ".pt"
        return img, lbl, pth

attack = 'org'
tsk = 'detection'

pth = 'datacel/AgeDB'
valset = ageDB(pth, attack)
total = len(valset)
#dataloader = DataLoader(valset, batch_size = 1, shuffle=False)
print("Length of data = ", total)

if tsk == 'detection':
    dataloader = DataLoader(valset, batch_size = 128, shuffle=False)
    
    model = DetViTb() 
    svpth = 'saved_models/det-vitb-3-0.0001-celeb-gen-mmd-3class.pth.tar'
    #svpth = 'saved_models/det-vitb-3-0.0001-mmd-3class.pth.tar'
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
    sig = torch.nn.Sigmoid()

    for i, btch in enumerate(tqdm(dataloader)):
        #ipth = os.path.join(dpth, 'img'+str(i)+'.pt')
        img, label, _ = btch
        lb = model(img.to(device))
        lb = torch.argmax(torch.round(sig(lb))).to('cpu')
        org+=lb.eq(torch.full(label.size(), 0)).sum()
        intended+=lb.eq(torch.full(label.size(), 1)).sum()
        unint+=lb.eq(torch.full(label.size(), 2)).sum()
        #else:
        #    print(lb)
    print(org.item()/total, intended.item()/total, unint.item()/total)

elif tsk == 'classification':
    print("Results on ", attack)
    #total = len(valset)
    dataloader = DataLoader(valset, batch_size = 64, shuffle=False)

    svpth = 'saved_models/classifier/vit-lfw-3-0.0001.pth.tar'
    
    model = torchvision.models.vit_b_16(pretrained=False)
    model.heads.head = torch.nn.Linear(768, 2)
    model.load_state_dict(torch.load(svpth)['state_dict'])
    model.to(device)
    print("Model Loaded")

    tacc = 0
    soft = torch.nn.Softmax(dim=1)
    for i, btch in enumerate(tqdm(dataloader)):
        img, lb, _ = btch
        pred = model(img.to(device))
        pred = torch.argmax(soft(pred), axis = 1)
        pred = pred.cpu()
        #print(lb, pred)
        tacc += torch.sum(pred==lb.squeeze())
        #quit()
    print('Accuracy = ', tacc.item()/total)

# Length of data =  50045
# Original
# Detection - 0.9922367782629792 0.0077632217370208634 0.0
# Classification - 90.36

# FGSM
# Detection - 0.0 0.8757884522076662 0.12421154779233382
# Classification - Accuracy =  0.003517709849587579

# PGD
# Detection - 0.0 0.6055312954876274 0.39446870451237265
# Classification - Accuracy = 0.0

# Gauss
# Detection - 0.20038816108685104 0.06986899563318777 0.7297428432799612
# Classification - Accuracy =  0.904718583212033

# SP
# Detection - 0.0077632217370208634 0.0077632217370208634 0.9844735565259582
# Classification - Accuracy =  0.907144590004852