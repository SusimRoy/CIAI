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
#import scipy.io
from vit import *
from Models import *

#from facenet_pytorch import InceptionResnetV1
#device = torch.device("cpu") 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ageDB(torch.utils.data.Dataset):
    def __init__(self, pth):
        self.pth = pth
        self.examples = os.listdir(self.pth)
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224))])
        
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
        img = Image.open(os.path.join(self.pth, pth))
        img = self.transform(img)
        if img.size(0) == 1:
            img = torch.stack((img, img, img))
            img = img.squeeze()

        pth = pth.split(".")[0] + ".pt"
        return img, lbl, pth

ft = 'get-attn'

pth = 'datacel/AgeDB'
valset = ageDB(pth)
total = len(valset)
dataloader = DataLoader(valset, batch_size = 1, shuffle=False)
print("Length of data = ", total)

if ft == 'gen-attack':
    svpth = 'saved_models/classifier/vit-lfw-3-0.0001.pth.tar'
    #svpth = 'saved_models/classifier/vitb-5-0.0001-celeb-gen.pth.tar'
    
    model = torchvision.models.vit_b_16(pretrained=False)
    model.heads.head = torch.nn.Linear(768, 2)
    state = torch.load(svpth)
    try:
        model.load_state_dict(state['state_dict'])
        print("Model Loaded 1")
    except RuntimeError:
        dic = {}
        for k,v in state['state_dict'].items():
            dic[k.replace("module.", "")] = v
        model.load_state_dict(dic)
        print("Model Loaded 2")
    model.to(device)
    print("Model Loaded")

    attacks = [adv.FGSM(model), adv.PGD(model)] #,adv.FFGSM(model),adv.RFGSM(model),adv.MIFGSM(model),adv.BIM(model),adv.UPGD(model),adv.CW(model), adv.PGDL2(model), adv.DeepFool(model)]
    att = ['fgsm','pgd'] #,'FFGSM','FGSM','RFGSM','MIFGSM','BIM','UPGD','CW','PGDL2','DeepFool']
    path1 = 'datacel/agedb'
    for j, attack in enumerate(attacks):
        print("Running...", j)
        #if j in [0]:
        #    continue
        for i, (img, labels, lnk) in enumerate(tqdm(dataloader)):
            #if i <= 289:
            #   continue
            #print(labels)
            img = img.to(device)
            labels = labels.to(device)
            att_img = attack(img, labels)
            #print(path1, att[j], lnk)
            lnk = os.path.join(path1, att[j], lnk[0])
            torch.save(att_img.squeeze(), lnk)

elif ft == 'gen-noise':
    """
    path1 = 'datacel/agedb/gauss'
    mean = 0
    var = 0.0005
    sigma = var**0.5
    for i, (img, labels, lnk) in enumerate(tqdm(dataloader)):
        #im = 'image'+str(i)+'.pt'
        ch, row, col = img.squeeze().size()
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(ch, row, col)
        noisy = img.squeeze() + torch.tensor(gauss)

        #lnk = lnk[0].split(".")
        #lnk = lnk[0] + ".pt"
        lnk = os.path.join(path1, lnk[0])
        torch.save(noisy, lnk)
    """
    path1 = 'datacel/agedb/sp'
    s_vs_p = 0.5
    amount = 0.004
    for i, (img, labels, lnk) in enumerate(tqdm(dataloader)):
        ch, row, col = img.squeeze().size()
        out = np.copy(np.array(img.squeeze().permute(1,2,0)))
        # salt mode
        num_salt = np.ceil(amount * out.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in out.shape]
        out[coords] = 1
        # Pepper mode
        num_pepper = np.ceil(amount* out.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in out.shape]
        out[coords] = 0
        out = torch.tensor(out)

        #lnk = lnk[0].split(".")
        #lnk = lnk[0] + ".pt"
        lnk = os.path.join(path1, lnk[0])
        torch.save(out.permute(2,0,1), lnk)

if ft == 'get-attn':
    dec_attn = []
    def op_decoder_layer(self, input, output):
        #print(len(output))
        #output = output[1]
        dec_attn.append(output[1])
        #print(output[0].size(), output[1].size())

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
            return img, lbl

    
    model = DetViTb() 
    #svpth = 'saved_models/det-vitb-3-0.0001-celeb-gen-mmd-3class.pth.tar'
    svpth = 'saved_models/det-vitb-3-0.0001-celeb-gen-mmd-3class.pth.tar'
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
    #print(model.model.model)
    #model.encoder.layers.encoder_layer_11.self_attention = torch.nn.MultiheadAttention(768,12,need_weights=True)
    hook = model.model.model.encoder.layers.encoder_layer_11.self_attention.register_forward_hook(op_decoder_layer)
    model = model.to(device)

    idd = 80
    idx = 4
     
    gp = [['org', 'att-imgs/imgorg.png'], ['fgsm', 'att-imgs/img1.png'], ['pgd', 'att-imgs/img2.png'], ['gauss', 'att-imgs/img3.png'], ['sp', 'att-imgs/img4.png']]
    pth = 'datacel/AgeDB'
    att = gp[idx][0]
    data = ageDB(pth, att)
    ppth = gp[idx][1]
    fset = data[idd][0]
    
    fset = torch.autograd.Variable(fset.data,requires_grad=True)
    pre = model(fset.unsqueeze(0).to(device))
    #print(dec_attn[0][0].size())
    dec = dec_attn[0][0][0][1:]
    print(dec.size())

    import cv2
    #dec = dec.detach().cpu().numpy().reshape(12,8,8)
    dec = dec.detach().cpu().numpy().reshape(14,14)
    dec = dec/dec.max()
    #dec = dec.mean(axis=0)
    print(dec.shape)
    dec = cv2.resize(dec.squeeze(), (224,224))
    dec = np.uint8(255 * dec)
    dec = cv2.applyColorMap(dec, cv2.COLORMAP_JET)
    dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    dec = dec/255

    dec = torch.tensor(dec).detach()
    #trans = transforms.Resize((64,64))
    #dec = 0.4*dec + 0.6*fset.detach().cpu().permute(1,2,0)
    dec = fset.detach().permute(1,2,0).cpu()
    #plt.imshow(trans(dec.permute(2,0,1)).permute(1,2,0).numpy())
    plt.imshow(dec.numpy())
    plt.axis('off')
    plt.savefig(ppth)


        

