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

from Models import *
from CenterLoss import *
from vit import *

imgs  = os.listdir('att-imgs')
for i, img in enumerate(imgs):
    #if i != 1:
    #    continue
    pth = os.path.join('att-imgs', img)
    img = Image.open(pth).convert('RGB')
    width, height = img.size
    im = np.array(torch.tensor(np.array(img)).permute(2,1,0))
    #print(img.size, im.shape)
    #quit()
    """
    for i in range(3):
        for x in range(width):
            for y in range(height):
                #print(im[i,x,y])
                if im[i,x,y] < 255:
                    print(i,x,y,im[i,x,y])
                    quit()
                else:
                    continue
    """
    left = 143
    top = 58
    right = width-143
    bottom = height-58
    img = img.crop((left, top, right, bottom))
    img.save(pth)

    #img.save('att-imgs/altered.png')
    #quit()
quit()




#from facenet_pytorch import InceptionResnetV1

#device = torch.device("cpu") 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224))
])

class LFW(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.pth = 'lfw-py/lfw_funneled'
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224)) ])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        olnk = self.df['pth_link'][idx]
        pt = olnk.split('_')
        #print(pt)
        pt = '_'.join(pt[:-1])
        lnk = os.path.join(self.pth, pt, olnk)
        #print(lnk)
        lb = self.df['label'][idx]
        img = self.transform(Image.open(lnk))
        #print(img.size())

        return img, torch.tensor(lb), olnk


"""
dataset = LFWPeople(root='./', split='10fold',download=False, transform = transform)
dataloader = DataLoader(dataset, batch_size = 1, shuffle=False)
print(len(dataset))
"""

ft = 'get-attn'

df = pd.read_csv('datacel/lfw/lfw-gen-train.csv')
valdf = pd.read_csv('datacel/lfw/lfw-gen-test.csv')
tdata = LFW(df)
valdata = LFW(valdf)
data = tdata + valdata
total = len(data)
dataloader = DataLoader(tdata, batch_size = 1, shuffle=False)
print("Length of data = ", total)

if ft == 'gen-attack':
    """
    #model = InceptionResnetV1(pretrained = 'vggface2').eval()
    model = InceptionResnetV1(num_classes = 8631, classify=True).eval()
    #print(model)
    #quit()
    model.load_state_dict(torch.load('20180402-114759-vggface2.pt'))
    model.to(device)
    """
    
    svpth = 'saved_models/classifier/vit-lfw-3-0.0001.pth.tar'
    
    model = torchvision.models.vit_b_16(pretrained=False)
    model.heads.head = torch.nn.Linear(768, 2)
    model.load_state_dict(torch.load(svpth)['state_dict'])
    model.to(device)
    print("Model Loaded")

    attacks = [adv.FGSM(model), adv.PGD(model), adv.CW(model)]#, adv.DeepFool(model)] #,adv.FFGSM(model),adv.RFGSM(model),adv.MIFGSM(model),adv.BIM(model),adv.UPGD(model),, adv.PGDL2(model)]
    att = ['fgsm','pgd', 'cw']#,'DeepFool'] #,'FFGSM','FGSM','RFGSM','MIFGSM','BIM','UPGD','CW','PGDL2']
    path1 = 'datacel/lfw'
    for j, attack in enumerate(attacks):
        print("Running...", j)
        if j in [0,1]:
            continue
        for i, (img, labels, lnk) in enumerate(tqdm(dataloader)):
            #if i <= 500:
            #   continue
            path = os.path.join(path1, att[j])
            img = img.to(device)
            labels = labels.to(device)
            #print(img)
            #print(img.size(), labels.size())
            att_img = attack(img, labels)

            lnk = lnk[0].split(".")
            lnk = lnk[0] + ".pt"
            lnk = os.path.join(path1, att[j], lnk)
            #print(lnk)
            #sev = os.path.join(path, f'img{i}.pt')
            torch.save(att_img.squeeze(), lnk)

elif ft == 'gen-noise':
    """
    path1 = 'datacel/lfw/gauss'
    mean = 0
    var = 0.0005
    sigma = var**0.5
    for i, (img, labels, lnk) in enumerate(tqdm(dataloader)):
        #im = 'image'+str(i)+'.pt'
        ch, row, col = img.squeeze().size()
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(ch, row, col)
        noisy = img.squeeze() + torch.tensor(gauss)

        lnk = lnk[0].split(".")
        lnk = lnk[0] + ".pt"
        lnk = os.path.join(path1, lnk)
        torch.save(noisy, lnk)
    """
    path1 = 'datacel/lfw/sp'
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

        lnk = lnk[0].split(".")
        lnk = lnk[0] + ".pt"
        lnk = os.path.join(path1, lnk)
        torch.save(out.permute(2,0,1), lnk)

elif ft == 'train-clss':  
    df = pd.read_csv('datacel/lfw/lfw-gen-train.csv')
    valdf = pd.read_csv('datacel/lfw/lfw-gen-test.csv')
    data = LFW(df)
    valdata = LFW(valdf)
    total = len(data)
    vtotal = len(valdata)
    #print(len(data[0]))
    dataloader = DataLoader(data, batch_size = 128, shuffle=True)
    valloader = DataLoader(valdata, batch_size = 64, shuffle=True)

    nc = 2
    model = torchvision.models.vit_b_16(pretrained=True)
    model.heads.head = torch.nn.Linear(768, nc)
    model.to(device)

    epochs = 3
    lr = 1e-4
    #weight = torch.tensor([0.8,0.2])
    celoss = torch.nn.CrossEntropyLoss()
    #sig = torch.nn.Sigmoid()

    savepth = 'saved_models/classifier/vit-lfw-{}-{}.pth.tar'.format(epochs,lr)
    log = 'logs/vit-lfw-{}-{}.txt'.format(epochs,lr)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for j in range(epochs):
        print("Epoch ====> ", j+1)
        tacc = 0
        tloss = 0
        for i, btch in enumerate(tqdm(dataloader)):
            #print(btch)
            imgs, lbls, _ = btch
            #print(imgs.size())
            imgs = imgs.to(device)
            pred = model(imgs)
            
            optimizer.zero_grad()
            loss = celoss(pred.squeeze(), lbls.to(device))
            loss.backward()
            optimizer.step()

            tloss += loss.item()

            pred = pred.cpu()
            pred = torch.argmax(pred, axis=1)
            tacc += torch.sum(pred==lbls.squeeze())

        # Validation
        vloss = 0
        vacc = 0 
        for i, btch in enumerate(tqdm(valloader)):
            #print(btch)
            imgs, lbls, _ = btch
            #print(imgs.size())
            imgs = imgs.to(device)
            pred = model(imgs)

            loss = celoss(pred.squeeze(), lbls.to(device))
            vloss += loss.item()

            pred = pred.cpu()
            pred = torch.argmax(pred, axis=1)
            vacc += torch.sum(pred==lbls.squeeze())

        print("Epoch {} ==> Training Loss = {}, Training Accuracy = {}, \
              Validation Loss = {}, Validation Accuracy = {}".format(j+1, tloss/len(dataloader), tacc.item()/total, vloss/len(valloader), vacc.item()/vtotal))

        with open(log, 'a') as f:
            f.write(str(j+1) + '\t'
            + str(tloss/len(dataloader)) + '\t'   # T LOSS
            + str(tacc.item()/total) + '\t'        # T ACC
            + str(vloss/len(valloader)) + '\t' # V LOSS
            + str(vacc.item()/vtotal) + '\n'        # V ACC
            )

        state = {
        'epoch': j,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        }
        torch.save(state, savepth)

elif ft == 'train-det':
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
    data1 = LFWDet(df, attack1 = 'gauss', attack2 = 'DeepFool')
    #data2 = LFWDet(df, attack1 = 'pgd', attack2 = 'cw')
    data = data1 #+ data2
    #valdf = pd.read_csv('datacel/lfw/lfw-gen-test.csv')
    #valdata = LFW(valdf)

    nc = 2
    spth = 'saved_models/classifier/vit-lfw-3-0.0001.pth.tar'
    model = ViTb(svpth=spth, nc=2)

    if torch.cuda.device_count() >= 2:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    epochs = 3
    lr = 1e-4
    bs = 64
    #weight = torch.tensor([0.8,0.2])
    mloss = MMDLoss(bs)
    celoss = torch.nn.CrossEntropyLoss()
    #sig = torch.nn.Sigmoid()

    total = len(data)
    dataloader = DataLoader(data, batch_size = bs, shuffle=True)
    #valloader = DataLoader(valdata, batch_size = 64, shuffle=True)

    savepth = 'saved_models/pretrain/vitb-lfw-{}-{}.pth.tar'.format(epochs,lr)
    log = 'logs/vitb-pretrain-lfw-{}-{}.txt'.format(epochs,lr)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for ep in range(epochs):
        tloss = 0
        t1loss, t2loss, t3loss = 0,0,0
        for img, inimg, noimg in tqdm(dataloader):
            img, inimg, noimg = img.to(device), inimg.to(device), noimg.to(device)#, label.to(device)
            
            img, feat = model(img)
            inimg, ifeat = model(inimg)
            noimg, nfeat = model(noimg)
           
            #loss2 = nloss(img, inimg, noimg)
            loss, center = mloss(img, inimg, noimg) 
            floss = loss

            t1loss += floss 
        
            optimizer.zero_grad()
            floss.backward()
            optimizer.step()
            
            tloss += loss
        
        print("Epoch = {}, TLoss = {}".format(ep, tloss/len(dataloader)))
       
        print(t1loss.item(), tloss.item())#, t2loss.item(), t3loss.item())
        state = {
            'epoch' : ep,
            'state_dict' : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }
        torch.save(state, savepth)

elif ft == 'train-det-class':
    class LFWDet(torch.utils.data.Dataset):
        def __init__(self, df, attack):
            self.df = df
            self.pth = 'lfw-py/lfw_funneled'
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224)) ])
            self.attack = attack

        def __len__(self):
            return len(self.df)
        
        def __getitem__(self,idx):
            olnk = self.df['pth_link'][idx]
            alnk = olnk.split(".")[0] + ".pt"

            if self.attack == 'org':
                pt = olnk.split('_')
                pt = '_'.join(pt[:-1])
                olnk = os.path.join('lfw-py/lfw_funneled', pt, olnk)
                img = self.transform(Image.open(olnk))
                lbl = torch.tensor(0)
            
            elif self.attack == 'cw' or self.attack == 'gauss':
                alnk = os.path.join('datacel/lfw', self.attack, alnk)
                img = torch.load(alnk, map_location=torch.device('cpu'))
                lbl = torch.tensor(2)

            elif self.attack == 'DeepFool' or self.attack == 'hdjdk':
                alnk = os.path.join('datacel/lfw', self.attack, alnk)
                img = torch.load(alnk, map_location=torch.device('cpu'))
                lbl = torch.tensor(1)

           
            if self.attack == 'gauss' or self.attack == 'sp':
                img = img.float()
           
            return img, lbl
   
    df = pd.read_csv('datacel/lfw/lfw-gen-train.csv')
    data = LFWDet(df, 'org')
    #fdata = LFWDet(df, 'fgsm')
    cdata = LFWDet(df, 'DeepFool')
    c2data = LFWDet(df, 'gauss')
    #gdata = LFWDet(df, 'pgd')
    data = data + cdata + c2data #+ gdata+ fdata 

    #valdf = pd.read_csv('datacel/lfw/lfw-gen-test.csv')
    #valdata = LFW(valdf)

    pth = 'saved_models/pretrain/vitb-lfw-3-0.0001.pth.tar'
    model = DetViTb(pth)

    if torch.cuda.device_count() >= 2:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    epochs = 3
    lr = 1e-4
    bs = 128
    
    #weights = torch.tensor([1.0, 1.0, 1.5])
    #celoss = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    celoss = torch.nn.CrossEntropyLoss()
    #sig = torch.nn.Sigmoid()

    total = len(data)
    dataloader = DataLoader(data, batch_size = bs, shuffle=True)
    #valloader = DataLoader(valdata, batch_size = 64, shuffle=True)

    savepth = 'saved_models/vitb-det-lfw-{}-{}.pth.tar'.format(epochs,lr)
    log = 'logs/vitb-det-lfw-{}-{}.txt'.format(epochs,lr)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for ep in range(epochs):
        tloss = 0
        tacc = 0
        for img, lbl in tqdm(dataloader):
            img, lbl = img.to(device), lbl.to(device)
            
            pred = model(img)
            
            optimizer.zero_grad()
            loss = celoss(pred.squeeze(), lbl)
            loss.backward()
            optimizer.step()

            tloss += loss.item()

            #pred = pred
            pred = torch.argmax(pred, axis=1)
            #print(pred)
            tacc += torch.sum(pred==lbl.squeeze())

        print("Epoch {} ==> Training Loss = {}, Training Accuracy = {}".format(ep+1, tloss/len(dataloader), tacc.item()/total))
        state = {
        'epoch': ep,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        }
        torch.save(state, savepth)

if ft == 'get-attn':
    dec_attn = []
    def op_decoder_layer(self, input, output):
        #output = output[1]
        dec_attn.append(output[1])
        #print(output[0].size(), output[1].size())
    """
    model = DetViTb() 
    #svpth = 'saved_models/vitb-det-lfw-3-0.0001.pth.tar'
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

    model.to(device)
    #print(model.model.model)
    #model.encoder.layers.encoder_layer_11.self_attention = torch.nn.MultiheadAttention(768,12,need_weights=True)
    hook = model.model.model.encoder.layers.encoder_layer_11.self_attention.register_forward_hook(op_decoder_layer)    
    """
    #svpth = 'saved_models/classifier/vit-lfw-3-0.0001.pth.tar'
    svpth = 'saved_models/classifier/vitb-5-0.0001-celeb-gen.pth.tar'
    model = vit_b_16()
    model.heads.head = torch.nn.Linear(768, 2)
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

    
    model.to(device)
    hook = model.encoder.layers.encoder_layer_11.self_attention.register_forward_hook(op_decoder_layer)
    
    idd = 230
    idx = 4

    class LFWDet(torch.utils.data.Dataset):
        def __init__(self, df, attack):
            self.df = df
            self.pth = 'lfw-py/lfw_funneled'
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224)) ])
            self.attack = attack

        def __len__(self):
            return len(self.df)
        
        def __getitem__(self,idx):
            olnk = self.df['pth_link'][idx]
            alnk = olnk.split(".")[0] + ".pt"

            if self.attack == 'org':
                print("ORG")
                pt = olnk.split('_')
                pt = '_'.join(pt[:-1])
                olnk = os.path.join('lfw-py/lfw_funneled', pt, olnk)
                img = self.transform(Image.open(olnk))
                lbl = torch.tensor(0)
            
            elif self.attack == 'sp' or self.attack == 'gauss':
                alnk = os.path.join('datacel/lfw', self.attack, alnk)
                img = torch.load(alnk, map_location=torch.device('cpu'))
                lbl = torch.tensor(2)

            elif self.attack == 'DeepFool' or self.attack == 'fgsm':
                alnk = os.path.join('datacel/lfw', self.attack, alnk)
                img = torch.load(alnk, map_location=torch.device('cpu'))
                lbl = torch.tensor(1)

           
            if self.attack == 'gauss' or self.attack == 'sp':
                img = img.float()
           
            return img, lbl
   
    pth = 'datacel/celebahq'
    class Celeb(torch.utils.data.Dataset):
        def __init__(self, pth, attack = 'fgsm'):
            self.opth = 'CelebA-HQ/images'
            self.oexamples = os.listdir(self.opth)

            self.pth = os.path.join(pth, attack)
            self.attack = attack
            #self.examples = os.listdir(self.pth)
            #print(self.examples[:10])
            #quit()
            self.transform = transforms.Compose([transforms.Resize((224,224)) ])
            self.ttransform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224)) ])

        def __len__(self):
            return len(self.oexamples)
        
        def __getitem__(self,idx):
            if self.attack == 'org':
                img = Image.open(os.path.join(self.opth, self.oexamples[idx]))
                img = self.ttransform(img)
            else:
                opth = self.oexamples[idx]
                opth = opth.split(".")[0]
                #print(os.path.join(self.pth, self.examples[idx]))
                img = torch.load(os.path.join(self.pth, opth+".pt"))
                img = self.transform(img.squeeze())
                if self.pth == 'datacel/celebahq/gaussian':
                    img = img.float() 
            #print(img.size())
            return img, torch.tensor(0)
    

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

    
    #valdata = Celeb(pth, attack)  
    gp = [['org', 'att-imgs/imgorg.png'], ['fgsm', 'att-imgs/img1.png'], ['pgd', 'att-imgs/img2.png'], ['gaussian', 'att-imgs/img3.png'], ['s&p', 'att-imgs/img4.png']]
    #gp = [['org', 'att-imgs/imgorg.png'], ['fgsm', 'att-imgs/img1.png'], ['DeepFool', 'att-imgs/img2.png'], ['gauss', 'att-imgs/img3.png'], ['sp', 'att-imgs/img4.png']]
    att = gp[idx][0]
    ppth = gp[idx][1]
    
    agepth = 'datacel/AgeDB'
    df = pd.read_csv('datacel/lfw/lfw-gen-train.csv')
    data = LFWDet(df, att)
    cdata = Celeb(pth, att)
    adata = ageDB(agepth, att)
    fset = cdata[idd][0]
    
    fset = torch.autograd.Variable(fset.data,requires_grad=True)
    pre = model(fset.unsqueeze(0).to(device))
    sig = torch.nn.Sigmoid()
    #print(pre)
    print(sig(pre))
    print(torch.round(sig(pre)))
    print("prediction = ", torch.argmax(sig(pre), axis=1))
    #print(dec_attn[0][0].size())
    dec = dec_attn[0][0][0][1:]
    #print(dec.size())

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
    dec = 0.4*dec.cpu() + 0.6*fset.detach().permute(1,2,0).cpu()
    #dec = fset.detach().permute(1,2,0).cpu()
    #plt.imshow(trans(dec.permute(2,0,1)).permute(1,2,0).numpy())
    plt.imshow(dec.numpy())
    plt.axis('off')
    plt.savefig(ppth)


        