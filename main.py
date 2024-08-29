#60211202048
import os
import torch
import torchvision
import torchvision.transforms as transforms
import requests
from zipfile import ZipFile
import random
import numpy as np 
import pandas as pd
#from sklearn.manifold import TSNE
#import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

from Models import *
from CenterLoss import *
from dataloader import *
from train import *
from vit import *

import h5py
#import torchattacks
from tqdm import tqdm

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from dataloader import CIFAR100DataLoader

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

"""
Datasets: 'euro', 'cifar', 'celeb', 'celeb-gen'

Classifier:
    train_class
    test_class

Pretraining:
    pretrain
    pretrain2
    3pre
    pre+det
    preall

Detector:
    detector
    testdet
"""
dset = 'celeb-gen'
ctype = 'check'
mt = 'vitb' #['r18', 'r50', 'vitb', 'cnextb']
"""
ttransforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224,224))])
pth1 = 'datacel/celeba/list_attr_celeba.csv'
#pth3 = 'datacel/celeba/list_eval_partition.csv' #"0" represents training image, "1" represents validation image, "2" represents testing image
split = 'train' #['train', 'val', 'test']

idd = 5
df = pd.read_csv(pth1)
imgpth = 'datacel/celeba/img_align_celeba'

data = CelebDataset(df, 'all', imgpth, transform=ttransforms)
plt.imshow(data[idd][0].permute(1,2,0))
plt.savefig('imgorg.png')

data = Image.open("datacel/processed_images/000006.jpg")
plt.imshow(data)
plt.savefig('img1.png')
quit()


im1 = 'datacel/celeb-vitb-fgsm-1.hdf5'
fs = h5py.File(im1, 'r')
fset = fs['data'][idd].reshape(3,224,224)
im2 = 'datacel/celeb-vitb-pgd-1.hdf5'
fs = h5py.File(im2, 'r')
fset2 = fs['data'][idd].reshape(3,224,224)

plt.imshow(torch.tensor(fset).permute(1,2,0))
plt.savefig('img1.png')

plt.imshow(torch.tensor(fset2).permute(1,2,0))
plt.savefig('img2.png')
pth1 = 'datacel/celeba/list_attr_celeba.csv'
df = pd.read_csv(pth1)
id = df['image_id'].iloc[idd]
id = id.split(".")[0]
p1 = 'datacel/imgs/gauss/'+id +'.pt'
img = torch.load(p1, map_location=torch.device('cpu'))
plt.imshow(img.detach().permute(1,2,0))
plt.savefig('img3.png')
p2 = 'datacel/imgs/s&p/'+id +'.pt'
img = torch.load(p2, map_location=torch.device('cpu'))           
plt.imshow(img.detach().permute(1,2,0))
plt.savefig('img4.png')
quit()
"""

def tiny_imagenet_dataloader(data, name, transform,subset=None):
    if data is None: 
        return None
    if transform is None:
        dataset_tiny = datasets.ImageFolder(data, transform=transforms.ToTensor())
    else:
        dataset_tiny = datasets.ImageFolder(data, transform=transform)
        if subset==True:
            data_index, _ = torch.utils.data.random_split(dataset_tiny, (1000,9000))
            dataloader = torch.utils.data.DataLoader(dataset_tiny, batch_size=16, shuffle=(name=="train"),sampler=torch.utils.data.SubsetRandomSampler(data_index.indices))
        else:
            dataloader = torch.utils.data.DataLoader(dataset_tiny, batch_size=16, shuffle=(name=="train"))
    return dataloader

if dset == 'cifar':
    ttransforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224))
            ])
    data = torchvision.datasets.CIFAR10(root = "cifar/", train = True, download = False, transform = ttransforms)
    testdata = torchvision.datasets.CIFAR10(root = "cifar/", train = False, download = False, transform = ttransforms)
    nc = 10

    data_dir = 'cifar/train'
    mot = 'vitb'
    tdata = ci3data(data_dir, testdata, mod=mot)  
    model = ViTb(False, pretrained = False) 
    
    svpth = 'saved_models/pretrain/vitb-3-0.0001-cifar-mmd-all.pth.tar'
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
    
    tlen = len(tdata)
    bs = 1
    train_dataloader = DataLoader(tdata, batch_size=bs, num_workers = 4, shuffle=True)
    
    import sklearn
    sc = 0
    i = 0
    for img, inimg, noimg, inimg1, noimg1 in tqdm(train_dataloader):
        emb,_ = model(img.to(device))
        emb1,_ = model(inimg1.to(device))
        emb = emb.cpu().detach()#.squeeze()
        emb1 = emb1.cpu().detach()#.squeeze()
        score = sklearn.metrics.pairwise.cosine_similarity(emb, emb1)
        sc += score[0][0]
        i+=1
        if i == 1000:
            print(sc/1000)
            break
    print(sc/1000)
#-0.015337843293324114
#no intent - 0.9999438462853432

## CIAR
# 0.42384985158219934
#no intent - 0.46881058944389226

elif dset == 'celeb':
    ttransforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224,224))])
    pth1 = 'datacel/celeba/list_attr_celeba.csv'
    #pth3 = 'datacel/celeba/list_eval_partition.csv' #"0" represents training image, "1" represents validation image, "2" represents testing image
    split = 'train' #['train', 'val', 'test']
    
    df = pd.read_csv(pth1)
    imgpth = 'datacel/celeba/img_align_celeba'

    data = CelebDataset(df, 'all', imgpth, ttransforms)
    traindata = CelebDataset(df, 'train', imgpth, ttransforms)
    valdata = CelebDataset(df, 'val', imgpth, ttransforms)
    testdata = CelebDataset(df, 'test', imgpth, ttransforms)
    nc = 40

elif dset == 'celeb-gen':
    ttransforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224,224))])
    pth1 = 'datacel/celeba/list_attr_celeba.csv'
    #pth3 = 'datacel/celeba/list_eval_partition.csv' #"0" represents training image, "1" represents validation image, "2" represents testing image
    split = 'train' #['train', 'val', 'test']
    
    df = pd.read_csv(pth1)
    imgpth = 'datacel/celeba/img_align_celeba'
    
    data_fl = pd.read_csv("CelebA-HQ.txt", header=None)
    idxs = []
    #print(data_fl)
    for idx in range(len(data_fl)):
        ln = data_fl[0][idx]
        pt = int(ln.split(".")[0].split("/")[1])
        idxs.append(pt)

    midxs = []    
    for i in range(202599):
        midxs.append(i)

    ridxs = [x for x in midxs if x not in idxs]
    print(len(midxs), len(ridxs), len(idxs))
    #print(df[df.index.isin(ridxs)])
    
    data = CelebDataset(df, 'all', imgpth, ttransforms)
    traindata = CelebDataset(df, 'train', imgpth, ttransforms, ridxs, idxs, midxs)
    valdata = CelebDataset(df, 'val', imgpth, ttransforms, ridxs, idxs, midxs)
    testdata = CelebDataset(df, 'test', imgpth, ttransforms, trainids = ridxs, valids = idxs, allids = midxs)
    #testdata = CelebDataset(df, 'all', imgpth, mod = None, transform = ttransforms, trainids = ridxs, valids = idxs)
    #print("LEN = ", len(testdata))
    nc = 2

    """
    import cv2
    sc = cv2.quality.QualityBRISQUE.compute( img )
    print(sc)
    """
    
    pth1 = 'datacel/celeba/list_attr_celeba.csv'
    df = pd.read_csv(pth1)
    nc=2
    data_dir = 'datacel'
    mot = 'vitb'
    tdata = ce3data(data_dir, traindata, df, mod=mot)   
    model = ViTb(False, pretrained = False) 
    
    svpth = 'saved_models/pretrain/vitb-3-0.0001-celeb-gen-mmd-2-all.pth.tar'
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
    
    tlen = len(tdata)
    bs = 1
    train_dataloader = DataLoader(tdata, batch_size=bs, num_workers = 4, shuffle=True)
    
    import sklearn
    sc = 0
    for img, inimg, noimg, inimg1, noimg1 in tqdm(train_dataloader):
        emb,_ = model(img.to(device))
        emb1,_ = model(noimg.to(device))
        emb = emb.cpu().detach()#.squeeze()
        emb1 = emb1.cpu().detach()#.squeeze()
        score = sklearn.metrics.pairwise.cosine_similarity(emb, emb1)
        sc += score[0][0]
    print(sc/1000)
#-0.015337843293324114
#no intent - 0.9999438462853432

elif dset == 'cifar100':
    ttransforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224))
            ])
    data = torchvision.datasets.CIFAR100(root = "cifar100/", train = True, download = False, transform = ttransforms)
    testdata = torchvision.datasets.CIFAR100(root = "cifar100/", train = False, download = False, transform = ttransforms)
    nc = 10

    data_dir = 'cifar100/train'
    mot = 'vitb'
    tdata = ci3data(data_dir, testdata, mod=mot)  
    model = ViTb(False, pretrained = False) 
    
    svpth = 'saved_models/pretrain/vitb-3-0.0001-cifar-mmd-all.pth.tar'
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
    
    tlen = len(tdata)
    bs = 1
    train_dataloader = DataLoader(tdata, batch_size=bs, num_workers = 4, shuffle=True)
    
    import sklearn
    sc = 0
    i = 0
    for img, inimg, noimg, inimg1, noimg1 in tqdm(train_dataloader):
        emb,_ = model(img.to(device))
        emb1,_ = model(inimg1.to(device))
        emb = emb.cpu().detach()#.squeeze()
        emb1 = emb1.cpu().detach()#.squeeze()
        score = sklearn.metrics.pairwise.cosine_similarity(emb, emb1)
        sc += score[0][0]
        i+=1
        if i == 1000:
            print(sc/1000)
            break
    print(sc/1000)

elif dset == 'euro':
    image_path = '2750'
    ttransforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224,224))])
    trainlst, vallst, testlst = [], [], []
    for id in os.listdir(image_path):
        pth = os.path.join(image_path, id, id)
        tlen = len(os.listdir(pth))
        val = int(0.2*tlen)//2
        tst = val
        lst = os.listdir(pth)

        #print(tlen, val, tst)

        trainlst += lst[:tlen-(tst+val)]
        vallst += lst[tlen-(tst+val):tlen-tst]
        testlst += lst[tlen-tst:]

    #print(len(trainlst), len(vallst), len(testlst))
    dirpth = '2750'
    traindata = EuroDataset(dirpth, trainlst, ttransforms)
    valdata = EuroDataset(dirpth, vallst, ttransforms)
    testdata = EuroDataset(dirpth, testlst, ttransforms)
    print(len(traindata), len(valdata), len(testdata))
    nc = 10


elif dset=='Tiny_Imagenet_200':
        r = requests.get('http://cs231n.stanford.edu/tiny-imagenet-200.zip', allow_redirects=True)
        zip = ZipFile('tiny-imagenet-200.zip')
        zip.extractall()
        DATA_DIR = '/content/tiny-imagenet-200' # Original images come in shapes of [3,64,64]
        TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
        VALID_DIR = os.path.join(DATA_DIR, 'val')
        val_img_dir = os.path.join(VALID_DIR, 'images')
        fp = open(os.path.join(VALID_DIR, 'val_annotations.txt'), 'r')
        data = fp.readlines()
        val_img_dict = {}
        for line in data:
            words = line.split('\t')
            val_img_dict[words[0]] = words[1]
        fp.close()
        for img, folder in val_img_dict.items():
            newpath = (os.path.join(val_img_dir, folder))
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            if os.path.exists(os.path.join(val_img_dir, img)):
                os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))
        train_tiny = tiny_imagenet_dataloader(TRAIN_DIR, "train",transform=transform_fn, subset = subset)
        test_tiny = tiny_imagenet_dataloader(val_img_dir, "val",transform=transform_fn, subset = subset)

if ctype == 'get_attn':
    dec_attn = []
    def op_decoder_layer(self, input, output):
        print(len(output))
        #output = output[1]
        dec_attn.append(output[1])
        print(output[0].size(), output[1].size())

    #spth = 'saved_models/classifier/vitb-5-0.0001.pth.tar'
    #spth = 'saved_models/classifier/vitb-5-0.0001-celeb-gen.pth.tar'
    #model = ViTb(svpth=spth, nc=2)
    #svpth = 'saved_models/pretrain/vitb-3-0.0001-celeb-gen-mmd-2-all.pth.tar'


    model = vit_b_16() #pretrained=False)
    model.heads.head = nn.Linear(768, 10)
    #svpth = 'saved_models/classifier/vitb-5-0.0001-celeb-gen.pth.tar'
    svpth = 'saved_models/classifier/vitb-5-0.0001.pth.tar'
    state = torch.load(svpth)
    """
    spth = 'saved_models/classifier/vitb-5-0.0001.pth.tar'
    #spth = 'saved_models/classifier/vitb-5-0.0001-celeb-gen.pth.tar'
    model = ViTb(svpth=spth, nc=10)
    svpth = 'saved_models/pretrain/vitb-3-0.0001-cifar-mmd-2-all.pth.tar'
    """

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
    #model.encoder.layers.encoder_layer_11.self_attention = torch.nn.MultiheadAttention(768,12,need_weights=True)
    hook = model.encoder.layers.encoder_layer_11.self_attention.register_forward_hook(op_decoder_layer)

    idd = 46

    ttransforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224,224))
                ])
    data = torchvision.datasets.CIFAR10(root = "cifar/", train = True, download = False, transform = ttransforms)
    fset = data[idd][0]

    #pth = 'cifar/vitb/train/pgdl2/image' + str(idd) + '.pt'
    #fset = torch.load(pth).cpu()
    #fset = ttransforms(fset)
    #fset = fset.float()

    """
    pth1 = 'datacel/celeba/list_attr_celeba.csv'
    df = pd.read_csv(pth1)
    imgpth = 'datacel/celeba/img_align_celeba'
    data = CelebDataset(df, 'all', imgpth, transform=ttransforms)
    #plt.imshow(data[idd][0].permute(1,2,0))
    fset = data[idd][0]

    im1 = 'datacel/celeb-vitb-pgd-1.hdf5'
    fs = h5py.File(im1, 'r')
    fset = fs['data'][idd].reshape(3,224,224)
    fset = torch.tensor(fset)

    pth1 = 'datacel/celeba/list_attr_celeba.csv'
    df = pd.read_csv(pth1)
    id = df['image_id'].iloc[idd]
    id = id.split(".")[0]
    p1 = 'datacel/imgs/gauss/'+id +'.pt'
    fset = torch.load(p1, map_location=torch.device('cpu'))
    fset=fset.float()
    """

    fset = torch.autograd.Variable(fset.data,requires_grad=True)
    pre = model(fset.unsqueeze(0))
    print(dec_attn[0][0].size())
    dec = dec_attn[0][0][0][1:]
    #print(dec)

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
    dec = 0.4*dec + 0.6*fset.detach().permute(1,2,0)
    #plt.imshow(trans(dec.permute(2,0,1)).permute(1,2,0).numpy())
    plt.imshow(dec.numpy())
    plt.axis('off')
    plt.savefig('img1.png')

if ctype == 'train_class':
    if dset =='euro':
        loss = nn.CrossEntropyLoss()
    elif dset!='celeb':
        split = int(0.1 * len(data))
        ids = [i for i in range(len(data))]
        vids = ids[:split]
        tids = ids[split:]
        traindata = torch.utils.data.Subset(data, tids)
        valdata = torch.utils.data.Subset(data, vids)
        loss = nn.CrossEntropyLoss()
    else:
        loss = nn.BCEWithLogitsLoss()
        
    epochs = 5
    lr = 1e-5

    if mt == 'r50':
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, nc)  
    elif mt == 'r18':
        model = torchvision.models.resnet18(pretrained=True)#weights=ResNet18_Weights.IMAGENET1K_V1) #MCNN()
        model.fc = nn.Linear(512, nc)
    elif mt == 'vitb':
        model = torchvision.models.vit_b_16(pretrained=True)
        model.heads.head = nn.Linear(768, nc)
        epochs = 5
    elif mt == 'cnextb':
        model = torchvision.models.convnext_base(pretrained=True)
        model.classifier[2] = nn.Linear(1024, nc)
        epochs=3
    
    if dset == 'cifar':
        savepth = 'saved_models/classifier/{}-{}-{}.pth.tar'.format(mt,epochs,lr)
        log = 'logs/{}-{}-{}.txt'.format(mt,epochs,lr)
    else:
        savepth = 'saved_models/classifier/{}-{}-{}-{}.pth.tar'.format(mt,epochs,lr,dset)
        log = 'logs/{}-{}-{}-{}.txt'.format(mt,epochs,lr,dset)
    
    tdata = len(traindata)
    vdata = len(valdata)

    if torch.cuda.device_count() >= 2:
        bs = 128
        trainloader = torch.utils.data.DataLoader(traindata, batch_size=bs, num_workers = 4, shuffle=True)
        valloader = torch.utils.data.DataLoader(valdata, batch_size=bs, num_workers = 4, shuffle=True)
        testloader = torch.utils.data.DataLoader(testdata, batch_size=bs, num_workers = 4, shuffle=True)
    else:
        bs = 32
        trainloader = torch.utils.data.DataLoader(traindata, batch_size=bs, shuffle=True)
        valloader = torch.utils.data.DataLoader(valdata, batch_size=bs, shuffle=True)
        testloader = torch.utils.data.DataLoader(testdata, batch_size=bs, shuffle=True)
    print("Length of training, validation, and testing data = ", len(traindata), len(valdata), len(testdata))

    if torch.cuda.device_count() >= 2:
        model = nn.DataParallel(model)
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    train(epochs, trainloader, valloader, model, optimizer, loss, savepth, device, log, tdata, vdata, dset=dset)

if ctype == 'dec': 
    if dset == 'cifar':
        #data_dir = 'cifar/train'
        data_dir = 'cifar/test'#.format(mt)
        pair = ['fgsm', 'gauss', 'pgd', 's&p']
        modt = mt #'r50' #'vitb'
        cleandata = detdata(data_dir, testdata, '', 'clean', mt=modt)
        fdata = detdata(data_dir, testdata, 'fgsm', mt=modt)
        pdata = detdata(data_dir, testdata, 'pgd', mt=modt)
        gdata = detdata(data_dir, testdata, 'gauss', mt=mt)
        sdata = detdata(data_dir, testdata, 's&p', mt=mt)
        cwdata = detdata(data_dir, testdata, 'poisson', mt=modt)
        #print(len(cwdata), cwdata[1])
        totdata = cleandata + fdata + pdata + gdata + sdata
        #totdata = cwdata
    testloader = DataLoader(totdata, batch_size=64, num_workers = 2, shuffle=True)

    if mt == 'r50':
        spth = 'saved_models/pretrain/r50-5-0.0001.pth.tar'
        model = Resnet50(spth)
    elif mt == 'r18':
        model = Resnet18()
    elif mt == 'vitb':
        spth = 'saved_models/pretrain/vitb-3-0.0001-mmd-all.pth.tar'
        model = ViTb(False)

    state = torch.load(spth)
    centers = state['center']
    print(centers.size())

    try:
        model.load_state_dict(state['state_dict'])
        print("Model1 Loaded")
    except RuntimeError:
        dic = {}
        for k,v in state['state_dict'].items():
            dic[k.replace("module.", "")] = v
        model.load_state_dict(dic)
        print("Model2 Loaded")

    model = model.to(device)
    centers = centers.to(device)

    tacc = 0
    pr = []
    tpr = []
    lendata = len(totdata)
    lloss = nn.L1Loss()

    for imgs, labels in tqdm(testloader):
        #print(labels)
        imgs = imgs.to(device)
        pred,_ = model(imgs)
        #pred = pred
    
        lab = []
        for bs in range(imgs.size(0)):
            lb = 0
            mindist = 10000000000
            for i in range(5):
                #print(pred[bs].size(), centers[i][0].size())
                #dist = MMD(pred[bs].unsqueeze(0), centers[i][0].unsqueeze(0))
                dist = lloss(pred[bs], centers[i][0])
                #print(dist)
                if dist<mindist:
                    mindist=dist
                    lb=i    
            lab.append(lb)

        lab = torch.tensor(lab)
        tacc += torch.sum(lab==labels.squeeze())
        if len(pr) == 0:
            pr = lab.unsqueeze(1)
            tpr = labels.cpu()
        else:
            pr = torch.vstack((pr, lab.unsqueeze(1)))
            tpr = torch.vstack((tpr, labels.cpu()))

    print("Testing Accuracy = ", tacc.item()/lendata)
    print(confusion_matrix(tpr, pr)) 

if ctype == 'test_class': 
    epochs = 5
    if mt == 'r50':
        model = torchvision.models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, nc)  
    elif mt == 'r18':
        model = torchvision.models.resnet18(pretrained=False)#weights=ResNet18_Weights.IMAGENET1K_V1) #MCNN()
        model.fc = nn.Linear(512, nc)
    elif mt == 'vitb':
        model = torchvision.models.vit_b_16(pretrained=False)
        model.heads.head = nn.Linear(768, nc)
        print("VIT IT IS")
        epochs = 5

    lr = 1e-4
    bs = 64
    if dset == 'cifar':
        savepth = 'saved_models/classifier/{}-{}-{}.pth.tar'.format(mt,epochs,lr)
    else:
        savepth = 'saved_models/classifier/{}-{}-{}-{}.pth.tar'.format(mt, epochs, lr, dset)
    data_dir = '{}/test/'.format(dset)
    
    
    state = torch.load(savepth)
    try:
        model.load_state_dict(state['state_dict'])
        print("Model Loaded 1")
    except RuntimeError:
        dic = {}
        for k,v in state['state_dict'].items():
            dic[k.replace("module.", "")] = v
        model.load_state_dict(dic)
        print("Model Loaded 2")
    model = model.to(device)

    mdd = 'clean'
    testloader = torch.utils.data.DataLoader(testdata, batch_size=bs, shuffle=True)
    test(model, testloader, testdata, device,mdd,dset=dset,cl='na')

    """
    ttransforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224,224))])
    pth1 = 'datacel/celeba/list_attr_celeba.csv' 
    df = pd.read_csv(pth1)
    imgpth = 'datacel/celeba/img_align_celeba'

    mdd = 'gauss'    
    print(mdd)
    cwdata = CelebDataset(df, 'test', imgpth, mdd, ttransforms)
    testloader = torch.utils.data.DataLoader(cwdata, batch_size=bs, shuffle=True)
    test(model, testloader, testdata, device, mdd, dset=dset, cl=None)

    mdd = 's&p'    
    print(mdd)
    cwdata = CelebDataset(df, 'test', imgpth, mdd, ttransforms)
    testloader = torch.utils.data.DataLoader(cwdata, batch_size=bs, shuffle=True)
    test(model, testloader, testdata, device, mdd, dset=dset, cl=None)
    """

    """
    #mt = 'r50'
    mdd = 'clean'
    data_dir = None
    print(mdd)
    cwdata = classdata(data_dir, testdata, mdd, 'clean', mt=mt, dset='celeb-gen')
    testloader = torch.utils.data.DataLoader(cwdata, batch_size=bs, shuffle=True)
    test(model, testloader, testdata, device, mdd, dset=dset, cl=None)
    
    mdd = 'pgd'
    data_dir = 'datacel/celeb-vitb-pgd-2.hdf5'
    print(mdd)
    cwdata = classdata(data_dir, testdata, mdd, mt=mt, dset='celeb')
    testloader = torch.utils.data.DataLoader(cwdata, batch_size=bs, shuffle=False)
    test(model, testloader, testdata, device, mdd, dset=dset, cl=None)
    
    mdd = 'fgsm'
    data_dir = 'datacel/celeb-vitb-fgsm-2.hdf5'
    print(mdd)
    cwdata = classdata(data_dir, testdata, mdd, mt=mt, dset='celeb')
    testloader = torch.utils.data.DataLoader(cwdata, batch_size=bs, shuffle=True)
    test(model, testloader, testdata, device, mdd, dset=dset, cl=None)
    
    mdd = 'fastfgsm'
    data_dir = 'datacel/celeb-vitb-fastfgsm.hdf5'
    print(mdd)
    cwdata = classdata(data_dir, testdata, mdd, mt=mt, dset='celeb')
    testloader = torch.utils.data.DataLoader(cwdata, batch_size=bs, shuffle=True)
    test(model, testloader, testdata, device, mdd, dset=dset, cl=None)

    mdd = 'rfgsm'
    data_dir = 'datacel/celeb-vitb-rfgsm.hdf5'
    print(mdd)
    cwdata = classdata(data_dir, testdata, mdd, mt=mt, dset='celeb')
    testloader = torch.utils.data.DataLoader(cwdata, batch_size=bs, shuffle=True)
    test(model, testloader, testdata, device, mdd, dset=dset, cl=None)

    mdd = 'bim'
    data_dir = 'datacel/celeb-vitb-bim.hdf5'
    print(mdd)
    cwdata = classdata(data_dir, testdata, mdd, mt=mt, dset='celeb')
    testloader = torch.utils.data.DataLoader(cwdata, batch_size=bs, shuffle=True)
    test(model, testloader, testdata, device, mdd, dset=dset, cl=None)
    """
    
    """
    cl='3cl'
    mdd = 'clean'
    print(mdd)
    cwdata = classdata(data_dir, testdata, mdd, 'clean', mt=mt, dset=dset)
    testloader = torch.utils.data.DataLoader(cwdata, batch_size=bs, shuffle=True)
    test(model, testloader, testdata, device, mdd, dset=dset, cl=None)
    
    
    mdd = 'fgsm'
    print(mdd)
    cwdata = classdata(data_dir, testdata, mdd, mt=mt, dset=dset)
    testloader = torch.utils.data.DataLoader(cwdata, batch_size=bs, shuffle=True)
    test(model, testloader, testdata, device,mdd,dset=dset,cl=cl)

    mdd = 'pgd'
    print(mdd)
    cwdata = classdata(data_dir, testdata, mdd, mt=mt, dset=dset)
    testloader = torch.utils.data.DataLoader(cwdata, batch_size=bs, shuffle=True)
    test(model, testloader, testdata, device,mdd,dset=dset,cl=cl)

    mdd = 'fastfgsm'
    print(mdd)
    cwdata = classdata(data_dir, testdata, mdd, mt=mt, dset=dset)
    testloader = torch.utils.data.DataLoader(cwdata, batch_size=bs, shuffle=True)
    test(model, testloader, testdata, device,mdd,dset=dset,cl=cl)

    mdd = 'rfgsm'
    print(mdd)
    cwdata = classdata(data_dir, testdata, mdd, mt=mt, dset=dset)
    testloader = torch.utils.data.DataLoader(cwdata, batch_size=bs, shuffle=True)
    test(model, testloader, testdata, device,mdd,dset=dset,cl=cl)

    mdd = 'mifgsm'
    print(mdd)
    cwdata = classdata(data_dir, testdata, mdd, mt=mt, dset=dset)
    testloader = torch.utils.data.DataLoader(cwdata, batch_size=bs, shuffle=True)
    test(model, testloader, testdata, device,mdd,dset=dset,cl=cl)

    mdd = 'bim'
    print(mdd)
    cwdata = classdata(data_dir, testdata, mdd, mt=mt, dset=dset)
    testloader = torch.utils.data.DataLoader(cwdata, batch_size=bs, shuffle=True)
    test(model, testloader, testdata, device,mdd,dset=dset,cl=cl)

    mdd = 'upgd'
    print(mdd)
    cwdata = classdata(data_dir, testdata, mdd, mt=mt, dset=dset)
    testloader = torch.utils.data.DataLoader(cwdata, batch_size=bs, shuffle=True)
    test(model, testloader, testdata, device,mdd,dset=dset,cl=cl)

    mdd = 'gauss'
    print(mdd)
    cwdata = classdata(data_dir, testdata, mdd, mt=mt, dset=dset)
    testloader = torch.utils.data.DataLoader(cwdata, batch_size=bs, shuffle=True)
    test(model, testloader, testdata, device,mdd,dset=dset,cl=cl)

    mdd = 's&p'
    print(mdd)
    cwdata = classdata(data_dir, testdata, mdd, mt=mt, dset=dset)
    testloader = torch.utils.data.DataLoader(cwdata, batch_size=bs, shuffle=True)
    test(model, testloader, testdata, device,mdd,dset=dset,cl=cl)
    
    mdd = 'cw'
    print(mdd)
    cwdata = classdata(data_dir, testdata, mdd, mt=mt, dset=dset)
    testloader = torch.utils.data.DataLoader(cwdata, batch_size=bs, shuffle=True)
    test(model, testloader, testdata, device,mdd,dset=dset,cl=cl)

    mdd = 'pgdl2'
    print(mdd)
    cwdata = classdata(data_dir, testdata, mdd, mt=mt, dset=dset)
    testloader = torch.utils.data.DataLoader(cwdata, batch_size=bs, shuffle=True)
    test(model, testloader, testdata, device,mdd,dset=dset,cl=cl)
    
    mdd = 'one'
    print(mdd)
    cwdata = classdata(data_dir, testdata, mdd, mt=mt, dset=dset)
    testloader = torch.utils.data.DataLoader(cwdata, batch_size=bs, shuffle=True)
    test(model, testloader, testdata, device,mdd,dset=dset,cl=cl)

    mdd = 'deepfool'
    print(mdd)
    cwdata = classdata(data_dir, testdata, mdd, mt=mt, dset=dset)
    testloader = torch.utils.data.DataLoader(cwdata, batch_size=bs, shuffle=True)
    test(model, testloader, testdata, device,mdd,dset=dset,cl=cl)
    
    mdd = 'fab'
    print(mdd)
    cwdata = classdata(data_dir, testdata, mdd, mt=mt, dset=dset)
    testloader = torch.utils.data.DataLoader(cwdata, batch_size=bs, shuffle=True)
    test(model, testloader, testdata, device,mdd,dset=dset,cl=cl)
    """

if ctype == 'gen_attack':
    lr = 1e-5
    epochs = 5
    if mt == 'r50':
        model = torchvision.models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, nc)  
    elif mt == 'r18':
        model = torchvision.models.resnet18(pretrained=False)#weights=ResNet18_Weights.IMAGENET1K_V1) #MCNN()
        model.fc = nn.Linear(512, nc)
    elif mt == 'vitb':
        model = torchvision.models.vit_b_16(pretrained=False)
        model.heads.head = nn.Linear(768, nc)
        print("VIT it is!")
        epochs = 5
    
    if dset == 'cifar':
        savepth = 'saved_models/classifier/{}-{}-{}.pth.tar'.format(mt, epochs, lr)
    else:
        savepth = 'saved_models/classifier/{}-{}-{}-{}.pth.tar'.format(mt, epochs, lr, dset)
    
    #savepth = 'saved_models/classifier/resnet18-ep10-lr1e-4.pth.tar'
    state = torch.load(savepth)
    try:
        model.load_state_dict(state['state_dict'])
        print("Model Loaded 1")
    except RuntimeError:
        dic = {}
        for k,v in state['state_dict'].items():
            dic[k.replace("module.", "")] = v
        model.load_state_dict(dic)
        print("Model Loaded 2")
    model = model.to(device)
    bs = 1
    noise = 's&p' #['fgsm', 'pgd', 'gauss', 's&p', 'deepfool']
    sdir = 'euro/vitb/'
    #sdir = 'datacel' #/{}'.format(mt)
    print(noise)
    lst = []

    if noise == 'rfgsm':
        svdir = sdir + '/imgs/{}'.format(noise)
        loader = torch.utils.data.DataLoader(testdata, batch_size=bs, shuffle=False)
        i = 0
        atck = torchattacks.RFGSM(model, eps=8/255, alpha=2/255, steps=10)
        for img, lb in loader:
            if (i+1)%500 == 0:
                print('Done with {}/{} images'.format(i+1, len(loader)))
            
            #im = 'image'+str(i)+'.pt'
            ch, row, col = img.squeeze().size()
            #im = iname[0].split(".")[0]
            #pth = os.path.join(svdir, im+'.pt')
            
            adv_images = atck(img, lb)
            #print(adv_images.squeeze().to('cpu').flatten().size())
            lst.append(adv_images.squeeze().to('cpu').flatten())
            #torch.save(adv_images.squeeze(), pth)
            i+=1
        #print(lst)
        """
        svdir = sdir + '/train/{}'.format(noise)
        loader = torch.utils.data.DataLoader(data, batch_size=bs, shuffle=False)
        #attack = torchattacks.PGDL2(model, eps=1.0, alpha=0.2, steps=10, random_start=True)
        #attack = torchattacks.CW(model, c=1, kappa=0, steps=50, lr=0.01) #torchattacks.FAB(model, norm='L2', steps=10, eps=8/255, n_restarts=1, alpha_max=0.1, eta=1.05, beta=0.9, verbose=False, seed=0, n_classes=10)
        #torchattacks.OnePixel(model, pixels=1, steps=10, popsize=10, inf_batch=128)
        attack = torchattacks.FGSM(model, eps=8/255)
        i = 0
        for img, lb in loader:
            if (i+1)%500 == 0:
                print('Done with {}/{} images'.format(i+1, len(loader)))
            #print(i)
            im = 'image'+str(i)+'.pt'
            pth = os.path.join(svdir,im)
            adv_images = attack(img, lb)
            torch.save(adv_images.squeeze(), pth)
            i+=1
        
        svdir = sdir + '/test/{}'.format(noise)
        loader = torch.utils.data.DataLoader(testdata, batch_size=bs, shuffle=False)
        #attack = torchattacks.PGDL2(model, eps=1.0, alpha=0.2, steps=10, random_start=True)
        #attack = torchattacks.CW(model, c=1, kappa=0, steps=50, lr=0.01)
        #attack = torchattacks.FAB(model, norm='L2', steps=10, eps=8/255, n_restarts=1, alpha_max=0.1, eta=1.05, beta=0.9, verbose=False, seed=0, n_classes=10)
        attack = torchattacks.FGSM(model, eps=8/255)
        i = 0
        for img, lb in loader:
            if (i+1)%500 == 0:
                print('Done with {}/{} images'.format(i+1, len(loader)))
            im = 'image'+str(i)+'.pt'
            pth = os.path.join(svdir,im)
            adv_images = attack(img, lb)
            torch.save(adv_images.squeeze(), pth)
            i+=1
        """  
        df = torch.stack(lst).numpy()
        print(df.shape)
        with h5py.File('celeb-vitb-rfgsm.hdf5', 'w') as f:
            f.create_dataset('data', data = df, dtype='float32')
        print("DONE")

        fs = h5py.File('celeb-vitb-rfgsm.hdf5', 'r')
        dset = fs['data']
        print(dset.shape)
        
    elif noise == 's&p':
        label = {'AnnualCrop':0, 'Forest':1, 'HerbaceousVegetation':2,
                      'Highway':3, 'Industrial':4, 'Pasture':5,
                      'PermanentCrop':6, 'Residential':7, 'River':8, 'SeaLake':9}
        

        svdir = sdir+'train/'+noise
        #loader = torch.utils.data.DataLoader(data, batch_size=bs, shuffle=False)
        #attack = torchattacks.CW(model, c=1, kappa=0, steps=50, lr=0.01)
        attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        
        i = 0
        """
        for pt in trainlst:
            if (i+1)%500 == 0:
                print('Done with {}/{} images'.format(i+1, len(trainlst)))
            dr = pt.split("_")[0]
            pth = os.path.join('2750', dr, dr, pt)
            img = Image.open(pth)
            img = ttransforms(img)

            im = pt.split(".")[0] + ".pt"
            pth = os.path.join(svdir,im)

            ch, row, col = img.size()
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
            torch.save(out.permute(2,0,1), pth)
            
            #lb = torch.tensor(label[dr]).unsqueeze(0)

            
            #torch.save(noisy, pth)
            #adv_images = attack(img.unsqueeze(0), lb)
            #torch.save(adv_images.squeeze(), pth)
            i+=1
        """
        i=0
        for pt in vallst:
            if (i+1)%500 == 0:
                print('Done with {}/{} images'.format(i+1, len(vallst)))
            dr = pt.split("_")[0]
            pth = os.path.join('2750', dr, dr, pt)
            img = Image.open(pth)
            img = ttransforms(img)

            im = pt.split(".")[0] + ".pt"
            pth = os.path.join(svdir,im)

            ch, row, col = img.size()
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
            torch.save(out.permute(2,0,1), pth)
            
            #lb = torch.tensor(label[dr]).unsqueeze(0)

            #torch.save(noisy, pth)
            #adv_images = attack(img.unsqueeze(0), lb)
            #torch.save(adv_images.squeeze(), pth)
            i+=1

        i=0
        svdir = sdir+'test/'+noise
        for pt in testlst:
            if (i+1)%500 == 0:
                print('Done with {}/{} images'.format(i+1, len(testlst)))
            dr = pt.split("_")[0]
            pth = os.path.join('2750', dr, dr, pt)
            img = Image.open(pth)
            img = ttransforms(img)

            im = pt.split(".")[0] + ".pt"
            pth = os.path.join(svdir,im)

            ch, row, col = img.size()
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
            torch.save(out.permute(2,0,1), pth)
            
            #lb = torch.tensor(label[dr]).unsqueeze(0)

            #torch.save(noisy, pth)
            #adv_images = attack(img.unsqueeze(0), lb)
            #torch.save(adv_images.squeeze(), pth)
            i+=1

        """
        for img, lb in loader:
            if (i+1)%500 == 0:
                print('Done with {}/{} images'.format(i+1, len(loader)))
            im = 'image'+str(i)+'.pt'
            pth = os.path.join(svdir,im)
            adv_images = attack(img, lb)
            torch.save(adv_images.squeeze(), pth)
            i+=1
        """
        
     

    elif noise == 'pgd':
        """
        svdir = sdir + '/imgs/{}'.format(noise)
        loader = torch.utils.data.DataLoader(data, batch_size=bs, shuffle=False)
        i = 0
        atck = torchattacks.FGSM(model, eps=8/255) #torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        for img, lb in loader:
            #if i <= 100000:
            #    i+=1
            #    continue
            if i == 100000:
                break
            if (i+1)%500 == 0:
                print('Done with {}/{} images'.format(i+1, len(loader)))
            
            #im = 'image'+str(i)+'.pt'
            ch, row, col = img.squeeze().size()
            #im = iname[0].split(".")[0]
            #pth = os.path.join(svdir, im+'.pt')
            
            adv_images = atck(img, lb)
            #print(adv_images.squeeze().to('cpu').flatten().size())
            lst.append(adv_images.squeeze().to('cpu').flatten())
            #torch.save(adv_images.squeeze(), pth)
            i+=1
        
        for img, lb, iname in loader:
            if (i+1)%500 == 0:
                print('Done with {}/{} images'.format(i+1, len(loader)))
            #im = 'image'+str(i)+'.pt'
            ch, row, col = img.squeeze().size()
            im = iname[0].split(".")[0]
            pth = os.path.join(svdir, im+'.pt')
            
            adv_images = atck(img, lb)
            torch.save(adv_images.squeeze(), pth)
            i+=1
        """
        
        svdir = sdir + '/train/{}'.format(noise)
        loader = torch.utils.data.DataLoader(data, batch_size=bs, shuffle=False) 
        attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        i = 0
        for img, lb in loader:
            if (i+1)%500 == 0:
                print('Done with {}/{} images'.format(i+1, len(loader)))
            im = 'image'+str(i)+'.pt'
            pth = os.path.join(svdir,im)
            adv_images = attack(img, lb)
            torch.save(adv_images.squeeze(), pth)
            i+=1
        
        svdir = sdir + '/test/{}'.format(noise)
        loader = torch.utils.data.DataLoader(testdata, batch_size=bs, shuffle=False) 
        attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        i = 0
        for img, lb in loader:
            if (i+1)%500 == 0:
                print('Done with {}/{} images'.format(i+1, len(loader)))
            im = 'image'+str(i)+'.pt'
            pth = os.path.join(svdir,im)
            adv_images = attack(img, lb)
            torch.save(adv_images.squeeze(), pth)
            i+=1
        """
        df = torch.stack(lst).numpy()
        print(df.shape)
        with h5py.File('celeb-vitb-fgsm-1.hdf5', 'w') as f:
            f.create_dataset('data', data = df, dtype='float32')
        print("DONE")

        fs = h5py.File('celeb-vitb-fgsm-1.hdf5', 'r')
        dset = fs['data']
        print(dset.shape)
        """
    
    elif noise == "gauss":
        svdir = sdir + '/imgs/{}'.format(noise)
        loader = torch.utils.data.DataLoader(data, batch_size=bs, shuffle=False)
        i = 0
        mean = 0
        var = 0.0005
        sigma = var**0.5
        for img, lb, iname in loader:
            if (i+1)%500 == 0:
                print('Done with {}/{} images'.format(i+1, len(loader)))
            #im = 'image'+str(i)+'.pt'
            ch, row, col = img.squeeze().size()
            im = iname[0].split(".")[0]
            pth = os.path.join(svdir, im+'.pt')
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(ch, row, col)
            noisy = img.squeeze() + torch.tensor(gauss)
            torch.save(noisy, pth)
            i+=1

        """
        svdir = sdir + '/train/{}'.format(noise)
        loader = torch.utils.data.DataLoader(data, batch_size=bs, shuffle=False)
        i = 0
        mean = 0
        var = 0.0005
        sigma = var**0.5
        for img, lb in loader:
            if (i+1)%500 == 0:
                print('Done with {}/{} images'.format(i+1, len(loader)))
            im = 'image'+str(i)+'.pt'
            ch, row, col = img.squeeze().size()

            pth = os.path.join(svdir, im)
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(ch, row, col)
            noisy = img.squeeze() + torch.tensor(gauss)
            torch.save(noisy, pth)
            i+=1
        
        svdir = sdir + '/test/{}'.format(noise)
        loader = torch.utils.data.DataLoader(testdata, batch_size=bs, shuffle=False)
        i = 0
        mean = 0
        var = 0.0005
        sigma = var**0.5
        for img, lb in loader:
            if (i+1)%500 == 0:
                print('Done with {}/{} images'.format(i+1, len(loader)))
            im = 'image'+str(i)+'.pt'
            ch, row, col = img.squeeze().size()

            pth = os.path.join(svdir, im)
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(ch, row, col)
            noisy = img.squeeze() + torch.tensor(gauss)
            torch.save(noisy, pth)
            i+=1
        """

    elif noise == "s&p":
        svdir = sdir + '/imgs/{}'.format(noise)
        loader = torch.utils.data.DataLoader(data, batch_size=bs, shuffle=False)
        id = 0
        s_vs_p = 0.5
        amount = 0.004
        for img, lb, iname in loader:
            if (id+1)%500 == 0:
                print('Done with {}/{} images'.format(id+1, len(loader)))
            #im = 'image'+str(id)+'.pt'
            ch, row, col = img.squeeze().size()

            im = iname[0].split(".")[0]
            pth = os.path.join(svdir, im+'.pt')

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
            torch.save(out.permute(2,0,1), pth)
            id+=1   
        
        """    
        svdir = sdir + '/train/{}'.format(noise)
        loader = torch.utils.data.DataLoader(data, batch_size=bs, shuffle=False)
        id = 0
        s_vs_p = 0.5
        amount = 0.004
        for img, lb in loader:
            if (id+1)%500 == 0:
                print('Done with {}/{} images'.format(id+1, len(loader)))
            im = 'image'+str(id)+'.pt'
            ch, row, col = img.squeeze().size()

            pth = os.path.join(svdir, im)
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
            torch.save(out.permute(2,0,1), pth)
            id+=1   

        svdir = sdir + '/test/{}'.format(noise)
        loader = torch.utils.data.DataLoader(testdata, batch_size=bs, shuffle=False)
        id=0
        for img, lb in loader:
            if (id+1)%500 == 0:
                print('Done with {}/{} images'.format(id+1, len(loader)))
            im = 'image'+str(id)+'.pt'
            ch, row, col = img.squeeze().size()

            pth = os.path.join(svdir, im)
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
            torch.save(out.permute(2,0,1), pth)
            id+=1   
        """

    elif noise == 'jitter':
        svdir = sdir + '/{}'.format(noise)
        loader = torch.utils.data.DataLoader(testdata, batch_size=bs, shuffle=False)
        i = 0
        atck = torchattacks.Jitter(model, eps=8/255, alpha=2/255, steps=10, scale=10, std=0.1, random_start=True)
        for img, lb in loader:
            if (i+1)%500 == 0:
                print('Done with {}/{} images'.format(i+1, len(loader)))
            im = 'image'+str(i)+'.pt'
            ch, row, col = img.squeeze().size()
            #im = iname[0].split(".")[0]
            pth = os.path.join(svdir, im) #+'.pt')
            
            adv_images = atck(img, lb)
            torch.save(adv_images.squeeze(), pth)
            i+=1

    elif noise == 'deepfool':
        svdir = sdir + '/imgs/{}'.format(noise)
        loader = torch.utils.data.DataLoader(testdata, batch_size=bs, shuffle=False)
        i = 0
        atck = DeepFool(model, device)
        for img, lb in loader:
            if (i+1)%500 == 0:
                print('Done with {}/{} images'.format(i+1, len(loader)))
            im = 'image'+str(i)+'.pt'
            ch, row, col = img.squeeze().size()
            #im = iname[0].split(".")[0]
            pth = os.path.join(svdir, im) #+'.pt')
            
            adv_images = atck(img, lb)
            torch.save(adv_images.squeeze(), pth)
            i+=1
        """
        svdir = sdir + '/test/{}'.format(noise)
        loader = torch.utils.data.DataLoader(testdata, batch_size=bs, shuffle=False)
        attack = torchattacks.attacks.deepfool.DeepFool(model, steps=50, overshoot=0.02)
        i = 0
        for img, lb in loader:
            if (i+1)%500 == 0:
                print('Done with {}/{} images'.format(i+1, len(loader)))
            im = 'image'+str(i)+'.pt'
            pth = os.path.join(svdir,im)
            adv_images = attack(img, lb)
            torch.save(adv_images.squeeze(), pth)
            i+=1
        """

    elif noise == "poisson":
        svdir = sdir + '/{}'.format(noise)
        loader = torch.utils.data.DataLoader(testdata, batch_size=bs, shuffle=False)
        i = 0
        for img, lb in loader:
            if (i+1)%500 == 0:
                print('Done with {}/{} images'.format(i+1, len(loader)))
            im = 'image'+str(i)+'.pt'
            ch, row, col = img.squeeze().size()
            pth = os.path.join(svdir, im)
            
            img = img.squeeze()
            vals = len(np.unique(img))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(img * vals) / float(vals)
            noisy = torch.tensor(noisy)
            noisy = torch.clamp(noisy, 0, 1)
            torch.save(noisy, pth)
            i+=1

    elif noise =="speckle":
        svdir = sdir + '/{}'.format(noise)
        loader = torch.utils.data.DataLoader(testdata, batch_size=bs, shuffle=False)
        i = 0
        for img, lb in loader:
            if (i+1)%500 == 0:
                print('Done with {}/{} images'.format(i+1, len(loader)))
            im = 'image'+str(i)+'.pt'
            ch, row, col = img.squeeze().size()
            pth = os.path.join(svdir, im)
            
            img = img.squeeze()
            gauss = np.random.randn(ch,row,col)
            gauss = gauss.reshape(ch,row,col) 
            noisy = img + img * gauss
            noisy = torch.tensor(noisy)
            noisy = torch.clamp(noisy, 0, 1)
            torch.save(noisy, pth)
            i+=1

if ctype == 'corrupt':
    """
    if dset == 'cifar':
        #data_dir = 'cifar/train'
        data_dir = 'cifar/test'#.format(mt)
        pair = ['fgsm', 'gauss', 'pgd', 's&p']
        modt = mt #'r50' #'vitb'

        cl = '3cl'

    if mt == 'r50':
        pth = 'saved_models/pretrain/r50-5-0.0001-mmd-3.pth.tar'
        model = DetResnet50(pth)
        savepth = 'saved_models/det-r50-5-0.0001-mmd-nc.pth.tar'
    elif mt == 'r18':
        pth = 'saved_models/pretrain/resnet50-ep3-0.0001-mmd.pth.tar'
        model = DetResnet18(pth)
    elif mt == 'vitb':
        pth = 'saved_models/pretrain/vitb-3-0.0001-mmd-3.pth.tar'
        model = DetViTb(pth)
        savepth = 'saved_models/det-vitb-3-0.0001-mmd-3class.pth.tar' #
        #'saved_models/det-vitb-3-0.0001-mmd-3class.pth.tar'
    
    state = torch.load(savepth)
    try:
        model.load_state_dict(state['state_dict'])
        print("Model Loaded 1")
    except RuntimeError:
        dic = {}
        for k,v in state['state_dict'].items():
            dic[k.replace("module.", "")] = v
        model.load_state_dict(dic)
        print("Model Loaded 2")
    model = model.to(device)
    """

    epochs = 5
    cl = '3cl'
    if mt == 'r50':
        model = torchvision.models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, nc)  
    elif mt == 'r18':
        model = torchvision.models.resnet18(pretrained=False)#weights=ResNet18_Weights.IMAGENET1K_V1) #MCNN()
        model.fc = nn.Linear(512, nc)
    elif mt == 'vitb':
        model = torchvision.models.vit_b_16(pretrained=False)
        model.heads.head = nn.Linear(768, nc)
        print("VIT IT IS")
        epochs = 5

    lr = 1e-4
    bs = 64
    if dset == 'cifar':
        savepth = 'saved_models/classifier/{}-{}-{}.pth.tar'.format(mt,epochs,lr)
    else:
        savepth = 'saved_models/classifier/{}-{}-{}-{}.pth.tar'.format(mt, epochs, lr, dset)
    data_dir = '{}/test/'.format(dset)
    
    
    state = torch.load(savepth)
    try:
        model.load_state_dict(state['state_dict'])
        print("Model Loaded 1")
    except RuntimeError:
        dic = {}
        for k,v in state['state_dict'].items():
            dic[k.replace("module.", "")] = v
        model.load_state_dict(dic)
        print("Model Loaded 2")
    model = model.to(device)
    
    #tar -xf
    dirs = ['brightness.npy', 'contrast.npy', 'defocus_blur.npy', 'elastic_transform.npy',
    'fog.npy', 'frost.npy', 'gaussian_blur.npy', 'gaussian_noise.npy', 'glass_blur.npy',
    'impulse_noise.npy', 'jpeg_compression.npy', 'motion_blur.npy', 'pixelate.npy', 'saturate.npy',
    'shot_noise.npy', 'snow.npy', 'spatter.npy', 'speckle_noise.npy', 'zoom_blur.npy']
    dirs = ['gaussian_noise.npy', 'impulse_noise.npy', 'shot_noise.npy','speckle_noise.npy']
    mdd = 'gauss'
    dt_dir = 'CIFAR-10-C/'
    label_dir = 'CIFAR-10-C/labels.npy' #None
    for dr in dirs:
        data_dir = dt_dir + dr
        for i in range(5):
            sev = i
            print(dr, sev)
            data = detCordata(data_dir, label_dir, sev)
            testloader = DataLoader(data, batch_size=64, num_workers = 2, shuffle=False)
            tpred, pred = test(model, testloader, data, device, mdd, dset=dset, cl=cl)
            #print(confusion_matrix(tpred, pred))

if ctype == 'testdet':
    if dset == 'euro':
        comb = ['fgsm', 'gauss','pgd', 's&p']
        data_dir = 'euro/vitb/train/'
        cdata = EuroDataset(data_dir+comb[0], testlst, ttransforms, True)
        bdata = EuroDataset(data_dir+comb[1], testlst, ttransforms, True)
        ddata = EuroDataset(data_dir+comb[2], testlst, ttransforms, True)
        fdata = EuroDataset(data_dir+comb[3], testlst, ttransforms, True)
        tcleandata = euDetdata(testdata, 0, 'clean')
        tfdata = euDetdata(testdata, cdata, comb[0])
        tgdata = euDetdata(testdata, bdata, comb[1])
        tpdata = euDetdata(testdata, ddata, comb[2])
        tsdata = euDetdata(testdata, fdata, comb[3])

        

    if dset == 'cifar':
        #data_dir = 'cifar/train'
        data_dir = 'cifar/test'#.format(mt)
        pair = ['fgsm', 'gauss', 'pgd', 's&p']
        modt = mt #'r50' #'vitb'

        cl = '3cl'
        #cleandata = detdata(data_dir, testdata, '', 'clean', mt=modt)
        #fdata = detdata(data_dir, testdata, 'fgsm', mt=modt,cl=cl)
        #pdata = detdata(data_dir, testdata, 'pgd', mt=modt,cl=cl)

    if mt == 'r50':
        pth = 'saved_models/pretrain/r50-5-0.0001-mmd-3.pth.tar'
        model = DetResnet50(pth)
        savepth = 'saved_models/det-r50-5-0.0001-mmd-nc.pth.tar'
    elif mt == 'r18':
        pth = 'saved_models/pretrain/resnet50-ep3-0.0001-mmd.pth.tar'
        model = DetResnet18(pth)
    elif mt == 'vitb':
        pth = 'saved_models/pretrain/vitb-3-0.0001-mmd-3.pth.tar'
        model = DetViTb(pth)
        savepth = 'saved_models/det-vitb-5-0.001-euro-mmd-3class.pth.tar'
        #'saved_models/det-vitb-3-0.0001-mmd-3class.pth.tar' #
        #'saved_models/det-vitb-3-0.0001-mmd-3class.pth.tar'
    bs = 64
    state = torch.load(savepth)
    try:
        model.load_state_dict(state['state_dict'])
        print("Model Loaded 1")
    except RuntimeError:
        dic = {}
        for k,v in state['state_dict'].items():
            dic[k.replace("module.", "")] = v
        model.load_state_dict(dic)
        print("Model Loaded 2")
    model = model.to(device)

    pth1 = 'datacel/celeba/list_attr_celeba.csv'
    df = pd.read_csv(pth1)

    data_dir = 'datacel/imgs'
    cl='3cl'

    modt = 'vitb'
    mdd = 'clean'
    print(mdd)
    testloader = DataLoader(tcleandata, batch_size=bs, num_workers = 2, shuffle=True)
    tpred, pred = test(model, testloader, tcleandata, device, mdd, dset=dset, cl=None)
    print(confusion_matrix(tpred, pred))

    mdd = 'fgsm'
    print(mdd)
    testloader = DataLoader(tfdata, batch_size=bs, num_workers = 2, shuffle=True)
    tpred, pred = test(model, testloader, tcleandata, device, mdd, dset=dset, cl=None)
    print(confusion_matrix(tpred, pred))

    mdd = 'pgd'
    print(mdd)
    testloader = DataLoader(tpdata, batch_size=bs, num_workers = 2, shuffle=True)
    tpred, pred = test(model, testloader, tcleandata, device, mdd, dset=dset, cl=None)
    print(confusion_matrix(tpred, pred))

    mdd = 's&p'
    print(mdd)
    testloader = DataLoader(tsdata, batch_size=bs, num_workers = 2, shuffle=True)
    tpred, pred = test(model, testloader, tcleandata, device, mdd, dset=dset, cl=None)
    print(confusion_matrix(tpred, pred))

    mdd = 'gauss'
    print(mdd)
    testloader = DataLoader(tgdata, batch_size=bs, num_workers = 2, shuffle=True)
    tpred, pred = test(model, testloader, tcleandata, device, mdd, dset=dset, cl=None)
    print(confusion_matrix(tpred, pred))
    
    """
    cleandata = detdata(data_dir, testdata, '', df, 'clean', mt=modt, cl=cl)
    mdd = 'clean'
    print("Org")
    testloader = DataLoader(cleandata, batch_size=bs, num_workers = 2, shuffle=True)
    tpred, pred = test(model, testloader, cleandata, device, mdd, dset=dset, cl=cl)
    print(confusion_matrix(tpred, pred))

    mdd = 'fgsm'
    print(mdd)
    cwdata = detdata(data_dir, testdata, mdd, df, mt=modt,cl=cl)    
    testloader = DataLoader(cwdata, batch_size=bs, num_workers = 2, shuffle=True)
    tpred, pred = test(model, testloader, cwdata, device, mdd, dset=dset,cl=cl)
    print(confusion_matrix(tpred, pred))

    mdd = 'pgd'
    print(mdd)
    cwdata = detdata(data_dir, testdata, mdd, df, mt=modt,cl=cl)    
    testloader = DataLoader(cwdata, batch_size=bs, num_workers = 2, shuffle=True)
    tpred, pred = test(model, testloader, cwdata, device, mdd, dset=dset,cl=cl)
    print(confusion_matrix(tpred, pred))
    
    mdd = 'fastfgsm'
    print(mdd)
    cwdata = detdata(data_dir, testdata, mdd, df, mt=modt,cl=cl)    
    testloader = DataLoader(cwdata, batch_size=bs, num_workers = 2, shuffle=True)
    tpred, pred = test(model, testloader, cwdata, device, mdd, dset=dset,cl=cl)
    print(confusion_matrix(tpred, pred))

    mdd = 'rfgsm'
    print(mdd)
    cwdata = detdata(data_dir, testdata, mdd, df, mt=modt,cl=cl)    
    testloader = DataLoader(cwdata, batch_size=bs, num_workers = 2, shuffle=True)
    tpred, pred = test(model, testloader, cwdata, device, mdd, dset=dset,cl=cl)
    print(confusion_matrix(tpred, pred))

    mdd = 'bim'
    print(mdd)
    cwdata = detdata(data_dir, testdata, mdd, df, mt=modt,cl=cl)    
    testloader = DataLoader(cwdata, batch_size=bs, num_workers = 2, shuffle=True)
    tpred, pred = test(model, testloader, cwdata, device, mdd, dset=dset,cl=cl)
    print(confusion_matrix(tpred, pred))
    """
    mdd = 'gauss'
    print(mdd)
    cwdata = detdata(data_dir, testdata, mdd, df, mt=modt,cl=cl)    
    testloader = DataLoader(cwdata, batch_size=bs, num_workers = 2, shuffle=True)
    tpred, pred = test(model, testloader, cwdata, device, mdd, dset=dset,cl=cl)
    print(confusion_matrix(tpred, pred))

    mdd = 's&p'
    print(mdd)
    cwdata = detdata(data_dir, testdata, mdd, df, mt=modt,cl=cl)    
    testloader = DataLoader(cwdata, batch_size=bs, num_workers = 2, shuffle=True)
    tpred, pred = test(model, testloader, cwdata, device, mdd, dset=dset,cl=cl)
    print(confusion_matrix(tpred, pred))
    
    """
    cleandata = detdata(data_dir, testdata, '', 'clean', mt=modt, cl=cl)
    mdd = 'clean'
    print("Org")
    testloader = DataLoader(cleandata, batch_size=bs, num_workers = 2, shuffle=True)
    tpred, pred = test(model, testloader, cleandata, device, mdd, dset=dset, cl=cl)
    print(confusion_matrix(tpred, pred))
    
    #modt = 'vitb'
    mdd = 'cw'
    print(mdd)
    cwdata = detdata(data_dir, testdata, mdd, mt=modt,cl=cl)    
    testloader = DataLoader(cwdata, batch_size=bs, num_workers = 2, shuffle=True)
    tpred, pred = test(model, testloader, cwdata, device, mdd, dset=dset,cl=cl)
    print(confusion_matrix(tpred, pred))
    
    mdd = 'deepfool'
    print(mdd)
    cwdata = detdata(data_dir, testdata, mdd, mt=modt,cl=cl)    
    testloader = DataLoader(cwdata, batch_size=bs, num_workers = 2, shuffle=True)
    tpred, pred = test(model, testloader, cwdata, device, mdd, dset=dset,cl=cl)
    #print(tpred, pred)
    print(confusion_matrix(tpred, pred))
    
    mdd = 'fab'
    print(mdd)
    cwdata = detdata(data_dir, testdata, mdd, mt=modt,cl=cl)    
    testloader = DataLoader(cwdata, batch_size=bs, num_workers = 2, shuffle=True)
    tpred, pred = test(model, testloader, cwdata, device, mdd, dset=dset,cl=cl)
    print(confusion_matrix(tpred, pred))

    mdd = 'fgsm'
    print(mdd)
    cwdata = detdata(data_dir, testdata, mdd, mt=modt,cl=cl)    
    testloader = DataLoader(cwdata, batch_size=bs, num_workers = 2, shuffle=True)
    tpred, pred = test(model, testloader, cwdata, device, mdd, dset=dset,cl=cl)
    print(confusion_matrix(tpred, pred))

    mdd = 'pgd'
    print(mdd)
    cwdata = detdata(data_dir, testdata, mdd, mt=modt,cl=cl)    
    testloader = DataLoader(cwdata, batch_size=bs, num_workers = 2, shuffle=True)
    tpred, pred = test(model, testloader, cwdata, device, mdd, dset=dset,cl=cl)
    print(confusion_matrix(tpred, pred))

    mdd = 'one'
    print(mdd)
    cwdata = detdata(data_dir, testdata, mdd, mt=modt,cl=cl)    
    testloader = DataLoader(cwdata, batch_size=bs, num_workers = 2, shuffle=True)
    tpred, pred = test(model, testloader, cwdata, device, mdd, dset=dset,cl=cl)
    print(confusion_matrix(tpred, pred))

    mdd = 'pgdl2'
    print(mdd)
    cwdata = detdata(data_dir, testdata, mdd, mt=modt,cl=cl)    
    testloader = DataLoader(cwdata, batch_size=bs, num_workers = 2, shuffle=True)
    tpred, pred = test(model, testloader, cwdata, device, mdd, dset=dset,cl=cl)
    print(confusion_matrix(tpred, pred))
    
    mdd = 'gauss'
    print(mdd)
    cwdata = detdata(data_dir, testdata, mdd, mt=modt,cl=cl)    
    testloader = DataLoader(cwdata, batch_size=bs, num_workers = 2, shuffle=True)
    tpred, pred = test(model, testloader, cwdata, device, mdd, dset=dset,cl=cl)
    print(confusion_matrix(tpred, pred))

    mdd = 's&p'
    print(mdd)
    cwdata = detdata(data_dir, testdata, mdd, mt=modt,cl=cl)    
    testloader = DataLoader(cwdata, batch_size=bs, num_workers = 2, shuffle=True)
    tpred, pred = test(model, testloader, cwdata, device, mdd, dset=dset,cl=cl)
    print(confusion_matrix(tpred, pred))
    
    mdd = 'fastfgsm'
    print(mdd)
    cwdata = detdata(data_dir, testdata, mdd, mt=modt,cl=cl)    
    testloader = DataLoader(cwdata, batch_size=bs, num_workers = 2, shuffle=True)
    tpred, pred = test(model, testloader, cwdata, device, mdd, dset=dset,cl=cl)
    print(confusion_matrix(tpred, pred))

    mdd = 'rfgsm'
    print(mdd)
    cwdata = detdata(data_dir, testdata, mdd, mt=modt,cl=cl)    
    testloader = DataLoader(cwdata, batch_size=bs, num_workers = 2, shuffle=True)
    tpred, pred = test(model, testloader, cwdata, device, mdd, dset=dset,cl=cl)
    print(confusion_matrix(tpred, pred))

    mdd = 'mifgsm'
    print(mdd)
    cwdata = detdata(data_dir, testdata, mdd, mt=modt,cl=cl)    
    testloader = DataLoader(cwdata, batch_size=bs, num_workers = 2, shuffle=True)
    tpred, pred = test(model, testloader, cwdata, device, mdd, dset=dset,cl=cl)
    print(confusion_matrix(tpred, pred))

    mdd = 'bim'
    print(mdd)
    cwdata = detdata(data_dir, testdata, mdd, mt=modt,cl=cl)    
    testloader = DataLoader(cwdata, batch_size=bs, num_workers = 2, shuffle=True)
    tpred, pred = test(model, testloader, cwdata, device, mdd, dset=dset,cl=cl)
    print(confusion_matrix(tpred, pred))

    mdd = 'upgd'
    print(mdd)
    cwdata = detdata(data_dir, testdata, mdd, mt=modt,cl=cl)    
    testloader = DataLoader(cwdata, batch_size=bs, num_workers = 2, shuffle=True)
    tpred, pred = test(model, testloader, cwdata, device, mdd, dset=dset,cl=cl)
    print(confusion_matrix(tpred, pred))
    """

if ctype == 'detector':
    if dset == 'euro':
        comb = ['fgsm', 'gauss','pgd', 's&p']
        data_dir = 'euro/vitb/train/'
        cdata = EuroDataset(data_dir+comb[0], trainlst, ttransforms, True)
        bdata = EuroDataset(data_dir+comb[1], trainlst, ttransforms, True)
        ddata = EuroDataset(data_dir+comb[2], trainlst, ttransforms, True)
        fdata = EuroDataset(data_dir+comb[3], trainlst, ttransforms, True)
        tcleandata = euDetdata(traindata, 0, 'clean')
        tfdata = euDetdata(traindata, cdata, comb[0])
        tgdata = euDetdata(traindata, bdata, comb[1])
        tpdata = euDetdata(traindata, ddata, comb[2])
        tsdata = euDetdata(traindata, fdata, comb[3])

        cdata = EuroDataset(data_dir+comb[0], vallst, ttransforms, True)
        bdata = EuroDataset(data_dir+comb[1], vallst, ttransforms, True)
        ddata = EuroDataset(data_dir+comb[2], vallst, ttransforms, True)
        fdata = EuroDataset(data_dir+comb[3], vallst, ttransforms, True)
        vcleandata = euDetdata(valdata, 0, 'clean')
        vfdata = euDetdata(valdata, cdata, comb[0])
        vgdata = euDetdata(valdata, bdata, comb[1])
        vpdata = euDetdata(valdata, ddata, comb[2])
        vsdata = euDetdata(valdata, fdata, comb[3])

    else:
        if dset == 'cifar':
            data_dir = 'cifar/train'
            cl = '3cl'
            df = None
            #tdata_dr = 'cifar/test'
            #pair = ['fgsm', 'gauss', 'pgd', 's&p']
        if dset == 'celeb-gen':
            data_dir = 'datacel/imgs'
            data = traindata
            cl = '3cl'

            pth1 = 'datacel/celeba/list_attr_celeba.csv'
            df = pd.read_csv(pth1)

        cleandata = detdata(data_dir, data, '', df, 'clean')
        fdata = detdata(data_dir, data, 'fgsm', df, mt=mt,cl=cl)
        pdata = detdata(data_dir, data, 'pgd', df, mt=mt,cl=cl)
        gdata = detdata(data_dir, data, 'gauss', df, mt=mt,cl=cl)
        sdata = detdata(data_dir, data, 's&p', df, mt=mt,cl=cl)
        split = int(0.1*len(cleandata))
        ids = [i for i in range(len(cleandata))]
        vids = ids[:split]
        tids = ids[split:]

        tcleandata = torch.utils.data.Subset(cleandata, tids)
        vcleandata = torch.utils.data.Subset(cleandata, vids)
        tfdata = torch.utils.data.Subset(fdata, tids)
        vfdata = torch.utils.data.Subset(fdata, vids)
        tpdata = torch.utils.data.Subset(pdata, tids)
        vpdata = torch.utils.data.Subset(pdata, vids)
        tgdata = torch.utils.data.Subset(gdata, tids)
        vgdata = torch.utils.data.Subset(gdata, vids)
        tsdata = torch.utils.data.Subset(sdata, tids)
        vsdata = torch.utils.data.Subset(sdata, vids)

    totdata = tcleandata + tfdata + tpdata + tgdata + tsdata
    valdata = vcleandata + vfdata + vpdata + vgdata + vsdata

    if torch.cuda.device_count() >= 2:
        bs = 400
        trainloader = DataLoader(totdata, batch_size=bs, num_workers = 4, shuffle=True)
        valloader = DataLoader(valdata, batch_size=100, num_workers = 4, shuffle=True)
    else:
        trainloader = DataLoader(totdata, batch_size=100, num_workers = 2, shuffle=True)
        valloader = DataLoader(valdata, batch_size=100, num_workers = 2, shuffle=True)
    
    if mt == 'r50':
        pth = 'saved_models/pretrain/r50-5-0.0001-mmd-3.pth.tar'
        model = DetResnet50(pth)
    elif mt == 'r18':
        pth = 'saved_models/pretrain/r18-3-0.0001-mmd.pth.tar'
        model = DetResnet18(pth)
    elif mt == 'vitb':
        #pth = 'saved_models/pretrain/vitb-3-0.0001-cifar-mmd-all.pth.tar'
        #pth = 'saved_models/pretrain/vitb-3-0.0001-celeb-gen-mmd-2-all.pth.tar'
        #pth = 'saved_models/pretrain/vitb-3-0.0001-euro-mmd-all.pth.tar'
        pth = 'saved_models/pretrain/vitb-3-0.0001-cifar-mmd-1st-term.pth.tar'
        model = DetViTb(pth)
    elif mt == 'cnextb':
        pth = 'saved_models/pretrain/cnextb-3-0.0001-mmd-all.pth.tar'
        model = DetConvnextb(pth)
    
    if torch.cuda.device_count() >= 2:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    lr = 1e-3
    epochs = 5
    tdata = len(totdata)
    vdata = len(valdata)
    savepth = 'saved_models/det-{}-{}-{}-{}-mmd-term.pth.tar'.format(mt, epochs, lr, dset)
    log = 'logs/det-{}-{}-{}.txt'.format(mt,epochs, lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    wlist = torch.tensor([1,1,1]).float()
    wlist = wlist.to(device)
    loss = nn.CrossEntropyLoss(weight=wlist)
    train(epochs, trainloader, valloader, model, optimizer, loss, savepth, device, log, tdata, vdata, dset=dset)

if ctype == 'pretrain':
    if dset == 'cifar':
        data_dir = 'cifar/train'
        #tdata_dir = 'cifar/test'
        pair1 = ['cw', 's&p']
        pair2 = ['pgdl2', 'gauss']
        mot = mt
        tdata1 = ci_data(data_dir, data, pair2[0], pair2[1], mod=mot)
        tdata2 = ci_data(data_dir, data, pair1[0], pair1[1], mod=mot)
        #testdata = ci_data(tdata_dir, testdata, pair2[0], pair2[1], mt)
        tdata = tdata1 + tdata2
    

    if mt == 'r50':
        spth = 'saved_models/classifier/r50-5-0.0001.pth.tar'
        model = Resnet50(spth)
    elif mt == 'r18':
        model = Resnet18()
    elif mt == 'vitb':
        spth = 'saved_models/classifier/vitb-5-0.0001.pth.tar'
        model = ViTb(svpth=spth)
    elif mt == 'cnextb':
        spth = 'saved_models/classifier/cnextb-5-0.0001.pth.tar'
        model = Convnextb(svpth=spth)

    if torch.cuda.device_count() >= 2:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    
    tlen = len(tdata)
    if torch.cuda.device_count() >= 2:
        bs = 128
        train_dataloader = DataLoader(tdata, batch_size=bs, num_workers = 4, shuffle=True)
        #plot_dataloader = DataLoader(tdata, batch_size=1, num_workers = 4, shuffle=False)
        #test_dataloader = DataLoader(testdata, batch_size=bs, num_workers=4, shuffle=False)
    else:
        bs = 100
        train_dataloader = DataLoader(tdata, batch_size=bs, num_workers = 1, shuffle=True)

    mloss = MMDLoss(bs)
    #celoss = torch.nn.CrossEntropyLoss()

    epochs = 3
    lr = 1e-5
    savepath = 'saved_models/pretrain/{}-{}-{}-mmd.pth.tar'.format(mt, epochs, lr)
    #log = 'logs/pre-{}-{}-{}.txt'.format(mt,epochs, lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        tloss = 0
        t1loss, t2loss, t3loss = 0,0,0
        for img, inimg, noimg, label in tqdm(train_dataloader):
            #print(img, inimg, noimg)
            #quit()
            #print(img.size())
            img, inimg, noimg, label = img.to(device), inimg.to(device), noimg.to(device), label.to(device)
            
            img, feat = model(img)
            inimg, ifeat = model(inimg)
            noimg, nfeat = model(noimg)
            """
            labels = torch.zeros(img.size(0))
            #print(labels, label)
            closs = celoss(feat, labels.to(torch.uint8).to(device))
            labels = torch.ones(img.size(0))
            closs2 = celoss(ifeat, labels.to(torch.uint8).to(device))
            labels[:] = 2
            closs3 = celoss(nfeat, labels.to(torch.uint8).to(device))
            """
            #loss2 = nloss(img, inimg, noimg)
            loss, center = mloss(img, inimg, noimg) 
            floss = loss

            t1loss += floss 
        
            optimizer.zero_grad()
            floss.backward()
            optimizer.step()
            
            tloss += loss
        
        print("Epoch = {}, TLoss = {}".format(ep, tloss/len(train_dataloader)))
        """
        with open(log, 'a') as f:
            f.write(str(ep+1) + '\t'
            + str(tloss.item()/len(train_dataloader)) + '\n'   
            )
        """
        print(t1loss.item(), tloss.item())#, t2loss.item(), t3loss.item())
        state = {
            'epoch' : ep,
            'state_dict' : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }
        torch.save(state, savepath)
        
if ctype == 'pretrain2':
    if dset == 'cifar':
        data_dir = 'cifar/train'
        tdata_dir = 'cifar/test'
        pair1 = ['fgsm', 's&p']
        pair2 = ['pgd', 'gauss']
        mot = mt
        tdata1 = ci_data(data_dir, data, pair2[0], pair2[1], mod=mot)
        tdata2 = ci_data(data_dir, data, pair1[0], pair1[1], mod=mot)
        #testdata = ci_data(tdata_dir, testdata, pair2[0], pair2[1], mt)
        tdata = tdata1 + tdata2
    elif dset == 'euro':
        ddir = ['Clean', 'Compression', 'FGSM_Attack', 'G_Noise', 'PGD_Attack']
        data_dir = 'Multimodal_Data_Train'
        tdata_dir = 'Multimodal_Data_Test'

        ddir = 'Clean'
        pair1 = ['FGSM_Attack', 'Compression']
        pair2 = ['PGD_Attack', 'G_Noise']

        tdata1 = eu_data(data_dir, ddir, pair1[0], pair1[1])
        tdata2 = eu_data(data_dir, ddir, pair2[0], pair2[1])
        #tdata = tdata1 + tdata2

    if mt == 'r50':
        spth = 'saved_models/classifier/r50-5-0.0001.pth.tar'
        model = Resnet50(spth)
    elif mt == 'r18':
        model = Resnet18()
    elif mt == 'vitb':
        spth = 'saved_models/classifier/vitb-5-0.0001.pth.tar'
        model = ViTb(svpth=spth)

    if torch.cuda.device_count() >= 2:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    

    tlen = len(tdata)
    if torch.cuda.device_count() >= 2:
        bs = 64
        t1dataloader = DataLoader(tdata1, batch_size=bs, num_workers = 4, shuffle=True)
        t2dataloader = DataLoader(tdata2, batch_size=bs, num_workers = 4, shuffle=True)
        #plot_dataloader = DataLoader(tdata, batch_size=1, num_workers = 4, shuffle=False)
        #test_dataloader = DataLoader(testdata, batch_size=bs, num_workers=4, shuffle=False)
    else:
        bs = 16
        t1dataloader = DataLoader(tdata1, batch_size=bs, num_workers = 1, shuffle=True)
        t2dataloader = DataLoader(tdata2, batch_size=bs, num_workers = 1, shuffle=True)

    mloss = MMDCombLoss(bs)
    nloss = MMDLoss(bs)
    #celoss = torch.nn.CrossEntropyLoss()

    epochs = 5
    lr = 1e-4
    savepath = 'saved_models/pretrain/{}-{}-{}-mmd-comb.pth.tar'.format(mt, epochs, lr)
    log = 'logs/pre-{}-{}-{}-mmd-comb.txt'.format(mt,epochs, lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        tloss = 0
        t1loss, t2loss, t3loss = 0,0,0
        for b1, b2 in tqdm(zip(t1dataloader,t2dataloader)):
            #print(img.size())
            img, inimg, noimg, label = b1
            img, inimg, noimg, label = img.to(device), inimg.to(device), noimg.to(device), label.to(device)
            
            img1, inimg1, noimg1, label1 = b2
            img1, inimg1, noimg1, label1 = img1.to(device), inimg1.to(device), noimg1.to(device), label1.to(device)
            

            img, feat = model(img)
            inimg, ifeat = model(inimg)
            noimg, nfeat = model(noimg)

            img1, feat = model(img1)
            inimg1, ifeat = model(inimg1)
            noimg1, nfeat = model(noimg1)

            loss1, _ = nloss(img, inimg, noimg)
            loss2, _ = nloss(img1, inimg1, noimg1)
            loss3 = mloss(img, inimg, noimg, img1, inimg1, noimg1) 
            floss = loss1+loss2+loss3

            t1loss += loss3
        
            optimizer.zero_grad()
            floss.backward()
            optimizer.step()
            
            tloss += loss1+loss2+loss3
        
        print("Epoch = {}, TLoss = {}".format(ep, tloss/len(tdata1)))
        """
        with open(log, 'a') as f:
            f.write(str(ep+1) + '\t'
            + str(tloss.item()/len(tdata1)) + '\n'   
            )
        """
        print(t1loss.item(), tloss.item())#, t2loss.item(), t3loss.item())
        state = {
            'epoch' : ep,
            'state_dict' : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }
        torch.save(state, savepath)

if ctype == '3pre': 
    if dset == 'cifar':
        data_dir = 'cifar/train'
        mot = mt
        tdata = ci3data(data_dir, data, mod=mot)

    if mt == 'r50':
        spth = 'saved_models/classifier/r50-5-0.0001.pth.tar'
        model = Resnet50(spth)
    elif mt == 'r18':
        model = Resnet18()
    elif mt == 'vitb':
        spth = 'saved_models/classifier/vitb-5-0.0001.pth.tar'
        model = ViTb(svpth=spth)

    if torch.cuda.device_count() >= 2:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    
    tlen = len(tdata)
    if torch.cuda.device_count() >= 2:
        bs = 128
        if mt == 'vitb':
            bs=64
        train_dataloader = DataLoader(tdata, batch_size=bs, num_workers = 4, shuffle=True)
    else:
        bs = 100
        train_dataloader = DataLoader(tdata, batch_size=bs, num_workers = 1, shuffle=True)

    close=False
    m2loss = MMD2Loss(bs, close)
    m2aloss = MMD2aLoss(bs, close)
    m3loss = MMD3Loss(bs, close)
    #celoss = torch.nn.CrossEntropyLoss()

    epochs = 5
    lr = 1e-4
    if not close:
        savepath = 'saved_models/pretrain/{}-{}-{}-mmd-3.pth.tar'.format(mt, epochs, lr)
    else:
        savepath = 'saved_models/pretrain/{}-{}-{}-mmd-close-3.pth.tar'.format(mt, epochs, lr)
    #log = 'logs/pre-{}-{}-{}.txt'.format(mt,epochs, lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        tloss = 0
        t1loss, t2loss, t3loss = 0,0,0

        ######################## Org & Modified ######################
        for img, inimg, noimg, inimg1, noimg1 in tqdm(train_dataloader):
            img, inimg, noimg, inimg1, noimg1 = img.to(device), inimg.to(device), noimg.to(device), inimg1.to(device), noimg1.to(device)
            
            img, _ = model(img)
            inimg, _ = model(inimg)
            noimg, _ = model(noimg)

            inimg1, _ = model(inimg1)
            noimg1, _ = model(noimg1)

            #loss2 = nloss(img, inimg, noimg)
            loss1 = m2loss(img, inimg, noimg) 
            loss2 = m2loss(img, inimg1, noimg1)

            loss3 = m2aloss(inimg, noimg) 
            loss4 = m2aloss(inimg1, noimg1)  

            loss5 = m3loss(inimg, noimg, inimg1, noimg1) 

            floss = loss1 + loss2 + loss3 + loss4 + loss5

            optimizer.zero_grad()
            floss.backward()
            optimizer.step()
            
            tloss += floss
            t1loss += floss
        """
        ######################## Intent & Non-Int ######################
        for img, inimg, noimg, inimg1, noimg1 in tqdm(train_dataloader):
            img, inimg, noimg, inimg1, noimg1 = img.to(device), inimg.to(device), noimg.to(device), inimg1.to(device), noimg1.to(device)
            
            inimg, _ = model(inimg)
            noimg, _ = model(noimg)

            inimg1, _ = model(inimg1)
            noimg1, _ = model(noimg1)

            loss1 = m2aloss(inimg, noimg) 
            loss2 = m2aloss(inimg1, noimg1) 

            floss = loss1 + loss2
        
            optimizer.zero_grad()
            floss.backward()
            optimizer.step()
            
            tloss += floss
            t2loss += floss

        ######################## Within Subclass ######################
        for img, inimg, noimg, inimg1, noimg1 in tqdm(train_dataloader):
            img, inimg, noimg, inimg1, noimg1 = img.to(device), inimg.to(device), noimg.to(device), inimg1.to(device), noimg1.to(device)
            
            inimg, _ = model(inimg)
            noimg, _ = model(noimg)

            inimg1, _ = model(inimg1)
            noimg1, _ = model(noimg1)

            loss = m3loss(inimg, noimg, inimg1, noimg1) 
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            tloss += loss
            t3loss += loss
        """
        print("Epoch = {}, TLoss = {}".format(ep, tloss/len(train_dataloader)))
        """
        with open(log, 'a') as f:
            f.write(str(ep+1) + '\t'
            + str(tloss.item()/len(train_dataloader)) + '\n'   
            )
        """
        print(t1loss.item(), tloss.item())#, t2loss.item(), t3loss.item())
        state = {
            'epoch' : ep,
            'state_dict' : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }
        torch.save(state, savepath)

if ctype == 'pre+det':
    if dset == 'cifar':
        data_dir = 'cifar/train'
        #tdata_dr = 'cifar/test'

        pair1 = ['fgsm', 's&p']
        pair2 = ['pgd', 'gauss']
        mot = mt
        tdata1 = ci_data(data_dir, data, pair2[0], pair2[1], mod=mot)
        tdata2 = ci_data(data_dir, data, pair1[0], pair1[1], mod=mot)
        #testdata = ci_data(tdata_dir, testdata, pair2[0], pair2[1], mt)
        tdata = tdata1 + tdata2

        pair = ['fgsm', 'gauss', 'pgd', 's&p']
        cleandata = detdata(data_dir, data, '', 'clean')
        fdata = detdata(data_dir, data, 'fgsm', mt=mot)
        pdata = detdata(data_dir, data, 'pgd', mt=mot)
        gdata = detdata(data_dir, data, 'gauss', mt=mot)
        sdata = detdata(data_dir, data, 's&p', mt=mot)
        
        split = int(0.1*len(cleandata))
        ids = [i for i in range(len(cleandata))]
        vids = ids[:split]
        tids = ids[split:]

        tcleandata = torch.utils.data.Subset(cleandata, tids)
        vcleandata = torch.utils.data.Subset(cleandata, vids)
        tfdata = torch.utils.data.Subset(fdata, tids)
        vfdata = torch.utils.data.Subset(fdata, vids)
        tpdata = torch.utils.data.Subset(pdata, tids)
        vpdata = torch.utils.data.Subset(pdata, vids)
        tgdata = torch.utils.data.Subset(gdata, tids)
        vgdata = torch.utils.data.Subset(gdata, vids)
        tsdata = torch.utils.data.Subset(sdata, tids)
        vsdata = torch.utils.data.Subset(sdata, vids)

        totdata = tcleandata + tfdata + tpdata + tgdata + tsdata
        valdata = vcleandata + vfdata + vpdata + vgdata + vsdata

    if torch.cuda.device_count() >= 2:
        bs = 100
        trainloader = DataLoader(totdata, batch_size=bs, num_workers = 4, shuffle=True)
        valloader = DataLoader(valdata, batch_size=100, num_workers = 4, shuffle=True)
        pretrain_dataloader = DataLoader(tdata, batch_size=bs, num_workers = 4, shuffle=True)
    else:
        bs=100
        trainloader = DataLoader(totdata, batch_size=bs, num_workers = 2, shuffle=True)
        valloader = DataLoader(valdata, batch_size=bs, num_workers = 2, shuffle=True)
        pretrain_dataloader = DataLoader(tdata, batch_size=bs, num_workers = 4, shuffle=True)
    
    if mt == 'r50':
        spth = 'saved_models/classifier/r50-5-0.0001.pth.tar'
        model = Resnet50(spth)
    elif mt == 'r18':
        model = Resnet18()
    elif mt == 'vitb':
        spth = 'saved_models/classifier/vitb-5-0.0001.pth.tar'
        model = ViTb(svpth=spth)

    model2 = Det()
    
    if torch.cuda.device_count() >= 2:
        model = torch.nn.DataParallel(model)
        model2 = torch.nn.DataParallel(model2)
    model = model.to(device)
    model2 = model2.to(device)

    epochs = 5
    lr = 1e-4
    lr2 = 1e-3
    
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    optim2 = torch.optim.AdamW(model2.parameters(), lr=lr2)

    tdata = len(totdata)
    vdata = len(valdata)
    savepth = 'saved_models/predet-{}-{}-{}-mmd-pre.pth.tar'.format(mt, epochs, lr)
    savepth2 = 'saved_models/predet-{}-{}-{}-mmd-det.pth.tar'.format(mt, epochs, lr2)
    log = 'logs/det-{}-{}-{}.txt'.format(mt,epochs, lr)
    
    #wlist = torch.tensor([1,1,1]).float()
    #wlist = wlist.to(device)
    mloss = MMDLoss(bs)
    celoss = nn.CrossEntropyLoss()

    for ep in range(epochs):
        tloss = 0
        t1loss, t2loss, t3loss = 0,0,0
        model.requires_grad=True
        for img, inimg, noimg, label in tqdm(pretrain_dataloader):

            img, inimg, noimg, label = img.to(device), inimg.to(device), noimg.to(device), label.to(device)
            
            img, feat = model(img)
            inimg, ifeat = model(inimg)
            noimg, nfeat = model(noimg)
            
            #loss2 = nloss(img, inimg, noimg)
            loss, center = mloss(img, inimg, noimg) 
            floss = loss

            t1loss += floss 
        
            optim.zero_grad()
            floss.backward()
            optim.step()
            
            tloss += loss
        
        print("Epoch = {}, TLoss = {}".format(ep, tloss/len(pretrain_dataloader)))
        """
        with open(log, 'a') as f:
            f.write(str(ep+1) + '\t'
            + str(tloss.item()/len(train_dataloader)) + '\n'   
            )
        """
        print(t1loss.item(), tloss.item())#, t2loss.item(), t3loss.item())
        state = {
            'epoch' : ep,
            'state_dict' : model.state_dict(),
            'optimizer' : optim.state_dict(),
        }
        torch.save(state, savepth)

        model.requires_grad=False
        trainboth(1, trainloader, valloader, model, model2, optim2, celoss, savepth2, device, log, tdata, vdata, dset=dset)

if ctype == 'preall': 
    nc=10
    if dset == 'cifar':
        data_dir = 'cifar/train'
        mot = 'vitb'
        tdata = ci3data(data_dir, data, mod=mot)

    elif dset == 'celeb':
        pth1 = 'datacel/celeba/list_attr_celeba.csv'
        df = pd.read_csv(pth1)

        data_dir = 'datacel'
        mot = 'vitb'
        tdata = ce3data(data_dir, traindata, df, mod=mot)

    elif dset == 'celeb-gen':
        pth1 = 'datacel/celeba/list_attr_celeba.csv'
        df = pd.read_csv(pth1)
        nc=2
        data_dir = 'datacel'
        mot = 'vitb'
        tdata = ce3data(data_dir, traindata, df, mod=mot)

    elif dset == 'euro':
        nc=10
        data_dir = 'euro/vitb/train/'
        mot = 'vitb'
        comb = ['fgsm', 'gauss','pgd', 's&p']
        cdata = EuroDataset(data_dir+comb[0], trainlst, ttransforms, True)
        bdata = EuroDataset(data_dir+comb[1], trainlst, ttransforms, True)
        ddata = EuroDataset(data_dir+comb[2], trainlst, ttransforms, True)
        fdata = EuroDataset(data_dir+comb[3], trainlst, ttransforms, True)
        
        tdata = eu3data(traindata, cdata, bdata, ddata, fdata)

    if mt == 'r50':
        spth = 'saved_models/classifier/r50-5-0.0001.pth.tar'
        model = Resnet50(spth)
    elif mt == 'r18':
        model = Resnet18()
    elif mt == 'vitb':
        #spth = 'saved_models/classifier/vitb-5-0.0001-euro.pth.tar'
        #spth = 'saved_models/classifier/vitb-5-0.0001-celeb-gen.pth.tar'
        spth = 'saved_models/classifier/vitb-5-0.0001.pth.tar'
        model = ViTb(svpth=spth, nc=nc)
    elif mt == 'cnextb':
        spth = 'saved_models/classifier/cnextb-3-0.0001.pth.tar'
        model = Convnextb(svpth=spth)


    
    """
    svpth = 'saved_models/pretrain/cnextb-3-0.0001-mmd-all.pth.tar'
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
    """
    if torch.cuda.device_count() >= 2:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    
    tlen = len(tdata)
    if torch.cuda.device_count() >= 2:
        bs = 128
        if mt == 'vitb' or mt == 'cnextb':
            bs=64
        train_dataloader = DataLoader(tdata, batch_size=bs, num_workers = 4, shuffle=True)
    else:
        bs = 100
        train_dataloader = DataLoader(tdata, batch_size=bs, num_workers = 1, shuffle=True)

    close=False
    m2loss = MMDallLoss(bs)
    #celoss = torch.nn.CrossEntropyLoss()

    epochs = 3
    lr = 1e-4
    savepath = 'saved_models/pretrain/{}-{}-{}-{}-mmd-2nd-term.pth.tar'.format(mt, epochs, lr, dset)
    #log = 'logs/pre-{}-{}-{}.txt'.format(mt,epochs, lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        tloss = 0
        t1loss, t2loss, t3loss = 0,0,0

        ######################## Org & Modified ######################
        for img, inimg, noimg, inimg1, noimg1 in tqdm(train_dataloader):
            img, inimg, noimg, inimg1, noimg1 = img.to(device), inimg.to(device), noimg.to(device), inimg1.to(device), noimg1.to(device)
            
            img, _ = model(img)
            inimg, _ = model(inimg)
            noimg, _ = model(noimg)

            inimg1, _ = model(inimg1)
            noimg1, _ = model(noimg1)

            loss, centers = m2loss(img, inimg, noimg, inimg1, noimg1) 
            floss = loss

            optimizer.zero_grad()
            floss.backward()
            optimizer.step()
            
            tloss += floss
            t1loss += floss
        
        print("Epoch = {}, TLoss = {}".format(ep, tloss/len(train_dataloader)))
        """
        with open(log, 'a') as f:
            f.write(str(ep+1) + '\t'
            + str(tloss.item()/len(train_dataloader)) + '\n'   
            )
        """
        print(t1loss.item(), tloss.item())#, t2loss.item(), t3loss.item())
        state = {
            'epoch' : ep,
            'state_dict' : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'center': centers
        }
        torch.save(state, savepath)

if ctype == 'zip':
    import zipfile
    def zipdir(path, ziph):
        # ziph is zipfile handle
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file),
                        os.path.relpath(os.path.join(root, file),
                                        os.path.join(path, '..')))


    def zipit(dir_list, zip_name):
        zipf = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
        for dir in dir_list:
            zipdir(dir, zipf)
        zipf.close()

    sdir = 'vitb/train/'
    #dir_list = [sdir+'fgsm', sdir+'pgd', sdir+'fastfgsm', sdir+'rfgsm',sdir+'mifgsm', \
    #            sdir+'bim', sdir+'eotpgd', sdir+'upgd']
    dir_list = [sdir+'fgsm', sdir+'pgd']
    zip_name = 'cifar/vitb_train_fgsm-pgd-pair.zip'
    zipit(dir_list, zip_name)
    print('done')
