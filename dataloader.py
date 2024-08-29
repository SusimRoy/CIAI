import os
import torch
import torch.nn as nn

import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
#import torchivision

from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class EuroDataset(torch.utils.data.Dataset):
    def __init__(self, dir, trainlst, transform=None, e3=False):
        super().__init__() 
        self.label = {'AnnualCrop':0, 'Forest':1, 'HerbaceousVegetation':2,
                      'Highway':3, 'Industrial':4, 'Pasture':5,
                      'PermanentCrop':6, 'Residential':7, 'River':8, 'SeaLake':9}
        self.dir = dir
        self.trainlst = trainlst
        if transform:
            self.transform = transform
        else:
            self.transform = None
        self.e3 = e3
        
    def __len__(self):
        return len(self.trainlst)
    
    def __getitem__(self,idx):
        if not self.e3:
            pth = self.trainlst[idx]
            dr = pth.split("_")[0]
            pth = os.path.join(self.dir, dr, dr, pth)
            img = Image.open(pth)
            if self.transform:
                img = self.transform(img)
        else:
            pth = self.trainlst[idx]
            dr = pth.split("_")[0]
            pth = os.path.join(self.dir, pth[:-3]+'pt')
            img = torch.load(pth, map_location=torch.device('cpu'))
            img = img.float()

        label = self.label[dr]
        label = torch.tensor(label)
        return img, label    


class CelebDataset(torch.utils.data.Dataset):
    def __init__(self,df_1,mode,image_path,mod=None, transform=None, trainids = None, valids = None, allids = None):
        super().__init__() 
        self.mode = mode
        if self.mode == 'all':
            ids = allids
        elif self.mode == 'train':
            ids = trainids[17000:]
        elif self.mode == 'val':
            ids = trainids[:17000]
            #strtidx = 162270
            #endidx = 182637
        elif self.mode == 'test':
            ids = valids
        df1_1 = df_1[df_1.index.isin(ids)]    
        self.attr=df_1.drop(['image_id'],axis=1)
        self.attr = df_1['Male']
        #print(len(self.attr[self.attr==1]))
        self.path=image_path
        #self.image_id=df_1['image_id'][strtidx:endidx]
        self.image_id=df_1['image_id']
        self.transform=transform
        self.mod = mod
    
    def __len__(self):
        if self.mod == 'fawkes' or self.mod == 'lowkey' or self.mod == 'org1k':
            return 1000
        else:
            return self.image_id.shape[0]
    
    def __getitem__(self,idx:int):
        image_name=self.image_id.iloc[idx]
        #print(image_name)
        image=Image.open(os.path.join(self.path,image_name))
        attributes=np.asarray(self.attr.iloc[idx].T,dtype=np.float32)
        attributes[attributes==-1] = 0
        if self.mod == 'fawkes':
            iname = self.image_id.iloc[idx]
            im = iname.split(".")[0]
            try:
                image = Image.open('datacel/processed_images/'+im+'_attacked.png')
            except FileNotFoundError:
                image = Image.open('datacel/processed_images/000001_attacked.png')
        if self.mod == 'lowkey':
            iname = self.image_id.iloc[idx]
            im = iname.split(".")[0]
            #print(iname)
            image = Image.open('datacel/processed_images/'+im+'_cloaked.png')
          
        if self.transform:
            image=self.transform(image)
            #print('hey')
        #print(image,torch.tensor(attributes))
        #quit()
        if self.mod == 'gauss' or self.mod == 's&p':
            iname = self.image_id.iloc[idx]
            im = iname.split(".")[0]
            pt = 'datacel/imgs/' + self.mod +'/' + im+ '.pt'
            img = torch.load(pt, map_location=torch.device('cpu'))
            #img = self.transforms(img)
            image = img.float()

        return image,torch.tensor(attributes).long()    


class classdata(torch.utils.data.Dataset):
    def __init__(self, data_dir, testdata, mod, ty='nc', **kwargs):
        self.data_dir = data_dir
        self.data = testdata
        self.mod = mod
        if ty != 'clean':
            self.examples = os.path.join(self.data_dir,self.mod)
            try:
                self.ln = len(os.listdir(self.examples))
                print(self.ln)
            except:
                self.ln = 10000
            if mod == 'jitter':
                self.ln = 10000
        else:
            self.examples = testdata
            self.ln = len(testdata)
        #print(self.examples)
        if kwargs:
            self.mt = kwargs['mt']
            self.dset = kwargs['dset']
            if self.dset == 'celeb':
                self.ln = len(testdata)
                fs = h5py.File(self.data_dir, 'r')
                self.fset = fs['data']#[:]
                if self.mod == 'fgsm' or self.mod == 'pgd':
                    tst = 202599 - 182637
                    self.id = 102599 - tst - 1
                else:
                    self.id = 0

        self.ty = ty

        self.transforms = transforms.Compose([
            transforms.Resize((224,224)),
            #transforms.Normalize(
            #    mean=[0.485, 0.456, 0.406],
            #    std=[0.229, 0.224, 0.225])
        ])
        self.atck = ['fgsm', 'pgd', 'deepfool', 'jitter', 'rfgsm', 'mifgsm'\
                     , 'fastfgsm', 'bim', 'eotpgd', 'upgd','cw', 'pgdl2', 'one', 'fab' ]
        #self.atck = []

    def __len__(self):
        return self.ln
        
    def __getitem__(self,idx):
        img,olb = self.data[idx]
        if self.ty == 'clean':
            img = self.transforms(img)

        else:
            oimg = self.transforms(img)
            if self.dset == 'cifar':
                if self.mod in self.atck and self.mt == 'vitb':
                    if 'train' in self.examples:
                        self.examples = 'cifar/vitb/train/' + self.mod
                    elif 'test' in self.examples:
                        self.examples = 'cifar/vitb/test/' + self.mod
                pth = self.examples + '/image' + str(idx) + '.pt'
                img = torch.load(pth, map_location=torch.device('cpu'))
                img = self.transforms(img)
                img = img.float()
            elif self.dset == 'celeb':
                img = torch.tensor(self.fset[self.id+idx].reshape(3,224,224))
                #img = self.transforms(img)
                #print(img.size())
                #quit()

            #print(img.size(), iimg.size(), niimg.size(), iimg1.size(), niimg1.size())
            
        return img, olb
   

class detdata(torch.utils.data.Dataset):
    def __init__(self, data_dir, testdata, mod, df, ty='nc',dset='cifar', **kwargs):
        self.data_dir = data_dir
        self.data = testdata
        self.mod = mod
        self.dset = dset
        if dset == 'cifar':
            self.image_id = testdata
        else:
            self.image_id = df['image_id']
        if ty != 'clean':
            self.examples = os.path.join(self.data_dir,self.mod)
            self.ln = 60000 #len(os.listdir(self.examples))
            if mod == 'jitter' or 'test' in self.data_dir:
                self.ln = 10000
        else:
            self.examples = testdata
            self.ln = len(testdata)
        #print(self.examples)
        if kwargs:
            self.mt = kwargs['mt']
            self.cl = kwargs['cl']
        self.ty = ty

        if self.dset == 'celeb-gen' and mod in ['fgsm', 'pgd']:
            if len(testdata) == 19962:
                fs2 = h5py.File('datacel/celeb-vitb-{}-2.hdf5'.format(mod), 'r')
                self.fset2 = fs2['data']
                self.fset2 = self.fset2[:19962]
                self.comb = 0 #100000
                self.ln = 19962
            else:
                fs1 = h5py.File('datacel/celeb-vitb-{}-1.hdf5'.format(mod), 'r')
                fs2 = h5py.File('datacel/celeb-vitb-{}-2.hdf5'.format(mod), 'r')
                self.fset1 = fs1['data']#[:]
                self.fset2 = fs2['data']
                self.comb = 100000
                self.ln = 162770
        elif self.dset == 'celeb-gen' and mod in ['fastfgsm', 'rfgsm', 'bim']:
            fs2 = h5py.File('datacel/celeb-vitb-{}.hdf5'.format(mod), 'r')
            self.fset2 = fs2['data']
            self.comb = 0 #100000
            self.ln = 19962

        self.transforms = transforms.Compose([
            transforms.Resize((224,224)),
            #transforms.Normalize(
            #    mean=[0.485, 0.456, 0.406],
            #    std=[0.229, 0.224, 0.225])
        ])
        self.atck = ['fgsm', 'deepfool', 'jitter', 'rfgsm', 'mifgsm'
                     , 'fastfgsm', 'bim', 'eotpgd', 'upgd', 
                     'cw', 'pgdl2', 'fab', 'one' ]

    def __len__(self):
        return self.ln
        
    def __getitem__(self,idx):
        #iname = self.image_id[idx]
        #im = iname.split(".")[0]
        #pt = "image" + str(idx) + '.pt'

        if self.ty == 'clean':
            img, _ = self.data[idx]
            img = self.transforms(img)
            label = torch.tensor([0])
            return img, label
        else:
            if (self.mod in self.atck or self.mod == 'pgd') and self.mt == 'vitb':
                if 'train' in self.examples:
                    self.examples = 'cifar/vitb/train/' + self.mod
                elif 'test' in self.examples:
                    self.examples = 'cifar/vitb/test/' + self.mod
            
            pth = self.examples + '/image' + str(idx) + '.pt'
            #print(pth)
            img = torch.load(pth, map_location=torch.device('cpu'))
            img = self.transforms(img)
            img = img.float()
            
            """
            if self.mod in ['fgsm', 'pgd', 'fastfgsm', 'rfgsm', 'bim']: 
                if self.ln != 19962:
                    if idx>=self.comb:
                        fset = self.fset2
                        idd = idx - self.comb 
                    else:
                        fset = self.fset1
                        idd = idx
                else:
                    fset = self.fset2
                    idd = idx

                
                img = torch.tensor(fset[idd].reshape(3,224,224))
                img = self.transforms(img)
                img = img.float()
            
            
            else:
            
                pth = self.examples +'/' + im + '.pt'
                #pth = self.examples + '/Image' + str(idx) + '.pt'
                #print(pth)
                img = torch.load(pth, map_location=torch.device('cpu'))
                img = self.transforms(img)
                img = img.float()
            
                if self.mod in ['fgsm', 'pgd', 'deepfool', 'jitter']:
                    label = torch.tensor([1])
                else:
                    label = torch.tensor([2])
        
            """    
            #if self.cl == '3cl':
            #    if self.mod in self.atck or self.mod == 'pgd':
            #        label = torch.tensor([1])
            #    elif self.mod in ['gauss', 'poisson', 'speckle', 's&p']:
            #        label = torch.tensor([2])
            if self.cl == '3cl':
                if self.mod in ['deepfool', 'jitter', 'rfgsm', 'mifgsm', 'fastfgsm', 'bim', 'eotpgd', 'upgd', 
                    'cw', 'pgdl2', 'fab', 'one', 'fgsm', 'pgd']:
                    label = torch.tensor([1])
                elif self.mod in ['gauss','s&p']:
                    label = torch.tensor([2])
            else:
                if self.mod in self.atck:
                    label = torch.tensor([1])
                elif self.mod == 'pgd':
                    label = torch.tensor([2])
                elif self.mod == 'gauss' or self.mod == 'poisson' or self.mod =='speckle':
                    label = torch.tensor([3])
                elif self.mod == 's&p':
                    label = torch.tensor([4])
        
            return img, label
   
    
class ci_data(torch.utils.data.Dataset):
    def __init__(self, data_dir, tdir, idir, nidir, **kwargs):
        #self.ddir = ddir
        self.path = data_dir
        self.idir = idir
        self.nidir = nidir
        self.examples = tdir

        self.ipth = os.path.join(self.path, idir)
        self.nipth = os.path.join(self.path, nidir)

        self.transforms = transforms.Compose([
            transforms.Resize((224,224)),
            #transforms.Normalize(
            #    mean=[0.485, 0.456, 0.406],
            #    std=[0.229, 0.224, 0.225])
            ])
        
        if kwargs:
            self.mod = kwargs['mod']
        
    def __len__(self):
        #return len(self.examples)
        return 1000

    def __getitem__(self, idx):
        oimg,label = self.examples[idx]
        #opth= os.path.join(self.pth, pt)
        pt = "image" + str(idx) + '.pt'
        
        if self.mod == 'vitb' and 'train' in self.path:
            self.ipth = os.path.join('cifar/vitb/train', self.idir)
        elif self.mod == 'vitb' and 'test' in self.path:
            self.ipth = os.path.join('cifar/vitb/test', self.idir)
            #print(self.idir, self.nidir)
            if self.idir == 'gauss' or self.idir == 's&p':
                self.ipth = os.path.join('cifar/test', self.idir)
                self.nipth = os.path.join('cifar/test', self.nidir)
        #print(self.mod, self.path, self.ipth)
        ipth = os.path.join(self.ipth, pt)
        nipth= os.path.join(self.nipth, pt)

        img = self.transforms(oimg)
        #print(img.size())
        iimg = torch.load(ipth, map_location=torch.device('cpu'))
        iimg = self.transforms(iimg)
        iimg = iimg.float()

        niimg = torch.load(nipth, map_location=torch.device('cpu'))
        niimg = self.transforms(niimg)
        niimg = niimg.float()
        
        return img, iimg, niimg, label
    

class ci3data(torch.utils.data.Dataset):
    def __init__(self, data_dir, tdir, **kwargs):
        #self.ddir = ddir
        self.path = data_dir
        self.examples = tdir

        self.iptha = os.path.join(self.path, 'fgsm')
        self.ipthb = os.path.join(self.path, 'pgd')
        
        self.niptha = os.path.join(self.path, 'gauss')
        self.nipthb = os.path.join(self.path, 's&p')

        self.transforms = transforms.Compose([
            transforms.Resize((224,224)),
            #transforms.Normalize(
            #    mean=[0.485, 0.456, 0.406],
            #    std=[0.229, 0.224, 0.225])
            ])
        
        if kwargs:
            self.mod = kwargs['mod']
        
    def __len__(self):
        return len(self.examples)
        #return 1000

    def __getitem__(self, idx):
        oimg, label = self.examples[idx]
       
        pt = "image" + str(idx) + '.pt'
        if self.mod == 'vitb' and 'train' in self.path:
            self.iptha = os.path.join('cifar/vitb/train/fgsm')
            self.ipthb = os.path.join('cifar/vitb/train/pgd')

            #self.niptha = os.path.join('cifar/vitb/train/fgsm')
            #self.nipthb = os.path.join('cifar/vitb/train/pgd')
        elif self.mod == 'vitb' and 'test' in self.path:
            self.iptha = os.path.join('cifar/vitb/test/fgsm')
            self.ipthb = os.path.join('cifar/vitb/test/pgd')

        iptha = os.path.join(self.iptha, pt)
        ipthb = os.path.join(self.ipthb, pt)

        niptha= os.path.join(self.niptha, pt)
        nipthb = os.path.join(self.nipthb, pt)

        img = self.transforms(oimg)
       
        iimg = torch.load(iptha, map_location=torch.device('cpu'))
        iimg = self.transforms(iimg)
        iimg = iimg.float()

        niimg = torch.load(niptha, map_location=torch.device('cpu'))
        niimg = self.transforms(niimg)
        niimg = niimg.float()

        iimg1 = torch.load(ipthb, map_location=torch.device('cpu'))
        iimg1 = self.transforms(iimg1)
        iimg1 = iimg1.float()

        niimg1 = torch.load(nipthb, map_location=torch.device('cpu'))
        niimg1 = self.transforms(niimg1)
        niimg1 = niimg1.float()
        
        return img, iimg, niimg, iimg1, niimg1


class ce3data(torch.utils.data.Dataset):
    def __init__(self, data_dir, tdir, df,**kwargs):
        #self.ddir = ddir
        self.path = data_dir
        self.examples = tdir

        self.strtidx = 0
        self.endidx = 202599
        self.tidx = 162770
        self.comb = 100000

        self.image_id=df['image_id']
        
        self.iptha1 = os.path.join(self.path, 'celeb-vitb-fgsm-1.hdf5')
        self.iptha2 = os.path.join(self.path, 'celeb-vitb-fgsm-2.hdf5')

        fs = h5py.File(self.iptha1, 'r')
        self.fset1 = fs['data']#[:]
        fs = h5py.File(self.iptha2, 'r')
        self.fset2 = fs['data']#[:]
        #fset = np.concatenate((fset1, fset2))
        #self.fset = fset[:self.tidx]

        self.ipthb1 = os.path.join(self.path, 'celeb-vitb-pgd-1.hdf5')
        self.ipthb2 = os.path.join(self.path, 'celeb-vitb-pgd-2.hdf5')
        
        fs = h5py.File(self.ipthb1, 'r')
        self.pset1 = fs['data']#[:]
        fs = h5py.File(self.ipthb2, 'r')
        self.pset2 = fs['data']#[:]
        #pset = np.concatenate((pset1, pset2))
        #self.pset = pset[:self.tidx]

        self.niptha = os.path.join(self.path, 'imgs/gauss')
        self.nipthb = os.path.join(self.path, 'imgs/s&p')

        self.niptha = 'datacel/processed_images'
        self.nipthb = 'datacel/processed_images'

        self.transforms = transforms.Compose([
            transforms.Resize((224,224)),
            #transforms.Normalize(
            #    mean=[0.485, 0.456, 0.406],
            #    std=[0.229, 0.224, 0.225])
            ])
        
        self.ttransforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
            #    mean=[0.485, 0.456, 0.406],
            #    std=[0.229, 0.224, 0.225])
            ])
        
        if kwargs:
            self.mod = kwargs['mod']
        
    def __len__(self):
        return self.tidx
        #return 1000

    def __getitem__(self, idx):
        oimg, label = self.examples[idx]

        iname = self.image_id[idx]
        im = iname.split(".")[0]
    
        if idx>=self.comb:
            fset = self.fset2
            pset = self.pset2
            idd = idx - self.comb 
        else:
            fset = self.fset1
            pset = self.pset1
            idd = idx

        img = self.ttransforms(oimg)
       
        iimg = torch.tensor(fset[idd].reshape(3,224,224))
        iimg = self.transforms(iimg)
        iimg = iimg.float()

        iimg1 = torch.tensor(pset[idd].reshape(3,224,224))
        iimg1 = self.transforms(iimg1)
        iimg1 = iimg1.float()
        """
        pt = im + '.pt'
        niptha= os.path.join(self.niptha, pt)
        nipthb = os.path.join(self.nipthb, pt)

        niimg = torch.load(niptha, map_location=torch.device('cpu'))
        niimg = self.transforms(niimg)
        niimg = niimg.float()

        niimg1 = torch.load(nipthb, map_location=torch.device('cpu'))
        niimg1 = self.transforms(niimg1)
        niimg1 = niimg1.float()
        """

        pt = im + '_cloaked.png'
        niptha= os.path.join(self.niptha, pt)
        niimg = Image.open(niptha)
        niimg = self.ttransforms(niimg)

        try:
            pt = im + '_attacked.png'
            nipthb = os.path.join(self.nipthb, pt)
            niimg1 = Image.open(nipthb)
        except FileNotFoundError:
            pt = im + '_cloaked.png'
            nipthb = os.path.join(self.nipthb, pt)
            niimg1 = Image.open(nipthb)
        niimg1 = self.ttransforms(niimg1)
        
        #print(img.size(), iimg.size(), niimg.size(), iimg1.size(), niimg1.size())
        """
        import matplotlib.pyplot as plt
        plt.imshow(img.permute(1,2,0))
        plt.savefig('img1.png')

        plt.imshow(iimg.permute(1,2,0))
        plt.savefig('img2.png')

        plt.imshow(niimg.permute(1,2,0))
        plt.savefig('img3.png')

        plt.imshow(iimg1.permute(1,2,0))
        plt.savefig('img4.png')

        plt.imshow(niimg1.permute(1,2,0))
        plt.savefig('img5.png')

        quit()
        """
        #print(img, iimg, niimg, iimg1, niimg1)
        return img, iimg, niimg, iimg1, niimg1
    

class ce_data(torch.utils.data.Dataset):
    def __init__(self, data_dir, tdir, idir, nidir, **kwargs):
        #self.ddir = ddir
        self.path = data_dir
        self.idir = idir
        self.nidir = nidir
        self.examples = tdir

        self.ipth = os.path.join(self.path, 'celeb-vitb-{}-2.hdf5'.format(idir))
        fs = h5py.File(self.ipth, 'r')
        self.pset = fs['data']#[:]

        tst = 202599 - 182637
        self.id = 102599 - tst - 1
        self.gid = 202599 - tst - 1

        self.nipth = os.path.join(self.path, 'imgs', nidir)

        self.transforms = transforms.Compose([
            transforms.Resize((224,224)),
            #transforms.Normalize(
            #    mean=[0.485, 0.456, 0.406],
            #    std=[0.229, 0.224, 0.225])
            ])
        
        if kwargs:
            self.mod = kwargs['mod']
            self.image_id = kwargs['df']['image_id']
        
    def __len__(self):
        #return len(self.examples)
        return 1000

    def __getitem__(self, idx):
        oimg,label = self.examples[idx]
        #opth= os.path.join(self.pth, pt)
        iname = self.image_id[self.gid + idx]
        im = iname.split(".")[0]
        pt = im + '.pt'
        nipth= os.path.join(self.nipth, pt)

        img = self.transforms(oimg)
        #print(img.size())

        iimg = torch.tensor(self.pset[self.id+idx].reshape(3,224,224))
        iimg = self.transforms(iimg)
        iimg = iimg.float()

        niimg = torch.load(nipth, map_location=torch.device('cpu'))
        niimg = self.transforms(niimg)
        niimg = niimg.float()

        """
        print(img.size(), iimg.size(), niimg.size(), iimg1.size(), niimg1.size())
        
        import matplotlib.pyplot as plt
        plt.imshow(img.permute(1,2,0))
        plt.savefig('img1.png')

        plt.imshow(iimg.permute(1,2,0))
        plt.savefig('img2.png')

        plt.imshow(niimg.permute(1,2,0))
        plt.savefig('img3.png')
        
        quit()
        """
        return img, iimg, niimg, label
    

class detCordata(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, sev):
        self.data_dir = data_dir
        self.label_dir = label_dir
        
        self.sev = sev
        self.stridx = sev * 10000
        self.endidx = self.stridx + 10000

        self.examples = np.load(self.data_dir)
        #print(self.stridx, self.endidx)
        try:
            self.labels = np.load(self.label_dir)
            self.labels = self.labels[self.stridx:self.endidx]
        except AttributeError:
            self.labels = 0
            
        self.examples = self.examples[self.stridx:self.endidx]
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            #transforms.Normalize(
            #    mean=[0.485, 0.456, 0.406],
            #    std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return 10000
        
    def __getitem__(self,idx):
        #print(len(self.examples))
        img = self.examples[idx]
        img = self.transforms(img)
        #img = img.float()

        try:
            label = self.labels[idx]
        except:
            label = torch.tensor([2])
            
        return img, label
       

class eu3data(torch.utils.data.Dataset):
    def __init__(self, tdir, data1, data2, data3, data4):
        #self.ddir = ddir
        self.examples = tdir

        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
        self.data4 = data4
        
    def __len__(self):
        return len(self.examples)
        #return 1000

    def __getitem__(self, idx):
        img, label = self.examples[idx]
        iimg, _ = self.data1[idx]
        niimg, _ = self.data2[idx]
        iimg1, _ = self.data3[idx]
        niimg1, _ = self.data4[idx]

        return img, iimg, niimg, iimg1, niimg1
    

class euDetdata(torch.utils.data.Dataset):
    def __init__(self, tdir, data1, ty):
        #self.ddir = ddir
        self.examples = tdir
        self.data1 = data1
        self.ty = ty
        
    def __len__(self):
        return len(self.examples)
        #return 1000

    def __getitem__(self, idx):
        if self.ty == 'clean':
            img, _ = self.examples[idx]
            label = torch.tensor([0])
        elif self.ty in ['fgsm', 'pgd']:
            img, _ = self.data1[idx]
            label = torch.tensor([1])
        elif self.ty in ['gauss', 's&p']:
            img, _ = self.data1[idx]
            label = torch.tensor([2])

        return img, label
    

class CIFAR100DataLoader:
    def __init__(self, batch_size=64, num_workers=4):
        self.batch_size = batch_size
        self.num_workers = num_workers

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])

        self.train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        self.test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    def get_train_loader(self):
        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        return train_loader

    def get_test_loader(self):
        test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        return test_loader