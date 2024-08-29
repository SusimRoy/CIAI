import torchattacks as adv
from celebahq_loader import *
from classifiers.attribute_classifier import ClassifierWrapper
import torch
from tqdm import tqdm
import os
import numpy as np


def get_classifier():
    attribute = 'Male'  # `celebahq__Smiling`
    ckpt_path = 'Male/net_best.pth'
    model = ClassifierWrapper(attribute, ckpt_path=ckpt_path)
    return model

def attack_img():
    loader = load_data()
    model = get_classifier()
    cls_model = model
    attacks = [adv.BIM(model),adv.RFGSM(model)]
    cls_model = cls_model.eval()
    for i, attack in enumerate(attacks):
        if i == 0:
            path1 = 'datacel/celebahq/bim'
        else:
            path1 = 'datacel/celebahq/rfgsm'
        correct1, correct2, total = 0,0,0
        for j , (img,label,lnk) in enumerate(tqdm(loader)):
            img = img.cuda()
            label = label.cuda()
            att_img = attack(img, label)
            lnk = lnk[0].split("/")
            lnk = (lnk[2].split("."))[0] + ".pt"
            lnk = os.path.join(path1, lnk)
            torch.save(att_img, lnk)
            total += label.size(0)

            out_att = cls_model(att_img)
            out = cls_model(img)

            _, predicted = torch.max(out, 1)
            correct1 += (predicted == label).item()

            _, predicted = torch.max(out_att, 1)
            correct2 += (predicted == label).item()
        
        print(f"Accuracy_og{i}", correct1*100/total)
        print(f"Accuracy_att{i}", correct2*100/total)


    # amount = 0.004
    # s_vs_p = 0.5
    # # path1 = 'datacel/celebahq/s&p'
    # correct1, correct2, total = 0,0,0
    # for i, (img,label,lnk) in enumerate(tqdm(loader)):
    #     ch, row, col = img.squeeze().size()
    #     out = np.copy(np.array(img.squeeze().permute(1,2,0)))
    #     # salt mode
    #     num_salt = np.ceil(amount * out.size * s_vs_p)
    #     coords = [np.random.randint(0, i - 1, int(num_salt))
    #             for i in out.shape]
    #     out[coords] = 1
    #     # Pepper mode
    #     num_pepper = np.ceil(amount* out.size * (1. - s_vs_p))
    #     coords = [np.random.randint(0, i - 1, int(num_pepper))
    #             for i in out.shape]
    #     out[coords] = 0
    #     out = torch.tensor(out)
    #     att_img = out.permute(2,0,1)

    #     img = img.cuda()
    #     att_img = att_img.unsqueeze(0).cuda()
    #     label = label.cuda()

    #     out_att = cls_model(att_img)
    #     out = cls_model(img)

    #     total += label.size(0)

    #     _, predicted = torch.max(out, 1)
    #     correct1 += (predicted == label).item()

    #     _, predicted = torch.max(out_att, 1)
    #     correct2 += (predicted == label).item()
        
    # print("Accuracy_og", correct1*100/total)
    # print("Accuracy_att", correct2*100/total)


    # path1 = 'datacel/celebahq/gaussian'
    # correct1, correct2, total = 0,0,0
    # mean = 0
    # var = 0.0005
    # sigma = var**0.5
    # for i, (img, labels, lnk) in enumerate(tqdm(loader)):
    #     #im = 'image'+str(i)+'.pt'
    #     ch, row, col = img.squeeze().size()
    #     gauss = np.random.normal(mean,sigma,(row,col,ch))
    #     gauss = gauss.reshape(ch, row, col)
    #     noisy = img.squeeze() + torch.tensor(gauss)

    #     img = img.cuda()
    #     noisy = noisy.unsqueeze(0).float().cuda()
    #     labels = labels.cuda()

    #     out_att = cls_model(noisy)
    #     out = cls_model(img)

    #     total += labels.size(0)

    #     _, predicted = torch.max(out, 1)
    #     correct1 += (predicted == labels).item()

    #     _, predicted = torch.max(out_att, 1)
    #     correct2 += (predicted == labels).item()
        
    # print("Accuracy_og", correct1*100/total)
    # print("Accuracy_att", correct2*100/total)

    #     lnk = lnk[0].split("/")
    #     lnk = (lnk[2].split("."))[0] + ".pt"
    #     lnk = os.path.join(path1, lnk)
    #     torch.save(noisy, lnk)

attack_img()



## FGSM
# Clean Accuracy: 99.9
# Robust Accuracy: 0.1

##PGD
# Clean Accuracy: 99.9
# Robust Accuracy: 0.0

##BIM
# Clean Accuracy: 99.9
# Robust Accuracy: 0.0

##RFGSM
# Clean Accuracy: 99.9
# Robust Accuracy: 0.0

##S&P
# Clean Accuracy: 99.9
# Robust Accuracy: 99.8

##Gaussian
# Clean Accuracy: 99.9
# Robust Accuracy: 99.8