import torch
from tqdm import tqdm

def train(epochs, trainloader, valloader, orgmodel, optimizer, celoss, savepth, device, log, tdata, vdata, **kwargs):
  if kwargs['dset'] == 'celeb':
    tdata = tdata * 40
    vdata = vdata * 40

  sig = torch.nn.Sigmoid()
  for epoch in range(epochs):
    ########### TRAINING ##############
    tacc = 0
    tloss = 0
    #nclst = [0,0,0]
    #tot = [0,0,0]
    for imgs, labels in tqdm(trainloader):
      imgs = imgs.to(device)
      pred = orgmodel(imgs)
      #print(imgs, labels)
      #print(pred.size(), labels.size())
      optimizer.zero_grad()
      loss = celoss(pred, labels.squeeze().to(device))
      loss.backward()
      optimizer.step()

      pred = pred.cpu()
      if kwargs['dset'] == 'celeb':
        pred = sig(pred)
        tacc+=torch.round(pred).eq(labels).sum()
        #tacc = tacc/40.0
      else:
        pred = torch.argmax(pred, axis=1)
        #print(pred)
        tacc += torch.sum(pred==labels.squeeze())
        """
        for idz in len(pred):
          if pred[idz]==labels[idz]:
            if pred[idz] == 0:
              nclst[0]+=1
            elif pred[idz] == 1:
              nclst[1]+=1
            elif pred[idz] == 2:
              nclst[2]+=1
          if label[idz] == 0:
            tot[0]+=1
          elif label[idz] == 1:
            tot[1]+=1
          elif label[idz] == 2:
            tot[2]+=1
          """
      tloss+=loss.item()

    ########### VALIDATION ##############
    vacc = 0
    vloss = 0
    for imgs, labels in tqdm(valloader):
      imgs = imgs.to(device)
      pred = orgmodel(imgs)

      loss = celoss(pred, labels.squeeze().to(device))
      
      pred = pred.cpu()
      if kwargs['dset'] == 'celeb':
        pred = sig(pred)
        vacc+=torch.round(pred).eq(labels).sum()
        #vacc=vacc/40.0
      else:
        pred = torch.argmax(pred, axis=1)
        vacc += torch.sum(pred==labels.squeeze())
      vloss+=loss.item()

    print("Epoch {} ==> Training Loss = {}, Training Accuracy = {}, \
    Validation Loss = {}, Validation Accuracy = {}".format(epoch+1, tloss/len(trainloader), tacc/tdata, vloss/len(valloader), vacc/vdata))
    """
    print(nclst/tot)
    print(nclst)
    print(tot)
    """
    
    with open(log, 'a') as f:
            f.write(str(epoch+1) + '\t'
            + str(tloss/len(trainloader)) + '\t'   # T LOSS
            + str(tacc.item()/tdata) + '\t'    # T ACC 
            + str(vloss/len(valloader)) + '\t' # V LOSS
            + str(vacc.item()/vdata) + '\n'        # V ACC
            )

    state = {
      'epoch': epoch,
      'state_dict': orgmodel.state_dict(),
      'optimizer': optimizer.state_dict(),
    }
    torch.save(state, savepth)

def test(model, testloader, tdata, device, mod, **kwargs):
    tacc = 0
    pr = []
    tpr = []
    lendata = len(tdata)
    cl = kwargs['cl']
    atck = ['deepfool', 'jitter', 'rfgsm', 'mifgsm'\
                     , 'fastfgsm', 'bim', 'eotpgd', 'upgd' ]
    
    if kwargs['dset'] == 'celeb':
      lendata = len(tdata) * 40
      sig = torch.nn.Sigmoid()
    for imgs, labels in tqdm(testloader):
        #print(labels)
        imgs = imgs.to(device)
        pred = model(imgs)
        pred = pred.cpu()
        
        if kwargs['dset'] == 'celeb':
          pred = sig(pred)
          tacc+=torch.round(pred).eq(labels).sum()
          #tacc/=40.0
        if (kwargs['dset'] == 'celeb-gen' or kwargs['dset'] == 'euro') and cl == None:
          pred = torch.argmax(pred, axis=1)
          tacc += torch.sum(pred==labels.squeeze())
        else:
          pred = torch.argmax(pred, axis=1)
          #print(pred)
          #tacc += torch.sum(pred==labels.squeeze())
          if cl == '3cl':
             tacc += torch.sum(pred==labels.squeeze())
          else:
            if mod in atck:
              labels = torch.full(labels.size(), 1)
              tacc += torch.sum(pred==labels.squeeze())
              labels = torch.full(labels.size(), 2)
              tacc += torch.sum(pred==labels.squeeze())
            elif mod in ['poisson']:
              labels = torch.full(labels.size(), 3)
              tacc += torch.sum(pred==labels.squeeze())
              labels = torch.full(labels.size(), 4)
              tacc += torch.sum(pred==labels.squeeze())
            else:
              tacc += torch.sum(pred==labels.squeeze())
             
        #print(pred)  
        if kwargs['dset'] != 'celeb':
          if len(pr) == 0:
            pr = pred#.unsqueeze(1)
            tpr = labels.cpu().squeeze()
          else:
            pr = torch.hstack((pr, pred))#.unsqueeze(1)))
            tpr = torch.hstack((tpr, labels.cpu().squeeze()))
            #print(pr.size(), pred.size())
            #quit()
        #pr.append(pred)
        #tpr.append(labels.cpu())
    print("Testing Accuracy = ", tacc.item()/lendata)
    return tpr, pr



def trainboth(epochs, trainloader, valloader, premodel, orgmodel, optimizer, celoss, savepth, device, log, tdata, vdata, **kwargs):
  if kwargs['dset'] == 'celeb':
    tdata = tdata * 40
    vdata = vdata * 40

  sig = torch.nn.Sigmoid()
  for epoch in range(epochs):
    ########### TRAINING ##############
    tacc = 0
    tloss = 0
    #nclst = [0,0,0]
    #tot = [0,0,0]
    for imgs, labels in tqdm(trainloader):
      imgs = imgs.to(device)
      imgs,_ = premodel(imgs)
      pred = orgmodel(imgs)
      #print(imgs, labels)
      #print(pred.size(), labels.size())
      optimizer.zero_grad()
      loss = celoss(pred, labels.squeeze().to(device))
      loss.backward()
      optimizer.step()

      pred = pred.cpu()
      if kwargs['dset'] == 'celeb':
        pred = sig(pred)
        tacc+=torch.round(pred).eq(labels).sum()
        #tacc = tacc/40.0
      else:
        pred = torch.argmax(pred, axis=1)
        #print(pred)
        tacc += torch.sum(pred==labels.squeeze())
      tloss+=loss.item()

    ########### VALIDATION ##############
    vacc = 0
    vloss = 0
    for imgs, labels in tqdm(valloader):
      imgs = imgs.to(device)
      imgs,_ = premodel(imgs)
      pred = orgmodel(imgs)

      loss = celoss(pred, labels.squeeze().to(device))
      
      pred = pred.cpu()
      if kwargs['dset'] == 'celeb':
        pred = sig(pred)
        vacc+=torch.round(pred).eq(labels).sum()
        #vacc=vacc/40.0
      else:
        pred = torch.argmax(pred, axis=1)
        vacc += torch.sum(pred==labels.squeeze())
      vloss+=loss.item()

    print("Epoch {} ==> Training Loss = {}, Training Accuracy = {}, \
    Validation Loss = {}, Validation Accuracy = {}".format(epoch+1, tloss/len(trainloader), tacc/tdata, vloss/len(valloader), vacc/vdata))
    
    with open(log, 'a') as f:
            f.write(str(epoch+1) + '\t'
            + str(tloss/len(trainloader)) + '\t'   # T LOSS
            + str(tacc.item()/tdata) + '\t'    # T ACC 
            + str(vloss/len(valloader)) + '\t' # V LOSS
            + str(vacc.item()/vdata) + '\n'        # V ACC
            )

    state = {
      'epoch': epoch,
      'state_dict': orgmodel.state_dict(),
      'optimizer': optimizer.state_dict(),
    }
    torch.save(state, savepth)
