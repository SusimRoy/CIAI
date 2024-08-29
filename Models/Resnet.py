import torch
import torchvision
import torchvision.models as models

class Resnet18(torch.nn.Module):
    def __init__(self, classes=3, feat_dim=128, nc=10, pretrained=True):
        super(Resnet18, self).__init__()
        if pretrained:
            #self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            model = models.resnet18(pretrained=True)
        else:
            model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(512, nc)
        savepth = 'saved_models/classifier/resnet18-ep10-lr1e-4.pth.tar'
        state = torch.load(savepth)
        try:
            model.load_state_dict(state['state_dict'])
            #print("Model Loaded 1")
        except RuntimeError:
            dic = {}
            for k,v in state['state_dict'].items():
                dic[k.replace("module.", "")] = v
            model.load_state_dict(dic)
            #print("Model Loaded 2")
        self.model = model
        self.model.fc = torch.nn.Linear(512, feat_dim) #classes)
        #self.lin = torch.nn.Linear(feat_dim, nc)
        #self.relu = torh.nn.ReLU()
        #self.mlp1 = torch.nn.Linear(feat_dim,1)
        #self.mlp2 = torch.nn.Linear(feat_dim,1)

    def forward(self, x):
        x = self.model(x)
        #cl = self.lin(x)
        #clean = self.mlp1(x)
        #intent = self.mlp2(x)
        return x, x#, clean, intent

class DetResnet18(torch.nn.Module):
    def __init__(self, pth, feat_dim=128):
        super(DetResnet18, self).__init__()       
        self.model = Resnet18(pretrained = False)
        state = torch.load(pth)
        try:
            self.model.load_state_dict(state['state_dict'])
        except RuntimeError:
            dic = {}
            for k,v in state['state_dict'].items():
                dic[k.replace("module.", "")] = v
            self.model.load_state_dict(dic)
        
        #self.model.load_state_dict(state['state_dict'])

        for param in self.model.parameters():
            param.requires_grad = False
        
        self.linear1 = torch.nn.Linear(feat_dim, 64)
        self.linear2 = torch.nn.Linear(64,16)
        self.linear3 = torch.nn.Linear(16,3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.model(x)
        x = self.relu(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x

class Resnet50(torch.nn.Module):
    def __init__(self, savepth, classes=10, feat_dim=128, pretrained=True):
        super(Resnet50, self).__init__()
        model = models.resnet50(pretrained=pretrained)
        #self.model.fc = torch.nn.Linear(2048, classes)
        model.fc = torch.nn.Linear(2048, classes)
        if savepth:
            state = torch.load(savepth)
            try:
                model.load_state_dict(state['state_dict'])
                #print("Model Loaded 1")
            except RuntimeError:
                dic = {}
                for k,v in state['state_dict'].items():
                    dic[k.replace("module.", "")] = v
                model.load_state_dict(dic)
                #print("Model Loaded 2")
        self.model = model
        self.model.fc = torch.nn.Linear(2048, feat_dim) #classes)

    def forward(self, x):
        x = self.model(x)
        return x,x

class DetResnet50(torch.nn.Module):
    def __init__(self, pth, feat_dim=128):
        super(DetResnet50, self).__init__()   
        #pt = 'saved_models/classifier/r50-5-0.0001.pth.tar'    
        #self.model = Resnet50(pt, pretrained = False)

        self.model = Resnet50(False, pretrained = False)
        if pth:
            state = torch.load(pth)
            try:
                self.model.load_state_dict(state['state_dict'])
            except RuntimeError:
                dic = {}
                for k,v in state['state_dict'].items():
                    dic[k.replace("module.", "")] = v
                self.model.load_state_dict(dic)
        
        #self.model.load_state_dict(state['state_dict'])

        for param in self.model.parameters():
            param.requires_grad = False
        
        self.linear1 = torch.nn.Linear(feat_dim, 64)
        self.linear2 = torch.nn.Linear(64,32)
        self.linear3 = torch.nn.Linear(32,16)
        self.linear4 = torch.nn.Linear(16,5)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x,_ = self.model(x)
        x = self.relu(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x1 = self.relu(x)
        x = self.linear3(x1)
        x = self.relu(x)
        x = self.linear4(x)
        return x
        #return x1,x


class Det(torch.nn.Module):
    def __init__(self, feat_dim=128):
        super(Det, self).__init__()       
        self.linear1 = torch.nn.Linear(feat_dim, 64)
        self.linear2 = torch.nn.Linear(64,32)
        self.linear3 = torch.nn.Linear(32,16)
        self.linear4 = torch.nn.Linear(16,3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x1 = self.relu(x)
        x = self.linear3(x1)
        x = self.relu(x)
        x = self.linear4(x)
        return x
        