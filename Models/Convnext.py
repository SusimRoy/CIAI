import torch
import torchvision
import torchvision.models as models


class Convnextb(torch.nn.Module):
    def __init__(self, svpth=False, classes=3, feat_dim=128, nc=10, pretrained=False):
        super(Convnextb, self).__init__()

        model = torchvision.models.convnext_base(pretrained=True)
        model.classifier[2] = torch.nn.Linear(1024, nc)
       
        if svpth: 
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
        
        self.model = model
        self.model.classifier[2] = torch.nn.Linear(1024, feat_dim) #classes)

    def forward(self, x):
        x = self.model(x)
        return x, x

class DetConvnextb(torch.nn.Module):
    def __init__(self, pth, feat_dim=128):
        super(DetConvnextb, self).__init__()       
        model = Convnextb(False, pretrained = False)
        print(pth)
        state = torch.load(pth)
        try:
            model.load_state_dict(state['state_dict'])
            print("Model1 Loaded")
        except RuntimeError:
            dic = {}
            for k,v in state['state_dict'].items():
                dic[k.replace("module.", "")] = v
            model.load_state_dict(dic)
            print("Model2 Loaded")
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.linear1 = torch.nn.Linear(feat_dim, 64)
        self.linear2 = torch.nn.Linear(64,16)
        self.linear3 = torch.nn.Linear(16,5)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x, _ = self.model(x)
        x = self.relu(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x