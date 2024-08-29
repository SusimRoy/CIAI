import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def MMD(x, y, kernel = 'multiscale'):
    #print(x.size(), y.size())
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx 
    dyy = ry.t() + ry - 2. * yy 
    dxy = rx.t() + ry - 2. * zz 

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

    return torch.mean(XX + YY - 2. * XY)

class MMDLoss(nn.Module):
    def __init__(self, bs, num_classes=2, feat_dim=128, use_gpu=True):
        super(MMDLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.bs = bs
        #self.device = device
        centers = torch.randn(3, 1, self.feat_dim)
        #print(centers.size(), centers[0].size())
        centers = torch.repeat_interleave(centers, bs, dim=1)

        if self.use_gpu:
            self.centers = nn.Parameter(centers).cuda()
        else:
            self.centers = nn.Parameter(centers)

    def forward(self, x, nx, ni):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        bs = x.size(0)
        #csize = self.centers.size()
        #self.centers = self.centers.view(csize[0], csize[1], -1)
        c1 = self.centers[0][:bs]
        c2 = self.centers[1][:bs]
        c3 = self.centers[2][:bs]
        
        """
        x = x.view(x.size(0), -1) 
        nx = nx.view(nx.size(0), -1) 
        ni = ni.view(ni.size(0), -1) 
        print(x.size(), c1.size())
        """ 

        posorg = MMD(c1,x)
        posnot = MMD(c2,nx)
        posint = MMD(c3,ni)
        """
        negon1 = MMD(c1,nx)
        negon2 = MMD(c1,ni)
        negoi1 = MMD(c2,x)
        negoi2 = MMD(c2,ni)
        negni1 = MMD(c3,x)
        negni2 = MMD(c3,nx)
        """
        negon1 = MMD(x,nx) #MMD(c1,nx)
        negoi1 = MMD(x,ni) #MMD(c2,x)
        negni1 = MMD(nx,ni) #MMD(c3,x)
        
        dist = 1 * (posorg + posnot + posint) - 0.7 * (negon1 + negoi1 + negni1) # + negon2 + negoi2 + negni2)
        #dist = 1 * (posorg + posnot + posint) - 0.1 * (negon1 + negoi1 + negni1 + negon2 + negoi2 + negni2)
        if self.use_gpu:
            loss = torch.clamp_min(dist, torch.tensor(0).cuda()) 
        else:   
            loss = torch.clamp_min(dist, torch.tensor(0)) 
        return loss, self.centers

class MMDCombLoss(nn.Module):
    def __init__(self, bs, num_classes=2, feat_dim=128, use_gpu=True):
        super(MMDCombLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.bs = bs


    def forward(self, x, nx, ni, x1, nx1, ni1):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        posorg = MMD(x,x1)
        posnot = MMD(nx,nx1)
        posint = MMD(ni,ni1)
        
        xnx1 = MMD(x,nx1) #MMD(c1,nx)
        xni1 = MMD(x,ni1) #MMD(c2,x)
        nxni1 = MMD(nx,ni1) #MMD(c3,x)
        nxx1 = MMD(nx,x1)
        ninx1 = MMD(ni,nx1)
        nix1 = MMD(ni,x1)
        
        dist = 1 * (posorg + posnot + posint) - 0.2 * (xnx1+xni1+nxni1+nxx1+ninx1+nix1) # + negon2 + negoi2 + negni2)
        if self.use_gpu:
            loss = torch.clamp_min(dist, torch.tensor(0).cuda()) 
        else:   
            loss = torch.clamp_min(dist, torch.tensor(0)) 
        return loss
    
class MMD2Loss(nn.Module):
    def __init__(self, bs, close = False, num_classes=2, feat_dim=128, use_gpu=True):
        super(MMD2Loss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.bs = bs
        #self.device = device
        centers = torch.randn(2, 1, self.feat_dim)
        #print(centers.size(), centers[0].size())
        centers = torch.repeat_interleave(centers, bs, dim=1)
        self.close = close

        if self.use_gpu:
            self.centers = nn.Parameter(centers).cuda()
        else:
            self.centers = nn.Parameter(centers)

    def forward(self, x, nx, ni):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        bs = x.size(0)
        
        c1 = self.centers[0][:bs]
        c2 = self.centers[1][:bs]
        
        pos1 = MMD(c1,x)
        pos2 = MMD(c2,nx)
        pos3 = MMD(c2,ni)

        neg1 = MMD(c2,x)
        neg2 = MMD(c1,nx)
        neg3 = MMD(c1,ni)
        
        if self.close:
            dist = 1 * (pos1 + pos2 + pos3)
        else:
            dist = 1 * (pos1 + pos2 + pos3) - 0.6 * (neg1 + neg2 + neg3)
        
        if self.use_gpu:
            loss = torch.clamp_min(dist, torch.tensor(0).cuda()) 
        else:   
            loss = torch.clamp_min(dist, torch.tensor(0)) 

        return loss
    
class MMD2aLoss(nn.Module):
    def __init__(self, bs, close=False, num_classes=2, feat_dim=128, use_gpu=True):
        super(MMD2aLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.bs = bs
        #self.device = device
        centers = torch.randn(2, 1, self.feat_dim)
        centers = torch.repeat_interleave(centers, bs, dim=1)
        self.close = close

        if self.use_gpu:
            self.centers = nn.Parameter(centers).cuda()
        else:
            self.centers = nn.Parameter(centers)

    def forward(self, nx, ni):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        bs = nx.size(0)
        
        c1 = self.centers[0][:bs]
        c2 = self.centers[1][:bs]
        
        pos1 = MMD(c1,nx)
        pos2 = MMD(c2,ni)

        neg1 = MMD(c2,nx)
        neg2 = MMD(c1,ni)
        
        if self.close:
            dist = 1 * (pos1 + pos2)
        else:
            dist = 1 * (pos1 + pos2) - 0.6 * (neg1 + neg2)
        
        if self.use_gpu:
            loss = torch.clamp_min(dist, torch.tensor(0).cuda()) 
        else:   
            loss = torch.clamp_min(dist, torch.tensor(0)) 
            
        return loss
    
class MMD3Loss(nn.Module):
    def __init__(self, bs, close=False, num_classes=2, feat_dim=128, use_gpu=True):
        super(MMD3Loss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.bs = bs
        #self.device = device
        centersa = torch.randn(2, 1, self.feat_dim)
        centersa = torch.repeat_interleave(centersa, bs, dim=1)

        centersb = torch.randn(2, 1, self.feat_dim)
        centersb = torch.repeat_interleave(centersb, bs, dim=1)
        self.close = close

        if self.use_gpu:
            self.centersa = nn.Parameter(centersa).cuda()
            self.centersb = nn.Parameter(centersb).cuda()
        else:
            self.centersa = nn.Parameter(centersa)
            self.centersb = nn.Parameter(centersb)

    def forward(self, nx, ni, nx1, ni1):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        bs = nx.size(0)
        
        c1 = self.centersa[0][:bs]
        c2 = self.centersa[1][:bs]
        c3 = self.centersb[0][:bs]
        c4 = self.centersb[1][:bs]
        
        pos1 = MMD(c1,nx)
        pos2 = MMD(c2,nx1)
        neg1 = MMD(c2,nx)
        neg2 = MMD(c1,nx1)
        
        if self.close:
            dista = 1 * (pos1 + pos2)
        else:
            dista = 1 * (pos1 + pos2) - 0.6 * (neg1 + neg2)

        pos1 = MMD(c3,ni)
        pos2 = MMD(c4,ni1)
        neg1 = MMD(c4,ni)
        neg2 = MMD(c3,ni1)
        
        if self.close:
            distb = 1 * (pos1 + pos2)
        else:
            distb = 1 * (pos1 + pos2) - 0.6 * (neg1 + neg2)

        dist = dista + distb
        if self.use_gpu:
            loss = torch.clamp_min(dist, torch.tensor(0).cuda()) 
        else:   
            loss = torch.clamp_min(dist, torch.tensor(0))             
        return loss

class MMDallLoss(nn.Module):
    def __init__(self, bs, close = False, num_classes=2, feat_dim=128, alpha=0.3, use_gpu=True):
        super(MMDallLoss, self).__init__()
        #self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.bs = bs
        #self.device = device
        centers = torch.randn(5, 1, self.feat_dim)
        #print(centers.size(), centers[0].size())
        centers = torch.repeat_interleave(centers, bs, dim=1)
        self.close = close
        self.alpha = alpha

        if self.use_gpu:
            self.centers = nn.Parameter(centers).cuda()
        else:
            self.centers = nn.Parameter(centers)

    def forward(self, x, nx, ni, nx1, ni1):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        bs = x.size(0)
        
        c1 = self.centers[0][:bs]
        c2 = self.centers[1][:bs]
        c3 = self.centers[2][:bs]
        c4 = self.centers[3][:bs]
        c5 = self.centers[4][:bs]
        
        ####### ORIGINAL & MOD #########
        pos1 = MMD(c1,x)
        pos2 = MMD(c2,nx)
        pos3 = MMD(c3,nx1)
        pos4 = MMD(c4,ni)
        pos5 = MMD(c5,ni1)

        ng1 = MMD(c2,x)
        ng2 = MMD(c3,x)
        ng3 = MMD(c4,x)
        ng4 = MMD(c5,x)

        neg2 = MMD(c1,nx)
        neg3 = MMD(c1,ni)
        neg4 = MMD(c1,nx1)
        neg5 = MMD(c1,ni1)
        
        dist1 = 1 * (pos1 + pos2 + pos3 + pos4 + pos5) - self.alpha * (ng1 + ng2 + ng3 + ng4 \
                                                                       + neg2 + neg3 + neg4 + neg5)

        ####### INT & NON-INT #########
        pos1 = MMD(c2,nx)
        pos2 = MMD(c4,ni)

        neg1 = MMD(c4,nx)
        neg2 = MMD(c2,ni)

        pos3 = MMD(c3,nx1)
        pos4 = MMD(c5,ni1)

        neg3 = MMD(c5,nx1)
        neg4 = MMD(c3,ni1)
        
        dist2 = 1 * (pos1 + pos2 + pos3 + pos4) - self.alpha * (neg1 + neg2 + neg3 + neg4)

        ############ INTERNAL INT & NON-INT ##########
        pos1 = MMD(c2,nx)
        pos2 = MMD(c3,nx1)
        neg1 = MMD(c3,nx)
        neg2 = MMD(c2,nx1)
        
        pos3 = MMD(c4,ni)
        pos4 = MMD(c5,ni1)
        neg3 = MMD(c5,ni)
        neg4 = MMD(c4,ni1)

        dist3 = 1 * (pos1 + pos2 + pos3 + pos4) - self.alpha * (neg1 + neg2 + neg3 + neg4)
        #dist = 0.8 *  dist1 + 0.2 * dist2 #+ 0.1 * dist3
        #dist = dist1 + 
        dist = dist1  + dist3

        if self.use_gpu:
            loss = torch.clamp_min(dist, torch.tensor(0).cuda()) 
        else:   
            loss = torch.clamp_min(dist, torch.tensor(0)) 

        return loss, self.centers