import torch
import torch.nn as nn

class MLoss(nn.Module):
    def __init__(self, bs, num_classes=2, feat_dim=128, use_gpu=True):
        super(MLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        #self.device = device

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(3 , 1, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(3 , 1, self.feat_dim))

    def forward(self, x, nx, ni):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(self.centers[0], 2).sum(dim=1, keepdim=True)         
        distmat.addmm_(x, self.centers[0].t(), beta=1, alpha=-2)

        ndistmat = torch.pow(nx, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(self.centers[1], 2).sum(dim=1, keepdim=True)
        ndistmat.addmm_(nx, self.centers[1].t(), beta=1, alpha=-2)

        idistmat = torch.pow(ni, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(self.centers[2], 2).sum(dim=1, keepdim=True)
        idistmat.addmm_(ni, self.centers[2].t(), beta=1, alpha=-2)
        
        """
        gdistmat1 = torch.pow(nx, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(self.centers[0], 2).sum(dim=1, keepdim=True)
        gdistmat1.addmm_(nx, self.centers[0].t(), beta=1, alpha=-2)

        gdistmat2 = torch.pow(ni, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(self.centers[0], 2).sum(dim=1, keepdim=True)
        gdistmat2.addmm_(ni, self.centers[0].t(), beta=1, alpha=-2)

        gdistmat = torch.pow(nx, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(self.centers[2], 2).sum(dim=1, keepdim=True)
        gdistmat.addmm_(nx, self.centers[2].t(), beta=1, alpha=-2)
        
        oidistmat = torch.pow(x, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(self.centers[2], 2).sum(dim=1, keepdim=True)
        oidistmat.addmm_(x, self.centers[2].t(), beta=1, alpha=-2)

        ondistmat = torch.pow(x, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(self.centers[1], 2).sum(dim=1, keepdim=True)
        ondistmat.addmm_(x, self.centers[1].t(), beta=1, alpha=-2)

        gndistmat = torch.pow(ni, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(self.centers[1], 2).sum(dim=1, keepdim=True)
        gndistmat.addmm_(ni, self.centers[1].t(), beta=1, alpha=-2)
        """
        gdistmat = torch.pow(x, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(nx, 2).sum(dim=1, keepdim=True)
        #gdistmat.addmm_(x, nx.t(), beta=1, alpha=-2)
        gdistmat = x*nx + gdistmat
        
        oidistmat = torch.pow(x, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(ni, 2).sum(dim=1, keepdim=True)
        #oidistmat.addmm_(x, ni.t(), beta=1, alpha=-2)
        oidistmat = x*ni + oidistmat

        ondistmat = torch.pow(ni, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(nx, 2).sum(dim=1, keepdim=True)
        #ondistmat.addmm_(ni, nx.t(), beta=1, alpha=-2)
        ondistmat = ni*nx + ondistmat
        
        dist = 1 * (distmat + ndistmat + idistmat) - 0.2*(gdistmat + oidistmat + ondistmat)
        #print((distmat + ndistmat + idistmat) - 0.2*(gdistmat + oidistmat + ondistmat))
        #dist = 1 * (distmat + ndistmat + idistmat) -  0.4*(gdistmat + oidistmat + ondistmat + gdistmat1 + gdistmat2 + gndistmat )
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        
        return loss, self.centers


class NLoss(nn.Module):
    def __init__(self, num_classes=2, feat_dim=128, use_gpu=True):
        super(NLoss, self).__init__()
        #self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        #self.device = device


    def forward(self, x, nx, ni):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        gdistmat = torch.pow(x, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(nx, 2).sum(dim=1, keepdim=True)
        #gdistmat.addmm_(x, nx.t(), beta=1, alpha=-2)
        gdistmat = x*nx + gdistmat
        
        oidistmat = torch.pow(x, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(ni, 2).sum(dim=1, keepdim=True)
        #oidistmat.addmm_(x, ni.t(), beta=1, alpha=-2)
        oidistmat = x*ni + oidistmat

        ondistmat = torch.pow(ni, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(nx, 2).sum(dim=1, keepdim=True)
        #ondistmat.addmm_(ni, nx.t(), beta=1, alpha=-2)
        ondistmat = ni*nx + ondistmat
    
        #dist = 0.6 * (distmat + ndistmat + idistmat) + 0.1 * gdistmat - 0.3 * (oidistmat + ondistmat)
        dist = 0.0001 * (gdistmat + oidistmat + ondistmat)
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        
        return loss


class SepLoss(nn.Module):
    def __init__(self, num_classes=2, feat_dim=128, use_gpu=True):
        super(SepLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        #self.device = device

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(2, 1, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(2, 1, self.feat_dim))

    def forward(self, x, nx):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(self.centers[0], 2).sum(dim=1, keepdim=True)         
        distmat.addmm_(x, self.centers[0].t(), beta=1, alpha=-2)

        ndistmat = torch.pow(nx, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(self.centers[1], 2).sum(dim=1, keepdim=True)
        ndistmat.addmm_(nx, self.centers[1].t(), beta=1, alpha=-2)

        
        dist = distmat + ndistmat 
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        
        return loss, self.centers
    

class MallLoss(nn.Module):
    def __init__(self, bs, close = False, num_classes=2, feat_dim=128, alpha=0.3, use_gpu=True):
        super(MallLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.bs = bs
        #self.device = device
        centers = torch.randn(5, 1, self.feat_dim)
        #print(centers.size(), centers[0].size())
        #centers = torch.repeat_interleave(centers, bs, dim=1)
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
        
        c1 = self.centers[0]#[:bs]
        c2 = self.centers[1]#[:bs]
        c3 = self.centers[2]#[:bs]
        c4 = self.centers[3]#[:bs]
        c5 = self.centers[4]#[:bs]
        
        ####### ORIGINAL & MOD #########
        pos1 = torch.pow(c1, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(x, 2).sum(dim=1, keepdim=True)         
        pos1.addmm_(x, c1.t(), beta=1, alpha=-2)

        pos2 = torch.pow(c2, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(nx, 2).sum(dim=1, keepdim=True)         
        pos2.addmm_(nx, c2.t(), beta=1, alpha=-2)

        pos3 = torch.pow(c3, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(nx1, 2).sum(dim=1, keepdim=True)         
        pos3.addmm_(nx1, c3.t(), beta=1, alpha=-2)

        pos4 = torch.pow(c4, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(ni, 2).sum(dim=1, keepdim=True)         
        pos4.addmm_(ni, c4.t(), beta=1, alpha=-2)

        pos5 = torch.pow(c5, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(ni1, 2).sum(dim=1, keepdim=True)         
        pos5.addmm_(ni1, c5.t(), beta=1, alpha=-2)


        ng1 = torch.pow(c2, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(x, 2).sum(dim=1, keepdim=True)         
        ng1.addmm_(x, c2.t(), beta=1, alpha=-2)

        ng2 = torch.pow(c3, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(x, 2).sum(dim=1, keepdim=True)         
        ng2.addmm_(x, c3.t(), beta=1, alpha=-2)

        ng3 = torch.pow(c4, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(x, 2).sum(dim=1, keepdim=True)         
        ng3.addmm_(x, c4.t(), beta=1, alpha=-2)

        ng4 = torch.pow(c5, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(x, 2).sum(dim=1, keepdim=True)         
        ng4.addmm_(x, c5.t(), beta=1, alpha=-2)


        neg2 = torch.pow(c1, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(nx, 2).sum(dim=1, keepdim=True)         
        neg2.addmm_(nx, c1.t(), beta=1, alpha=-2)

        neg3 = torch.pow(c1, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(ni, 2).sum(dim=1, keepdim=True)         
        neg3.addmm_(ni, c1.t(), beta=1, alpha=-2)

        neg4 = torch.pow(c1, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(nx1, 2).sum(dim=1, keepdim=True)         
        neg4.addmm_(nx1, c1.t(), beta=1, alpha=-2)

        neg5 = torch.pow(c1, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(ni1, 2).sum(dim=1, keepdim=True)         
        neg5.addmm_(ni1, c1.t(), beta=1, alpha=-2)
        
        dist1 = 1 * (pos1 + pos2 + pos3 + pos4 + pos5) - self.alpha * (ng1 + ng2 + ng3 + ng4 \
                                                                       + neg2 + neg3 + neg4 + neg5)

        ####### INT & NON-INT #########
        pos1 = torch.pow(c2, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(nx, 2).sum(dim=1, keepdim=True)         
        pos1.addmm_(nx, c2.t(), beta=1, alpha=-2)

        pos2 = torch.pow(c4, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(ni, 2).sum(dim=1, keepdim=True)         
        pos2.addmm_(ni, c4.t(), beta=1, alpha=-2)

        pos3 = torch.pow(c3, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(nx1, 2).sum(dim=1, keepdim=True)         
        pos3.addmm_(nx1, c3.t(), beta=1, alpha=-2)

        pos4 = torch.pow(c5, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(ni1, 2).sum(dim=1, keepdim=True)         
        pos4.addmm_(ni1, c5.t(), beta=1, alpha=-2)

        neg1 = torch.pow(c4, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(nx, 2).sum(dim=1, keepdim=True)         
        neg1.addmm_(nx, c4.t(), beta=1, alpha=-2)

        neg2 = torch.pow(c2, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(ni, 2).sum(dim=1, keepdim=True)         
        neg2.addmm_(ni, c2.t(), beta=1, alpha=-2)

        neg3 = torch.pow(c5, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(nx1, 2).sum(dim=1, keepdim=True)         
        neg3.addmm_(nx1, c5.t(), beta=1, alpha=-2)

        neg4 = torch.pow(c3, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(ni1, 2).sum(dim=1, keepdim=True)         
        neg4.addmm_(ni1, c3.t(), beta=1, alpha=-2)

        
        dist2 = 1 * (pos1 + pos2 + pos3 + pos4) - self.alpha * (neg1 + neg2 + neg3 + neg4)

        ############ INTERNAL INT & NON-INT ##########
        pos1 = torch.pow(c2, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(nx, 2).sum(dim=1, keepdim=True)         
        pos1.addmm_(nx, c2.t(), beta=1, alpha=-2)

        pos2 = torch.pow(c3, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(nx1, 2).sum(dim=1, keepdim=True)         
        pos2.addmm_(nx1, c3.t(), beta=1, alpha=-2)

        pos3 = torch.pow(c4, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(ni, 2).sum(dim=1, keepdim=True)         
        pos3.addmm_(ni, c4.t(), beta=1, alpha=-2)

        pos4 = torch.pow(c5, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(ni1, 2).sum(dim=1, keepdim=True)         
        pos4.addmm_(ni1, c5.t(), beta=1, alpha=-2)

        neg1 = torch.pow(c3, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(nx, 2).sum(dim=1, keepdim=True)         
        neg1.addmm_(nx, c3.t(), beta=1, alpha=-2)

        neg2 = torch.pow(c2, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(nx1, 2).sum(dim=1, keepdim=True)         
        neg2.addmm_(nx1, c2.t(), beta=1, alpha=-2)

        neg3 = torch.pow(c5, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(ni, 2).sum(dim=1, keepdim=True)         
        neg3.addmm_(ni, c5.t(), beta=1, alpha=-2)

        neg4 = torch.pow(c4, 2).sum(dim=1, keepdim=True) + \
                  torch.pow(ni1, 2).sum(dim=1, keepdim=True)         
        neg4.addmm_(ni1, c4.t(), beta=1, alpha=-2)

        dist3 = 1 * (pos1 + pos2 + pos3 + pos4) - self.alpha * (neg1 + neg2 + neg3 + neg4)
        dist = dist1 + dist2 + dist3
        dist = dist.sum()  #/bs
        if self.use_gpu:
            loss = torch.clamp_min(dist, torch.tensor(0).cuda()) 
        else:   
            loss = torch.clamp_min(dist, torch.tensor(0)) 

        return loss, self.centers