import torch
import torch.nn as nn
import torch.nn.functional as F

from e2cnn import gspaces
from e2cnn import nn as e2nn

import utils


        
# -----------------------------------------------------------------------------

class VanillaLeNet(nn.Module):
    def __init__(self, in_chan, out_chan, imsize, kernel_size=5, N=None):
        super(VanillaLeNet, self).__init__()
        
        z = 0.5*(imsize - 2)
        z = int(0.5*(z - 2))
        z = 18496
        
        self.mask = utils.build_mask(imsize, margin=1)

        self.conv1 = nn.Conv2d(in_chan, 6, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size, padding=1)
        self.fc1   = nn.Linear(16*z*z, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, out_chan)
        self.drop  = nn.Dropout(p=0.5)
        
        # dummy parameter for tracking device
        self.dummy = nn.Parameter(torch.empty(0))
        
    def loss(self,p,y):
        
        # check device for model:
        device = self.dummy.device
        
        # p : softmax(x)
        loss_fnc = nn.NLLLoss().to(device=device)
        loss = loss_fnc(torch.log(p),y)
        
        return loss
     
    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        return
        
    def forward(self, x):
        
        # check device for model:
        device = self.dummy.device
        mask = self.mask.to(device=device)
        
        x = x*mask
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = x.view(x.size()[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
    
        return x

# -----------------------------------------------------------------------------

class CNSteerableLeNet(nn.Module):
    def __init__(self, in_chan, out_chan, imsize, kernel_size=5, N=8):
        super(CNSteerableLeNet, self).__init__()
        
        z = 0.5*(imsize - 2)
        z = int(0.5*(z - 2))
        
        self.r2_act = gspaces.Rot2dOnR2(N)
        
        in_type = e2nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type
        
        out_type = e2nn.FieldType(self.r2_act, 6*[self.r2_act.regular_repr])
        self.mask = e2nn.MaskModule(in_type, imsize, margin=1)
        self.conv1 = e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False)
        self.relu1 = e2nn.ReLU(out_type, inplace=True)
        self.pool1 = e2nn.PointwiseMaxPoolAntialiased(out_type, kernel_size=2)

        in_type = self.pool1.out_type
        out_type = e2nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr])
        self.conv2 = e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False)
        self.relu2 = e2nn.ReLU(out_type, inplace=True)
        self.pool2 = e2nn.PointwiseMaxPoolAntialiased(out_type, kernel_size=2)
        
        self.gpool = e2nn.GroupPooling(out_type)

        self.fc1   = nn.Linear(16*z*z, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, out_chan)
        
        self.drop  = nn.Dropout(p=0.5)
        
        # dummy parameter for tracking device
        self.dummy = nn.Parameter(torch.empty(0))
        
        
    def loss(self,p,y):
        
        # check device for model:
        device = self.dummy.device
        
        # p : softmax(x)
        loss_fnc = nn.NLLLoss().to(device=device)
        loss = loss_fnc(torch.log(p),y)
        
        return loss
     
    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        return
      
      
    def forward(self, x):
        
        x = e2nn.GeometricTensor(x, self.input_type)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.gpool(x)
        x = x.tensor
        
        x = x.view(x.size()[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
    
        return x

# -----------------------------------------------------------------------------

class DNSteerableLeNet(nn.Module):
    def __init__(self, in_chan, out_chan, imsize, kernel_size=5, N=8):
        super(DNSteerableLeNet, self).__init__()
        
        z = 0.5*(imsize - 2)
        z = int(0.5*(z - 2))
        
        self.r2_act = gspaces.FlipRot2dOnR2(N)
        
        in_type = e2nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type
        
        out_type = e2nn.FieldType(self.r2_act, 6*[self.r2_act.regular_repr])
        self.mask = e2nn.MaskModule(in_type, imsize, margin=1)
        self.conv1 = e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False)
        self.relu1 = e2nn.ReLU(out_type, inplace=True)
        self.pool1 = e2nn.PointwiseMaxPoolAntialiased(out_type, kernel_size=2)

        in_type = self.pool1.out_type
        out_type = e2nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr])
        self.conv2 = e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False)
        self.relu2 = e2nn.ReLU(out_type, inplace=True)
        self.pool2 = e2nn.PointwiseMaxPoolAntialiased(out_type, kernel_size=2)
        
        self.gpool = e2nn.GroupPooling(out_type)

        self.fc1   = nn.Linear(16*z*z, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, out_chan)
        
        self.drop  = nn.Dropout(p=0.5)
        
        # dummy parameter for tracking device
        self.dummy = nn.Parameter(torch.empty(0))
        
    def loss(self,p,y):
        
        # check device for model:
        device = self.dummy.device
        
        # p : softmax(x)
        loss_fnc = nn.NLLLoss().to(device=device)
        loss = loss_fnc(torch.log(p),y)
        
        return loss
     
    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        return
 
    def forward(self, x):
        
        x = e2nn.GeometricTensor(x, self.input_type)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.gpool(x)
        x = x.tensor
        
        x = x.view(x.size()[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
    
        return x

# -----------------------------------------------------------------------------

class DNRestrictedLeNet(nn.Module):
    def __init__(self, in_chan, out_chan, imsize, kernel_size=5, N=8):
        super(DNRestrictedLeNet, self).__init__()
        
        z = 0.5*(imsize - 2)
        z = int(0.5*(z - 2))
        
        self.r2_act = gspaces.FlipRot2dOnR2(N)
        
        in_type = e2nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type
        
        out_type = e2nn.FieldType(self.r2_act, 6*[self.r2_act.regular_repr])
        self.mask = e2nn.MaskModule(in_type, imsize, margin=1)
        self.conv1 = e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False)
        self.relu1 = e2nn.ReLU(out_type, inplace=True)
        self.pool1 = e2nn.PointwiseMaxPoolAntialiased(out_type, kernel_size=2)

        self.gpool = e2nn.GroupPooling(out_type)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size, padding=1)
        
        self.fc1   = nn.Linear(16*z*z, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, out_chan)
        
        self.drop  = nn.Dropout(p=0.5)
        
        # dummy parameter for tracking device
        self.dummy = nn.Parameter(torch.empty(0))
        
    def loss(self,p,y):
        
        # check device for model:
        device = self.dummy.device
        
        # p : softmax(x)
        loss_fnc = nn.NLLLoss().to(device=device)
        loss = loss_fnc(torch.log(p),y)
        
        return loss
     
    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        return
 
    def forward(self, x):
        
        x = e2nn.GeometricTensor(x, self.input_type)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.gpool(x)
        x = x.tensor
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = x.view(x.size()[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
    
        return x

# -----------------------------------------------------------------------------

class HMTNet(nn.Module):
    
    """
        This network has been taken directly from "transfer learning for radio galaxy classification"
        https://arxiv.org/abs/1903.11921
        """
    
    def __init__(self, in_chan, out_chan, imsize, kernel_size=11, N=None):
        super(HMTNet,self).__init__()
        
        
        self.conv1 = nn.Conv2d(in_channels=in_chan,out_channels=6,kernel_size=(11,11),padding=5,stride=1)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=(5,5),padding=2,stride=1)
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=24,kernel_size=(3,3),padding=1,stride=1)
        self.conv4 = nn.Conv2d(in_channels=24,out_channels=24,kernel_size=(3,3),padding=1,stride=1)
        self.conv5 = nn.Conv2d(in_channels=24,out_channels=16,kernel_size=(3,3),padding=1,stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=5, stride=5)
        self.bnorm1 = nn.BatchNorm2d(6)
        self.bnorm2 = nn.BatchNorm2d(16)
        self.bnorm3 = nn.BatchNorm2d(24)
        self.bnorm4 = nn.BatchNorm2d(24)
        self.bnorm5 = nn.BatchNorm2d(16)
        
        self.fc1 = nn.Linear(400,256) #channel_size * width * height
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,out_chan)
        
        self.dropout = nn.Dropout(p=0.5)
        
        # dummy parameter for tracking device
        self.dummy = nn.Parameter(torch.empty(0))

    
    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


    def loss(self,p,y):
        
        # check device for model:
        device = self.dummy.device
        
        # p : softmax(x)
        loss_fnc = nn.NLLLoss().to(device=device)
        loss = loss_fnc(torch.log(p),y)
        
        return loss

    
    def forward(self, x):
        
        x  = F.relu(self.conv1(x))
        x = self.bnorm1(x)
        x = self.pool1(x)
        
        x  = F.relu(self.conv2(x))
        x = self.bnorm2(x)
        x = self.pool2(x)
        
        x  = F.relu(self.conv3(x))
        x = self.bnorm3(x)
        
        x  = F.relu(self.conv4(x))
        x = self.bnorm4(x)
        
        x  = F.relu(self.conv5(x))
        x = self.bnorm5(x)
        x = self.pool3(x)
        
        x = x.view(x.size()[0], -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        v = self.fc3(x)

        return v


# -----------------------------------------------------------------------------

