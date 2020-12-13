import torch

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchsummary import summary

import os, sys
import numpy as np
import csv
from PIL import Image

from models import VanillaLeNet, CNSteerableLeNet, DNSteerableLeNet
from utils import *
from FRDEEP import FRDEEPF

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

epochs        = 600                 # number of training epochs
imsize        = 50                  # size of input images
batch_size    = 50                  # batch size for mini-batching
learning_rate = 1e-4                # Initial learning rate
weight_decay  = 1e-6                # weght decay
nclass        = 2                   # number of output classes
csvfile       = 'frdeepf_dnlenet.csv' # output file
datamean      = 0.0019
datastd       = 0.0270
quiet         = False
early_stopping= False
Nrot          = int(sys.argv[-1])

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------------------------------------------------------
# Data loading:

crop     = transforms.CenterCrop(imsize)
pad      = transforms.Pad((0, 0, 1, 1), fill=0)
resize1  = transforms.Resize((imsize+1)*3)
resize2  = transforms.Resize(imsize+1)
totensor = transforms.ToTensor()
normalise= transforms.Normalize((datamean,), (datastd,))

transform = transforms.Compose([
    crop,
    pad,
#    resize1,
    transforms.RandomRotation(360, resample=Image.BILINEAR, expand=False),
#    resize2,
    totensor,
    normalise,
])


train_data = FRDEEPF('first', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = FRDEEPF('first', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

# -----------------------------------------------------------------------------

model = DNSteerableLeNet(1, nclass, imsize+1, kernel_size=5, N=Nrot).to(device)

if not quiet:
    summary(model, (1, imsize+1, imsize+1))

# -----------------------------------------------------------------------------

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=2, factor=0.9)

# -----------------------------------------------------------------------------
# training loop:

rows = ['epoch', 'train_loss', 'test_loss', 'test_accuracy', 'test_loss_mc', 'accuracy_mc']
                        
with open(csvfile, 'w+', newline="") as f_out:
        writer = csv.writer(f_out, delimiter=',')
        writer.writerow(rows)
 
_bestloss1 = 1e10; _bestloss2 = 1e10
for epoch in range(epochs):  # loop over the dataset multiple times
    
    train_loss = train(model, train_loader, optimizer, device)
    test_loss, accuracy = test(model, test_loader, device)
    test_loss_mc, accuracy_mc = test_mc(model, test_loader, device)
        
    scheduler.step(test_loss)

    # check early stopping criteria:
    if early_stopping and test_loss<_bestloss1:
        _bestloss1 = test_loss
        torch.save(model.state_dict(), outfile1)
        best_acc = accuracy
        best_epoch = epoch
        
    # check mc dropout early stopping criteria:
    if early_stopping and test_loss_mc<_bestloss2:
        _bestloss2 = test_loss_mc
        torch.save(model.state_dict(), outfile2)
        
    # create output row:
    _results = [epoch, train_loss, test_loss, accuracy, test_loss_mc, accuracy_mc]
    
    with open(csvfile, 'a', newline="") as f_out:
        writer = csv.writer(f_out, delimiter=',')
        writer.writerow(_results)
            
    if not quiet:
        print('Epoch: {}, Validation Loss: {:4f}, Validation Accuracy: {:4f}'.format(epoch, test_loss, accuracy))
        print('Current learning rate is: {}'.format(optimizer.param_groups[0]['lr']))
        
print("Final validation error [Standard]: ",100.*(1 - accuracy))
print("Final validation error [MC Dropout]: ",100.*(1 - accuracy_mc))
if early_stopping:
    print("Best validation error: ",100.*(1 - best_acc)," @ epoch: "+str(best_epoch))


# -----------------------------------------------------------------------------
# create outputs:

#if not early_stopping:
#    torch.save(model.state_dict(), outfile)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# END
