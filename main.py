import torch

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torchsummary import summary

import numpy as np
import csv
from PIL import Image

from models import VanillaLeNet, CNSteerableLeNet, DNSteerableLeNet, DNRestrictedLeNet
from utils import *
from FRDEEP import FRDEEPF
from MiraBest import MBFRConfident

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# extract information from config file:

vars = parse_args()
config_dict, config = parse_config(vars['config'])

batch_size     = config_dict['training']['batch_size']
frac_val       = config_dict['training']['frac_val']
epochs         = config_dict['training']['epochs']
imsize         = config_dict['training']['imsize']
nclass         = config_dict['training']['num_classes']
learning_rate  = torch.tensor(config_dict['training']['lr0'])
weight_decay   = torch.tensor(config_dict['training']['decay'])

early_stopping = config_dict['model']['early_stopping']
quiet          = config_dict['model']['quiet']
nrot           = config_dict['model']['nrot']

csvfile        = config_dict['output']['csvfile']
modfile        = config_dict['output']['modfile']

config         = vars['config'].split('/')[-1][:-4]

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------------------------------------------------------
# Data loading:

crop     = transforms.CenterCrop(imsize)
pad      = transforms.Pad((0, 0, 1, 1), fill=0)
totensor = transforms.ToTensor()
normalise= transforms.Normalize((config_dict['data']['datamean'],), (config_dict['data']['datastd'],))

transform = transforms.Compose([
    crop,
    pad,
    transforms.RandomRotation(360, resample=Image.BILINEAR, expand=False),
    totensor,
    normalise,
])


train_data = locals()[config_dict['data']['dataset']](config_dict['data']['datadir'], train=True, download=True, transform=transform)

if frac_val>0.:
    dataset_size = len(train_data)
    nval = int(frac_val*dataset_size)

    indices = list(range(dataset_size))
    train_indices, val_indices = indices[nval:], indices[:nval]

    train_sampler = Subset(train_data, train_indices)
    valid_sampler = Subset(train_data, val_indices)

    train_loader = torch.utils.data.DataLoader(train_sampler, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(valid_sampler, batch_size=batch_size, shuffle=True)
else:
    # setting frac_val to zero will use the test set for validation
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    test_data = locals()[config_dict['data']['dataset']](config_dict['data']['datadir'], train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(valid_sampler, batch_size=batch_size, shuffle=True)

# -----------------------------------------------------------------------------


model = locals()[config_dict['model']['base']](1, nclass, imsize+1, kernel_size=5, N=nrot).to(device)

if not quiet:
    summary(model, (1, imsize+1, imsize+1))

# -----------------------------------------------------------------------------

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=2, factor=0.9)

# -----------------------------------------------------------------------------
# training loop:

rows = ['epoch', 'train_loss', 'test_loss', 'test_accuracy']
                        
with open(csvfile, 'w+', newline="") as f_out:
        writer = csv.writer(f_out, delimiter=',')
        writer.writerow(rows)
 
_bestacc = 0.
for epoch in range(epochs):  # loop over the dataset multiple times
    
    train_loss = train(model, train_loader, optimizer, device)
    test_loss, accuracy = test(model, test_loader, device)
        
    scheduler.step(test_loss)

    # check early stopping criteria:
    if early_stopping and accuracy>_bestacc:
        _bestacc = accuracy
        torch.save(model.state_dict(), modfile)
        best_acc = accuracy
        best_epoch = epoch
        
    # create output row:
    _results = [epoch, train_loss, test_loss, accuracy]
    
    with open(csvfile, 'a', newline="") as f_out:
        writer = csv.writer(f_out, delimiter=',')
        writer.writerow(_results)
            
    if not quiet:
        print('Epoch: {}, Validation Loss: {:4f}, Validation Accuracy: {:4f}'.format(epoch, test_loss, accuracy))
        print('Current learning rate is: {}'.format(optimizer.param_groups[0]['lr']))
        
print("Final validation error: ",100.*(1 - accuracy))
if early_stopping:
    print("Best validation error: ",100.*(1 - best_acc)," @ epoch: "+str(best_epoch))


# -----------------------------------------------------------------------------
# create outputs:

if not early_stopping:
    torch.save(model.state_dict(), modfile)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# END
