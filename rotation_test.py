import torch

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torchsummary import summary

import numpy as np
import csv
import glob
from PIL import Image

from models import VanillaLeNet, CNSteerableLeNet, DNSteerableLeNet
from utils import *
from plots import *
from FRDEEP import FRDEEPF
from MiraBest import MBFRConfident

# -----------------------------------------------------------------------------
# extract information from config file:

vars = parse_args()
config_dict, config = parse_config(vars['config'])

batch_size     = 1
epochs         = config_dict['training']['epochs']
imsize         = config_dict['training']['imsize']
nclass         = config_dict['training']['num_classes']

nrot           = config_dict['model']['nrot']

outdir         = config_dict['metrics']['outputdir']
modelfiles     = glob.glob(outdir+"/*.pt")

config         = vars['config'].split('/')[-1][:-4]
csvfile        = 'overlap.csv'

# -----------------------------------------------------------------------------
# check if gpu is available:

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
print("Device: ", device)

# -----------------------------------------------------------------------------
# load data set:

crop     = transforms.CenterCrop(imsize)
pad      = transforms.Pad((0, 0, 1, 1), fill=0)
totensor = transforms.ToTensor()
normalise= transforms.Normalize((config_dict['data']['datamean'],), (config_dict['data']['datastd'],))

transform = transforms.Compose([
    crop,
    pad,
    totensor,
    normalise,
])


test_data = locals()[config_dict['data']['dataset']](config_dict['data']['datadir'], train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

N = len(test_loader)
targ = np.zeros(N, dtype=np.int8)

# -----------------------------------------------------------------------------
# specify model:

model = locals()[config_dict['model']['base']](1, nclass, imsize+1, kernel_size=5, N=nrot).to(device)

# -----------------------------------------------------------------------------

metrics=[]

# load saved model:
if use_cuda:
    model.load_state_dict(torch.load(modelfiles[0]))
else:
    model.load_state_dict(torch.load(modelfiles[0], map_location=torch.device('cpu')))

rows = ['target', 'softmax prob', 'average overlap', 'overlap variance']
                        
with open(csvfile, 'a', newline="") as f_out:
        writer = csv.writer(f_out, delimiter=',')
        writer.writerow(rows)

for i in range(0,N):

    subset_indices = [i] # select your indices here as a list
    subset = torch.utils.data.Subset(test_data, subset_indices)
    testloader_ordered = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)
    data, target = iter(testloader_ordered).next()
    
    data = data.to(device)
    
    # get straight prediction:
    model.eval()
    x = model(data)
    p = F.softmax(x,dim=1)[0].detach().cpu().numpy()
    
    av_overlap, std_overlap = fr_rotation_test(model, data, target, i)
    print(i, av_overlap, std_overlap)
    
    # create output row:
    _results = [target[0].item(), p, av_overlap, std_overlap]
    
    with open(csvfile, 'a', newline="") as f_out:
        writer = csv.writer(f_out, delimiter=',')
        writer.writerow(_results)
    
