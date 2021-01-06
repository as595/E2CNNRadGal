# Configuration Files

Each configuration file contains the parameters for the model, data, optimiser etc. The format is:

```text
[model]
base: 'VanillaLeNet'            # network: [VanillaLeNet, CNSteerableLeNet, DNSteerableLeNet]
nrot: None                      # number of rotations for E(2) subgroups
early_stopping: True            # early stopping [True/False]
quiet: True                     # verbose or not

[data]
dataset: 'MBFRConfident'        # dataset class [FRDEEPF, MBFRConfident]
datadir: 'mirabest'             # name of directory to download data into
datamean: 0.0031                # mean for normalisation
datastd: 0.0350                 # stdev for normalisation  

[training]
batch_size: 50                  # samples per minibatch
frac_val: 0.2                   # fraction of training set for validation
epochs: 600                     # total number of epochs
imsize: 150                     # pixels on side of image
num_classes: 2                  # number of target classes
lr0: 0.0001                     # initial learning rate
decay: 0.000001                 # weight decay

[output]
csvfile: 'mirabest_lenet.csv'    # output file for loss, accuracy etc per epoch 
modfile: 'mirabest_lenet.pt'     # output file for pytorch model
```
