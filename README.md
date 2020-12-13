# E2CNNRadGal

This code will run in a Python 3.7 or 3.8 environment with all the relevant libraries installed (see requirements.txt), but because it's a neural network you will probably want to use a GPU for speed. 

## Run the code

The input parameters for each run are contained in the configuration files located in the `configs` directory. To run a particular experiment use:

```python
python main.py --config configs/config_fr_lenet.txt
```

Each configuration file contains the parameters for the model, data, optimiser etc. These look like:

```text
[model]
base: 'VanillaLeNet'
nrot: None
early_stopping: True
quiet: True

[data]
dataset: 'FRDEEPF'
datadir: 'frdeepf'
datamean: 0.0019
datastd: 0.0270

[training]
batch_size: 50                  # samples per minibatch
frac_val: 0.2                   # fraction of training set for validation
epochs: 600                     # total number of epochs
imsize: 50                      # pixels on side of image
num_classes: 2                  # number of target classes
lr0: 0.0001                     # initial learning rate
decay: 0.000001                 # weight decay

[output]
csvfile: 'frdeepf_lenet.csv'
modfile: 'frdeepf_lenet.pt'
```


## Using a Kaggle kernel

In a Kaggle notebook you can make a local copy of the github repo quickly by running:

```python
!git clone https://username:password@github.com/username/repository.git
```

and replacing the ```username``` and ```password``` with your own github details. The down side to this is that your username and password are then openly visible in the notebook, which gets saved automatically to your Google Drive. To avoid that happening you can [do this](https://stackoverflow.com/a/57539179) instead:

```python
import os
from getpass import getpass
import urllib
```

```python
user = input('User name: ')
password = getpass('Password: ')
password = urllib.parse.quote(password) # your password is converted into url format
repo_name = input('Repo name: ')
```

i.e. for me these would be:

User name: as595 <br/>
Password: ·········· <br/>
Repo name: E2CNNRadGal <br/>

```python
cmd_string = 'git clone https://{0}:{1}@github.com/{0}/{2}.git'.format(user, password, repo_name)

os.system(cmd_string)
cmd_string, password = "", ""
```

The repo will then appear as a folder (left hand side) and you can run the code directly:

```python
%cd SDSS-AL-MSc/CODE/
```

```python
%run sdss_mlp.py
```


## Data

For the moment I have included a subset of the full SDSS dataset: ```sdss_10k.pkl```. This file contains 10,000 randomly selected objects from the full 2.4 million object spectroscopically labelled training data set and can be used for quick testing of code. Once you're using the GPU on the university cluster then you should switch to using the full data set. 
