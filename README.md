# E2CNNRadGal

This code will run in a Python 3.6 or 3.8 environment with all the relevant libraries installed (see requirements.txt). For training the equivariant [models](./models.py) (CNSteerableLeNet, DNSteerableLeNet) you will probably want to use a GPU for speed. For the VanillaLeNet, you're better off on a CPU.

## Run the code

The input parameters for each run are contained in the configuration files located in the [configs](./configs) directory. To run a particular experiment use:

```python
python main.py --config configs/config_fr_lenet.txt
```
An overview of the configuration file format can be found [here](./configs/README.md).


## Using a Kaggle kernel

In a Kaggle notebook you can make a local copy of the github repo quickly by running:

```python
!git clone https://github.com/username/repository.git
```

The repo will then appear as a folder in the working directory. To run the code as above you will need to import the [e2cnn]() library and the [torchsummary]() library:

```python
!pip install e2cnn
!pip install torchsummary
```

```python
!python main.py --config configs/config_fr_lenet.txt
```

or

```python
%run main.py --config configs/config_fr_lenet.txt
```


## Data

Configuration files are provided for experiments using the [FRDEEP]() and [MiraBest]() batched data sets. If you use these data sets please cite:

* [FRDEEP]() : Hongming Tang, Anna M. M. Scaife & J. Paddy Leahy, *Transfer learning for radio galaxy classification*, **2019**, MNRAS, 488, 3358 [arXiv](https://arxiv.org/abs/1903.11921)

* [MiraBest]() : Fiona Porter, Anna M. M. Scaife et al., **2020** [zenodo]()
