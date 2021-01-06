# E2CNNRadGal

This code will run in a Python 3.6 or 3.8 environment with all the relevant libraries installed (see requirements.txt). For training the equivariant [models](./models.py) (CNSteerableLeNet, DNSteerableLeNet) you will probably want to use a GPU for speed. For the VanillaLeNet, you're better off on a CPU.

## Run the code

The input parameters for each run are contained in the configuration files located in the [configs](./configs) directory. To run a particular experiment use:

```python
python main.py --config configs/config_mb_lenet.txt
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
!python main.py --config configs/config_mb_lenet.txt
```

or

```python
%run main.py --config configs/config_mb_lenet.txt
```


## Data

Configuration files are provided for experiments using the [FRDEEP]() and [MiraBest]() batched data sets. If you use these data sets please cite:

* [MiraBest](https://zenodo.org/record/4288837#.X_XjDC-l3Aw) : Fiona Porter, Anna M. M. Scaife et al., **2020** [Zenodo: 10.5281/zenodo.4288837](https://zenodo.org/record/4288837#.X_XjDC-l3Aw)


## Demo 

Since E(2)-steerable CNNs are equivariant under rotations and reflections, their inference is independent from the choice of image orientation.
The visualization below demonstrates this claim by feeding rotated images into a randomly initialized E(2)-steerable CNN (left).
The middle plot shows the equivariant transformation of a feature space, consisting of one scalar field (color-coded) and one vector field (arrows), after a few layers.
In the right plot we transform the feature space into a comoving reference frame by rotating the response fields back (stabilized view).

![Equivariant CNN output](https://github.com/QUVA-Lab/e2cnn/raw/master/visualizations/vectorfield.gif)

The invariance of the features in the comoving frame validates the rotational equivariance of E(2)-steerable CNNs empirically.
Note that the fluctuations of responses are discretization artifacts due to the sampling of the image on a pixel grid, which does not allow for exact continuous rotations.
<!-- Note that the fluctuations of responses are due to discretization artifacts coming from the  -->

For comparison, we show a feature map response of a conventional CNN for different image orientations below.

![Conventional CNN output](https://github.com/QUVA-Lab/e2cnn/raw/master/visualizations/conventional_cnn.gif)

Since conventional CNNs are not equivariant under rotations, the response varies randomly with the image orientation.
This prevents CNNs from automatically generalizing learned patterns between different reference frames.
