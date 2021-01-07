import pylab as pl
import pandas as pd
import numpy as np

import glob
from utils import *
from MiraBest import MBFRConfident

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import Subset

from matplotlib.ticker import ScalarFormatter,StrMethodFormatter

pl.rcParams["font.family"] = "Times"
pl.rcParams['font.size'] = 12

# ------------------------------------------------------------------------------

def plot_err_csv(filename1):
    
    pl.subplot(111)
    
    df = pd.read_csv(filename1)
    test_err = (1 - df["test_accuracy"])*100
    epoch = df["epoch"]
    
    pl.plot(epoch,test_err)
    
    pl.ylabel("Validation Error [%]")
    pl.xlabel("Epoch")
    pl.title('Test error')
    pl.show()
    
    return

# ------------------------------------------------------------------------------

def plot_err_comp():
    
    pl.subplot(111)
    
    filename1 = "mnist.csv"
    df = pd.read_csv(filename1)
    test_err = (1 - df["test_accuracy"])*10000
    epoch = df["epoch"]
    pl.plot(epoch,test_err,label="with L2")
    
    filename1 = "mnist_nol2_minusalpha.csv"
    df = pd.read_csv(filename1)
    test_err = (1 - df["test_accuracy"])*10000
    epoch = df["epoch"]
    pl.plot(epoch,test_err,label="SGD minus")
    
    filename1 = "mnist_nol2_plusalpha.csv"
    df = pd.read_csv(filename1)
    test_err = (1 - df["test_accuracy"])*10000
    epoch = df["epoch"]
    pl.plot(epoch,test_err,label="SGD plus")
    
    
    pl.plot([0.,1000.],[160.,160.],ls='-',c='black')
    
    pl.axis([0.,1000.,80.,220.])
    pl.ylabel("Number of Errors")
    pl.xlabel("Epoch")
    pl.title('Test error')
    pl.legend()
    pl.show()
    
    return

# -----------------------------------------------------------------------------

def plot_loss_csv(filename1):
    
    pl.subplot(111)
    
    df = pd.read_csv(filename1)
    
    train_loss = df["train_loss"]
    test_loss = df["test_loss"]
    epoch = df["epoch"]
    
    pl.plot(epoch,train_loss,label="Train")
    pl.plot(epoch,test_loss,label="Test")
    
    pl.ylabel("Loss")
    pl.xlabel("Epoch")
    pl.legend()
    pl.show()
    
    return

# -----------------------------------------------------------------------------

def plot_loss_comb(snippet):
    
    files = glob.glob(snippet)
    print(files)
    i=0
    for file in files:

        df = pd.read_csv(file)
    
        train = df["train_loss"].values
        test  = df["test_loss"].values
        epoch  = df["epoch"].values
        
        if i==0:
            train_loss, test_loss = train, test
            train_loss_sq, test_loss_sq = train**2, test**2
            i=1
        else:
            train_loss += train
            test_loss  += test
            
            train_loss_sq += train**2
            test_loss_sq  += test**2
        
    train_mean = train_loss/len(files)
    test_mean  = test_loss/len(files)
    
    train_std = np.sqrt(train_loss_sq/len(files) - train_mean**2)
    test_std  = np.sqrt(test_loss_sq/len(files) - test_mean**2)
    
    pl.subplot(111)
    
    pl.plot(epoch,train_mean, label='Train')
    pl.fill_between(epoch, train_mean-train_std, train_mean+train_std, alpha=0.3)
    
    pl.plot(epoch,test_mean, label='Validation')
    pl.fill_between(epoch, test_mean-test_std, test_mean+test_std, alpha=0.3)
    
    pl.ylabel("Loss")
    pl.xlabel("Epoch")
    pl.legend()
    
    pl.show()
    
    return

# -----------------------------------------------------------------------------

def plot_loss_multi(dataset='cn',N=10):
    
    if dataset=='cn':
        path1 = 'mb_lenet/mirabest_lenet_*.csv'
        path2 = 'mb_cn4lenet/mirabest_cnlenet_*.csv'
        path3 = 'mb_cn8lenet/mirabest_cnlenet_*.csv'
        path4 = 'mb_cn16lenet/mirabest_cnlenet_*.csv'
        path5 = 'mb_cn20lenet/mirabest_cnlenet_*.csv'
    elif dataset=='dn':
        path1 = 'mb_lenet/mirabest_lenet_*.csv'
        path2 = 'mb_dn4lenet/mirabest_dnlenet_*.csv'
        path3 = 'mb_dn8lenet/mirabest_dnlenet_*.csv'
        path4 = 'mb_dn16lenet/mirabest_dnlenet_*.csv'
        path5 = 'mb_dn20lenet/mirabest_dnlenet_*.csv'
    else:
        return
    
    ax = pl.subplot(111)
    
    files = glob.glob(path1)
    i=0
    for file in files:

        df = pd.read_csv(file)
    
        test  = df["test_loss"].values
        epoch  = df["epoch"].values
        
        if i==0:
            test_loss = test
            test_loss_sq = test**2
            i=1
        else:
            test_loss += test
            test_loss_sq += test**2
        
    test_mean  = test_loss/len(files)
    test_std  = np.sqrt(test_loss_sq/len(files) - test_mean**2)
    
    test_mean = np.convolve(test_mean, np.ones(N)/N, mode='valid')
    test_std = np.convolve(test_std, np.ones(N)/N, mode='valid')
    epoch = np.convolve(epoch, np.ones(N)/N, mode='valid')
    print(test_std[-1])
    
    pl.plot(epoch,test_mean, label='{e}')
    pl.fill_between(epoch, test_mean-test_std, test_mean+test_std, alpha=0.3)
    
    files = glob.glob(path2)
    i=0
    for file in files:

        df = pd.read_csv(file)
    
        test  = df["test_loss"].values
        epoch  = df["epoch"].values
        
        if i==0:
            test_loss = test
            test_loss_sq = test**2
            i=1
        else:
            test_loss += test
            test_loss_sq += test**2
        
    test_mean  = test_loss/len(files)
    test_std  = np.sqrt(test_loss_sq/len(files) - test_mean**2)
    
    test_mean = np.convolve(test_mean, np.ones(N)/N, mode='valid')
    test_std = np.convolve(test_std, np.ones(N)/N, mode='valid')
    epoch = np.convolve(epoch, np.ones(N)/N, mode='valid')
    print(test_std[-1])
        
    if dataset=='cn':
        label = r'$C_4$'
    elif dataset=='dn':
        label = r'$D_4$'
    
    pl.plot(epoch,test_mean, label=label)
    pl.fill_between(epoch, test_mean-test_std, test_mean+test_std, alpha=0.3)
    
    files = glob.glob(path3)
    i=0
    for file in files:

        df = pd.read_csv(file)
    
        test  = df["test_loss"].values
        epoch  = df["epoch"].values
        
        if i==0:
            test_loss = test
            test_loss_sq = test**2
            i=1
        else:
            test_loss += test
            test_loss_sq += test**2
        
    test_mean  = test_loss/len(files)
    test_std  = np.sqrt(test_loss_sq/len(files) - test_mean**2)
    
    test_mean = np.convolve(test_mean, np.ones(N)/N, mode='valid')
    test_std = np.convolve(test_std, np.ones(N)/N, mode='valid')
    epoch = np.convolve(epoch, np.ones(N)/N, mode='valid')
    print(test_std[-1])
        
    if dataset=='cn':
        label = r'$C_8$'
    elif dataset=='dn':
        label = r'$D_8$'
        
    pl.plot(epoch,test_mean, label=label)
    pl.fill_between(epoch, test_mean-test_std, test_mean+test_std, alpha=0.3)
    
    files = glob.glob(path4)
    i=0
    for file in files:

        df = pd.read_csv(file)
    
        test  = df["test_loss"].values
        epoch  = df["epoch"].values
        
        if i==0:
            test_loss = test
            test_loss_sq = test**2
            i=1
        else:
            test_loss += test
            test_loss_sq += test**2
        
    test_mean  = test_loss/len(files)
    test_std  = np.sqrt(test_loss_sq/len(files) - test_mean**2)
    
    test_mean = np.convolve(test_mean, np.ones(N)/N, mode='valid')
    test_std = np.convolve(test_std, np.ones(N)/N, mode='valid')
    epoch = np.convolve(epoch, np.ones(N)/N, mode='valid')
    print(test_std[-1])
        
    if dataset=='cn':
        label = r'$C_{16}$'
    elif dataset=='dn':
        label = r'$D_{16}$'
    
    pl.plot(epoch,test_mean, label=label)
    pl.fill_between(epoch, test_mean-test_std, test_mean+test_std, alpha=0.3)
    
    files = glob.glob(path5)
    i=0
    for file in files:

        df = pd.read_csv(file)
    
        test  = df["test_loss"].values
        epoch  = df["epoch"].values
        
        if i==0:
            test_loss = test
            test_loss_sq = test**2
            i=1
        else:
            test_loss += test
            test_loss_sq += test**2
        
    test_mean  = test_loss/len(files)
    test_std  = np.sqrt(test_loss_sq/len(files) - test_mean**2)
    
    test_mean = np.convolve(test_mean, np.ones(N)/N, mode='valid')
    test_std = np.convolve(test_std, np.ones(N)/N, mode='valid')
    epoch = np.convolve(epoch, np.ones(N)/N, mode='valid')
    print(test_std[-1])
        
    if dataset=='cn':
        label = r'$C_{20}$'
    elif dataset=='dn':
        label = r'$D_{20}$'
    
    pl.plot(epoch,test_mean, label=label)
    pl.fill_between(epoch, test_mean-test_std, test_mean+test_std, alpha=0.3)
    
    
    pl.ylabel("Validation Loss")
    pl.xlabel("Epoch")
    
    ax.semilogy()
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(ScalarFormatter())
    
    pl.legend()
    pl.axis([0,600,0.12,0.45])
    
    pl.show()
    
    return


# -----------------------------------------------------------------------------

def plot_err_multi(dataset='dn', N=10):
    
    if dataset=='cn':
        path1 = 'mb_lenet/mirabest_lenet_*.csv'
        path2 = 'mb_cn4lenet/mirabest_cnlenet_*.csv'
        path3 = 'mb_cn8lenet/mirabest_cnlenet_*.csv'
        path4 = 'mb_cn16lenet/mirabest_cnlenet_*.csv'
        path5 = 'mb_cn20lenet/mirabest_cnlenet_*.csv'
    elif dataset=='dn':
        path1 = 'mb_lenet/mirabest_lenet_*.csv'
        path2 = 'mb_dn4lenet/mirabest_dnlenet_*.csv'
        path3 = 'mb_dn8lenet/mirabest_dnlenet_*.csv'
        path4 = 'mb_dn16lenet/mirabest_dnlenet_*.csv'
        path5 = 'mb_dn20lenet/mirabest_dnlenet_*.csv'
    else:
        return
        
    ax = pl.subplot(111)
    
    files = glob.glob(path1)
    i=0
    for file in files:

        df = pd.read_csv(file)
    
        test  = (1 - df["test_accuracy"].values)*100.
        epoch  = df["epoch"].values
        
        if i==0:
            test_loss = test
            test_loss_sq = test**2
            i=1
        else:
            test_loss += test
            test_loss_sq += test**2
        
    test_mean  = test_loss/len(files)
    test_std  = np.sqrt(test_loss_sq/len(files) - test_mean**2)
    
    test_mean = np.convolve(test_mean, np.ones(N)/N, mode='valid')
    test_std = np.convolve(test_std, np.ones(N)/N, mode='valid')
    epoch = np.convolve(epoch, np.ones(N)/N, mode='valid')
    
    pl.plot(epoch,test_mean, label='{e}')
    pl.fill_between(epoch, test_mean-test_std, test_mean+test_std, alpha=0.3)
    
    files = glob.glob(path2)
    i=0
    for file in files:

        df = pd.read_csv(file)
    
        test  = (1 - df["test_accuracy"].values)*100.
        epoch  = df["epoch"].values
        
        if i==0:
            test_loss = test
            test_loss_sq = test**2
            i=1
        else:
            test_loss += test
            test_loss_sq += test**2
        
    test_mean  = test_loss/len(files)
    test_std  = np.sqrt(test_loss_sq/len(files) - test_mean**2)
    
    test_mean = np.convolve(test_mean, np.ones(N)/N, mode='valid')
    test_std = np.convolve(test_std, np.ones(N)/N, mode='valid')
    epoch = np.convolve(epoch, np.ones(N)/N, mode='valid')
    
    pl.plot(epoch,test_mean, label=r'$C_4$')
    pl.fill_between(epoch, test_mean-test_std, test_mean+test_std, alpha=0.3)
    
    files = glob.glob(path3)
    i=0
    for file in files:

        df = pd.read_csv(file)
    
        test  = (1 - df["test_accuracy"].values)*100.
        epoch  = df["epoch"].values
        
        if i==0:
            test_loss = test
            test_loss_sq = test**2
            i=1
        else:
            test_loss += test
            test_loss_sq += test**2
        
    test_mean  = test_loss/len(files)
    test_std  = np.sqrt(test_loss_sq/len(files) - test_mean**2)
    
    test_mean = np.convolve(test_mean, np.ones(N)/N, mode='valid')
    test_std = np.convolve(test_std, np.ones(N)/N, mode='valid')
    epoch = np.convolve(epoch, np.ones(N)/N, mode='valid')
    
    pl.plot(epoch,test_mean, label=r'$C_8$')
    pl.fill_between(epoch, test_mean-test_std, test_mean+test_std, alpha=0.3)
    
    files = glob.glob(path4)
    i=0
    for file in files:

        df = pd.read_csv(file)
    
        test  = (1 - df["test_accuracy"].values)*100.
        epoch  = df["epoch"].values
        
        if i==0:
            test_loss = test
            test_loss_sq = test**2
            i=1
        else:
            test_loss += test
            test_loss_sq += test**2
        
    test_mean  = test_loss/len(files)
    test_std  = np.sqrt(test_loss_sq/len(files) - test_mean**2)
    
    test_mean = np.convolve(test_mean, np.ones(N)/N, mode='valid')
    test_std = np.convolve(test_std, np.ones(N)/N, mode='valid')
    epoch = np.convolve(epoch, np.ones(N)/N, mode='valid')
    
    pl.plot(epoch,test_mean, label=r'$C_{16}$')
    pl.fill_between(epoch, test_mean-test_std, test_mean+test_std, alpha=0.3)
    
    files = glob.glob(path5)
    i=0
    for file in files:

        df = pd.read_csv(file)
    
        test  = (1 - df["test_accuracy"].values)*100.
        epoch  = df["epoch"].values
        
        if i==0:
            test_loss = test
            test_loss_sq = test**2
            i=1
        else:
            test_loss += test
            test_loss_sq += test**2
        
    test_mean  = test_loss/len(files)
    test_std  = np.sqrt(test_loss_sq/len(files) - test_mean**2)
    
    test_mean = np.convolve(test_mean, np.ones(N)/N, mode='valid')
    test_std = np.convolve(test_std, np.ones(N)/N, mode='valid')
    epoch = np.convolve(epoch, np.ones(N)/N, mode='valid')
    
    pl.plot(epoch,test_mean, label=r'$C_{20}$')
    pl.fill_between(epoch, test_mean-test_std, test_mean+test_std, alpha=0.3)
    
    
    pl.ylabel("Validation Error [%]")
    pl.xlabel("Epoch")
    
    ax.semilogy()
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_minor_formatter(StrMethodFormatter('{x:,.0f}'))
    
    pl.legend()
    
    pl.show()
    
    return


# -----------------------------------------------------------------------------

def plot_fr_err_order():

    nlist = [2,4,8,16,20]
    
    path = 'lenet/frdeepf_lenet_*'
        
    files = glob.glob(path)
    i=0
    for file in files:
            
        df = pd.read_csv(file)
    
        test  = (1 - df["test_accuracy"].values)*100.
            
        if i==0:
            best = np.min(test)
            best_sq = np.min(test)**2
            i=1
        else:
            best += np.min(test)
            best_sq += np.min(test)**2
        
    cnn_mean  = best/len(files)
    cnn_std  = np.sqrt(best_sq/len(files) - cnn_mean**2)
        

    cn = []; cn_std = []
    dn = []; dn_std = []
    for n in nlist:
        path = 'cn'+str(n)+'lenet/frdeepf_cnlenet_*'
        
        files = glob.glob(path)
        i=0
        for file in files:

            df = pd.read_csv(file)
    
            test  = (1 - df["test_accuracy"].values)*100.
            
            if i==0:
                best = np.min(test)
                best_sq = np.min(test)**2
                i=1
            else:
                best += np.min(test)
                best_sq += np.min(test)**2
        
        best_mean  = best/len(files)
        best_std  = np.sqrt(best_sq/len(files) - best_mean**2)
        
        cn.append(best_mean)
        cn_std.append(best_std)
 
#        path = 'dn'+str(n)+'lenet/frdeepf_dnlenet_*'
#
#        files = glob.glob(path)
#        i=0
#        for file in files:
#
#            df = pd.read_csv(file)
#
#            test  = (1 - df["test_accuracy"].values)*100.
#
#            if i==0:
#                best = np.min(test)
#                best_sq = np.min(test)**2
#                i=1
#            else:
#                best += np.min(test)
#                best_sq += np.min(test)**2
#
#        best_mean  = best/len(files)
#        best_std  = np.sqrt(best_sq/len(files) - best_mean**2)
#
#        dn.append(best_mean)
#        dn_std.append(best_std)

    
    cn = np.array(cn)
    #dn = np.array(dn)
    
    cn_std = np.array(cn_std)
    #dn_std = np.array(dn_std)
        
    ax = pl.subplot(111)
    
    ax.plot(nlist, cnn_mean*np.ones(len(nlist)), label = r"$\{e\}$")
    ax.fill_between(nlist, (cnn_mean+cnn_std)*np.ones(len(nlist)), (cnn_mean-cnn_std)*np.ones(len(nlist)), alpha=0.3)
        
    ax.plot(nlist, cn, label=r"$C_N$")
    ax.fill_between(nlist, cn+cn_std, cn-cn_std, alpha=0.3)
    
    #pl.plot(nlist, dn, label=r"$D_N$")
    #pl.fill_between(nlist, dn+dn_std, dn-dn_std, alpha=0.3)
    
    pl.ylabel("Validation Error [%]")
    pl.xlabel("Epoch")
    ax.semilogy()
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(ScalarFormatter())
    
    pl.legend()
    
    pl.show()
    

    return

# -----------------------------------------------------------------------------

def plot_mb_err_order(N=10):

    nlist = [2,4,6,8,10,12,14,16,20]
    
    path = 'mb_lenet/mirabest_lenet_*.csv'
        
    files = glob.glob(path)
    i=0
    for file in files:
            
        df = pd.read_csv(file)
    
        test  = (1 - df["test_accuracy"].values)*100.
            
        if i==0:
            best = test
            best_sq = test**2
            i=1
        else:
            best += test
            best_sq += test**2
        
    cnn_mean  = best/len(files)
    cnn_std  = np.sqrt(best_sq/len(files) - cnn_mean**2)
        
    cnn_mean = np.convolve(cnn_mean, np.ones(N)/N, mode='valid')[-1]
    cnn_std = np.convolve(cnn_std, np.ones(N)/N, mode='valid')[-1]

    cn = []; cn_std = []
    dn = []; dn_std = []
    for n in nlist:
        path = 'mb_cn'+str(n)+'lenet/mirabest_cnlenet_*.csv'
        
        files = glob.glob(path)
        i=0
        for file in files:

            df = pd.read_csv(file)
    
            test  = (1 - df["test_accuracy"].values)*100.
            
            if i==0:
                best = test
                best_sq = test**2
                i=1
            else:
                best += test
                best_sq += test**2
        
        best_mean  = best/len(files)
        best_std  = np.sqrt(best_sq/len(files) - best_mean**2)
        
        best_mean = np.convolve(best_mean, np.ones(N)/N, mode='valid')[-1]
        best_std = np.convolve(best_std, np.ones(N)/N, mode='valid')[-1]

        cn.append(best_mean)
        cn_std.append(best_std)
 
        path = 'mb_dn'+str(n)+'lenet/mirabest_dnlenet_*.csv'

        files = glob.glob(path)
        i=0
        for file in files:

            df = pd.read_csv(file)

            test  = (1 - df["test_accuracy"].values)*100.

            if i==0:
                best = test
                best_sq = test**2
                i=1
            else:
                best += test
                best_sq += test**2

        best_mean  = best/len(files)
        best_std  = np.sqrt(best_sq/len(files) - best_mean**2)

        best_mean = np.convolve(best_mean, np.ones(N)/N, mode='valid')[-1]
        best_std = np.convolve(best_std, np.ones(N)/N, mode='valid')[-1]

        dn.append(best_mean)
        dn_std.append(best_std)

    
    cn = np.array(cn)
    dn = np.array(dn)
    
    cn_std = np.array(cn_std)
    dn_std = np.array(dn_std)
    print(cn_std)
    print(dn_std)
    
    ax = pl.subplot(111)
    
    ax.plot(nlist, cnn_mean*np.ones(len(nlist)), label = r"$\{e\}$")
    ax.fill_between(nlist, (cnn_mean+cnn_std)*np.ones(len(nlist)), (cnn_mean-cnn_std)*np.ones(len(nlist)), alpha=0.3)
        
    ax.plot(nlist, cn, label=r"$C_N$")
    ax.fill_between(nlist, cn+cn_std, cn-cn_std, alpha=0.3)
    
    ax.plot(nlist, dn, label=r"$D_N$")
    ax.fill_between(nlist, dn+dn_std, dn-dn_std, alpha=0.3)
    
    pl.ylabel("Validation Error [%]")
    pl.xlabel(r"$N$")
    ax.set_ylim([4.5,13.])
    ax.semilogy()
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_minor_formatter(StrMethodFormatter('{x:,.0f}'))
    
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_minor_formatter(StrMethodFormatter('{x:,.0f}'))
    
    pl.legend()
    pl.show()
    

    return


# -----------------------------------------------------------------------------

def test_overlap(beta = 0.1):

    n = 50
    
    x = np.random.normal(0.6,0.1,n)
    y = np.random.normal(0.6,0.1,n)
    
    print(overlapping(x,y,beta))
    
    return

# -----------------------------------------------------------------------------

def plot_hist(targets, overlaps):

    pl.subplot(111)
    
    x = overlaps[np.where(targets==0)]
    y = overlaps[np.where(targets==1)]
    
    _ = pl.hist(x, bins=20, facecolor='r', alpha=0.75)
    _ = pl.hist(y, bins=20, facecolor='b', alpha=0.75)
    
    pl.show()

    return

# -----------------------------------------------------------------------------

def plot_overlap():

    path1 = 'lenet_overlap.csv'
    path2 = 'dn16_overlap.csv'
    
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    targ1 = df1['target'].values
    olap1 = df1['average overlap'].values
    slap1 = df1['overlap variance'].values

    targ2 = df2['target'].values
    olap2 = df2['average overlap'].values
    slap2 = df2['overlap variance'].values
    
    pl.subplot(111)
    pl.scatter(olap1, slap1, c='r', label=r"$\{e\}$")
    pl.scatter(olap2, slap2, c='b', label=r"$D_{16}$")
    pl.legend()
    pl.show()

    return

# -----------------------------------------------------------------------------

def plot_ranked(reverse=False, N=5):

    imsize = 150
    
    crop     = transforms.CenterCrop(imsize)
    pad      = transforms.Pad((0, 0, 1, 1), fill=0)
    totensor = transforms.ToTensor()
    normalise= transforms.Normalize((0.0031,), (0.0350,))

    transform = transforms.Compose([
        crop,
        pad,
        totensor,
        normalise,
    ])
    
    test_data = MBFRConfident('mirabest', train=False, transform=transform)
    
    path1 = 'lenet_overlap.csv'
    path2 = 'dn16_overlap.csv'
    
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    targ1 = df1['target'].values
    p1    = df1['softmax prob'].values
    olap1 = df1['average overlap'].values
    slap1 = df1['overlap variance'].values

    targ2 = df2['target'].values
    p2    = df2['softmax prob'].values
    olap2 = df2['average overlap'].values
    slap2 = df2['overlap variance'].values
    
    p1 = [np.array(p1[i].lstrip('[').rstrip(']').split(), dtype=float) for i in range(len(targ1))]
    p2 = [np.array(p2[i].lstrip('[').rstrip(']').split(), dtype=float) for i in range(len(targ1))]
    
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    diff = slap1 - slap2
    
    if reverse:
        i_ep = np.argsort(diff)[::-1]
    else:
        i_ep = np.argsort(diff)
    print(diff[i_ep[0]])
    
    from mpl_toolkits.axes_grid1 import ImageGrid

    fig = pl.figure()
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(1, N),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    j=0
    for i in i_ep[0:N]:
        subset_indices = [i] # select your indices here as a list
        subset = torch.utils.data.Subset(test_data, subset_indices)
        testloader_ordered = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)
        data, target = iter(testloader_ordered).next()
            
        grid[j].imshow(np.squeeze(data))  # The AxesGrid object work as a list of axes.
        rectangle = pl.Circle((75,75), 25, fill=False, ec="white")
        grid[j].add_patch(rectangle)
        grid[j].text(5,15,"Source: {:2d}".format(i),{'color': 'white', 'fontsize': 10})
        grid[j].text(5,145,"True: {}, Predicted: [{},{}]".format(targ1[i],np.argmax(p1[i,:]),np.argmax(p2[i,:])),{'color': 'white', 'fontsize': 10})
        grid[j].axis('off')
        j+=1

    #grid[j].axis('off'); grid[j+1].axis('off'); grid[j+2].axis('off')

    pl.show()

    return

# -----------------------------------------------------------------------------

def plot_loss_norot(N=20):
    
    path1 = 'mb_lenet/mirabest_lenet_*.csv'
    path2 = 'mb_norot/mirabest_lenet_norot*.csv'
        
    path3 = 'mb_dn16lenet/mirabest_dnlenet_*.csv'
    path4 = 'mb_norot/mirabest_dnlenet_norot*.csv'
    
    
    ax = pl.subplot(111)
    
    files = glob.glob(path1)
    i=0
    for file in files:

        df = pd.read_csv(file)
    
        test  = df["test_loss"].values
        epoch  = df["epoch"].values
        
        if i==0:
            test_loss = test
            test_loss_sq = test**2
            i=1
        else:
            test_loss += test
            test_loss_sq += test**2
        
    test_mean  = test_loss/len(files)
    test_std  = np.sqrt(test_loss_sq/len(files) - test_mean**2)
    
    test_mean = np.convolve(test_mean, np.ones(N)/N, mode='valid')
    test_std = np.convolve(test_std, np.ones(N)/N, mode='valid')
    epoch = np.convolve(epoch, np.ones(N)/N, mode='valid')
    print(test_std[-1])
    
    pl.plot(epoch,test_mean, label='{e}')
    pl.fill_between(epoch, test_mean-test_std, test_mean+test_std, alpha=0.3)
    
    files = glob.glob(path3)
    i=0
    for file in files:

        df = pd.read_csv(file)
    
        test  = df["test_loss"].values
        epoch  = df["epoch"].values
        
        if i==0:
            test_loss = test
            test_loss_sq = test**2
            i=1
        else:
            test_loss += test
            test_loss_sq += test**2
        
    test_mean  = test_loss/len(files)
    test_std  = np.sqrt(test_loss_sq/len(files) - test_mean**2)
    
    test_mean = np.convolve(test_mean, np.ones(N)/N, mode='valid')
    test_std = np.convolve(test_std, np.ones(N)/N, mode='valid')
    epoch = np.convolve(epoch, np.ones(N)/N, mode='valid')
        
    label = r'$D_16$'
    
    pl.plot(epoch,test_mean, label=label)
    pl.fill_between(epoch, test_mean-test_std, test_mean+test_std, alpha=0.3)
    
    files = glob.glob(path2)
    i=0
    for file in files:

        df = pd.read_csv(file)
    
        test  = df["test_loss"].values
        epoch  = df["epoch"].values
        
        if i==0:
            test_loss = test
            test_loss_sq = test**2
            i=1
        else:
            test_loss += test
            test_loss_sq += test**2
        
    test_mean  = test_loss/len(files)
    test_std  = np.sqrt(test_loss_sq/len(files) - test_mean**2)
    
    test_mean = np.convolve(test_mean, np.ones(N)/N, mode='valid')
    test_std = np.convolve(test_std, np.ones(N)/N, mode='valid')
    epoch = np.convolve(epoch, np.ones(N)/N, mode='valid')
    
    pl.plot(epoch, test_mean, ls=':')
    
    files = glob.glob(path4)
    i=0
    for file in files:

        df = pd.read_csv(file)
    
        test  = df["test_loss"].values
        epoch  = df["epoch"].values
        
        if i==0:
            test_loss = test
            test_loss_sq = test**2
            i=1
        else:
            test_loss += test
            test_loss_sq += test**2
        
    test_mean  = test_loss/len(files)
    test_std  = np.sqrt(test_loss_sq/len(files) - test_mean**2)
    
    test_mean = np.convolve(test_mean, np.ones(N)/N, mode='valid')
    test_std = np.convolve(test_std, np.ones(N)/N, mode='valid')
    epoch = np.convolve(epoch, np.ones(N)/N, mode='valid')
    
    pl.plot(epoch, test_mean, ls=':')
    
    
    pl.ylabel("Validation Loss")
    pl.xlabel("Epoch")
    
    ax.semilogy()
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(ScalarFormatter())
    
    pl.legend()
    
    pl.show()
    
    return


# -----------------------------------------------------------------------------

def plot_loss_restricted(N=10):
    
    path1 = 'mb_lenet/mirabest_lenet_*.csv'
    path2 = 'mb_dn16lenet/mirabest_dnlenet_*.csv'
    path3 = 'mb_dn16lenet_rest/mirabest_dnlenet_*.csv'
    
    ax = pl.subplot(111)
    
    files = glob.glob(path1)
    i=0
    for file in files:

        df = pd.read_csv(file)
    
        test  = df["test_loss"].values
        epoch  = df["epoch"].values
        
        if i==0:
            test_loss = test
            test_loss_sq = test**2
            i=1
        else:
            test_loss += test
            test_loss_sq += test**2
        
    test_mean  = test_loss/len(files)
    test_std  = np.sqrt(test_loss_sq/len(files) - test_mean**2)
    
    test_mean = np.convolve(test_mean, np.ones(N)/N, mode='valid')
    test_std = np.convolve(test_std, np.ones(N)/N, mode='valid')
    epoch = np.convolve(epoch, np.ones(N)/N, mode='valid')
    print(test_std[-1])
    
    pl.plot(epoch,test_mean, label='{e}')
    pl.fill_between(epoch, test_mean-test_std, test_mean+test_std, alpha=0.3)
    
    files = glob.glob(path2)
    i=0
    for file in files:

        df = pd.read_csv(file)
    
        test  = df["test_loss"].values
        epoch  = df["epoch"].values
        
        if i==0:
            test_loss = test
            test_loss_sq = test**2
            i=1
        else:
            test_loss += test
            test_loss_sq += test**2
        
    test_mean  = test_loss/len(files)
    test_std  = np.sqrt(test_loss_sq/len(files) - test_mean**2)
    
    test_mean = np.convolve(test_mean, np.ones(N)/N, mode='valid')
    test_std = np.convolve(test_std, np.ones(N)/N, mode='valid')
    epoch = np.convolve(epoch, np.ones(N)/N, mode='valid')
    print(test_std[-1])
        
    label = r'$D_{16}$'
    
    pl.plot(epoch,test_mean, label=label)
    pl.fill_between(epoch, test_mean-test_std, test_mean+test_std, alpha=0.3)
    
    files = glob.glob(path3)
    i=0
    for file in files:

        df = pd.read_csv(file)
    
        test  = df["test_loss"].values
        epoch  = df["epoch"].values
        
        if i==0:
            test_loss = test
            test_loss_sq = test**2
            i=1
        else:
            test_loss += test
            test_loss_sq += test**2
        
    test_mean  = test_loss/len(files)
    test_std  = np.sqrt(test_loss_sq/len(files) - test_mean**2)
    
    test_mean = np.convolve(test_mean, np.ones(N)/N, mode='valid')
    test_std = np.convolve(test_std, np.ones(N)/N, mode='valid')
    epoch = np.convolve(epoch, np.ones(N)/N, mode='valid')
    print(test_std[-1])
        
    label = r'$D_{16}|_1 \{e\}$'
        
    pl.plot(epoch, test_mean, ls=':', label=label)
    pl.fill_between(epoch, test_mean-test_std, test_mean+test_std, alpha=0.3)
    
    pl.ylabel("Validation Loss")
    pl.xlabel("Epoch")
    
    ax.semilogy()
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(ScalarFormatter())
    
    pl.legend()
    pl.axis([0,600,0.12,0.45])
    
    pl.show()
    
    return

# -------------------------------------------------------------------------------

def plot_image(i=0, imagename="image.jpg"):

    imsize = 150
    
    crop     = transforms.CenterCrop(imsize)
    pad      = transforms.Pad((0, 0, 1, 1), fill=0)
    totensor = transforms.ToTensor()
    normalise= transforms.Normalize((0.0031,), (0.0350,))

    transform = transforms.Compose([
        crop,
        pad,
        totensor
    ])
    
    test_data = MBFRConfident('mirabest', train=False, transform=transform)
    
    subset_indices = [i] # select your indices here as a list
    subset = torch.utils.data.Subset(test_data, subset_indices)
    testloader_ordered = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)
    data, target = iter(testloader_ordered).next()
        
    from torchvision.utils import save_image
    save_image(data, imagename)
    
    return

# -------------------------------------------------------------------------------

def plot_bar():

    N = 3
    fri = (2.96, 1.76, 1.71)
    frii = (3.24, 2.33, 1.87)

    ind = np.arange(N)
    width = 0.35
    pl.bar(ind, fri, width, color='cyan', label='FRI')
    pl.bar(ind + width, frii, width, color='lightgrey',
        label='FRII')

    pl.ylabel('Average Number of Misclassifications')
    
    pl.xticks(ind + width / 2, (r'$\{e\}$', r'C$_{16}$', r'D$_{16}$'))
    pl.legend(loc='best')
    
    pl.show()

    return

# -------------------------------------------------------------------------------
