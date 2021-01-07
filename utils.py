import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import configparser as ConfigParser
import ast

import math
import numpy as np
import pylab as pl
import pandas as pd
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)

# ----------------------------------------------------------

def parse_args():
    """
        Parse the command line arguments
        """
    parser = argparse.ArgumentParser()
    parser.add_argument('-C','--config', default="myconfig.txt", required=True, help='Name of the input config file')
    
    args, __ = parser.parse_known_args()
    
    return vars(args)

# -----------------------------------------------------------

def parse_config(filename):
    
    config = ConfigParser.SafeConfigParser(allow_no_value=True)
    config.read(filename)
    
    # Build a nested dictionary with tasknames at the top level
    # and parameter values one level down.
    taskvals = dict()
    for section in config.sections():
        
        if section not in taskvals:
            taskvals[section] = dict()
        
        for option in config.options(section):
            # Evaluate to the right type()
            try:
                taskvals[section][option] = ast.literal_eval(config.get(section, option))
            except (ValueError,SyntaxError):
                err = "Cannot format field '{0}' in config file '{1}'".format(option,filename)
                err += ", which is currently set to {0}. Ensure strings are in 'quotes'.".format(config.get(section, option))
                raise ValueError(err)

    return taskvals, config

# -----------------------------------------------------------

def build_mask(s, margin=2, dtype=torch.float32):
    mask = torch.zeros(1, 1, s, s, dtype=dtype)
    c = (s-1) / 2
    t = (c - margin/100.*c)**2
    sig = 2.
    for x in range(s):
        for y in range(s):
            r = (x - c) ** 2 + (y - c) ** 2
            if r > t:
                mask[..., x, y] = math.exp((t - r)/sig**2)
            else:
                mask[..., x, y] = 1.
    return mask
    
# -----------------------------------------------------------

def train(model, trainloader, optimiser, device):

    train_loss = 0.0

    model.train()
    for batch_idx, (data, labels) in enumerate(trainloader):
        data, labels = data.to(device), labels.to(device)

        optimiser.zero_grad()

        p_y = F.softmax(model(data), dim=1)
        loss = model.loss(p_y, labels)
            
        train_loss += loss.item() * data.size(0)

        loss.backward()
        optimiser.step()

    train_loss /= len(trainloader.dataset)
    return train_loss

# -----------------------------------------------------------------------------

def test(model, testloader, device):

    correct = 0
    total = 0
    test_loss = 0.0

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(testloader):
            data, labels = data.to(device), labels.to(device)

            p_y = F.softmax(model(data), dim=1)
            loss = model.loss(p_y, labels)
                
            test_loss += loss.item() * data.size(0)

            preds = p_y.argmax(dim=1, keepdim=True)
            correct += preds.eq(labels.view_as(preds)).sum().item()

        test_loss /= len(testloader.dataset)
        accuracy = correct / len(testloader.dataset)

    return test_loss, accuracy

# -----------------------------------------------------------------------------

def test_mc(model, testloader, device, T=100):

    correct = 0
    total = 0
    test_loss = 0.0

    model.eval()
    model.enable_dropout()
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(testloader):
            data, labels = data.to(device), labels.to(device)

            _prob = torch.zeros(labels.size()[0],2).to(device=device)
            for _ in range(T):
                p_y = F.softmax(model(data), dim=1)
                _prob += p_y
                
            _prob /= T
            loss = model.loss(_prob,labels)
            
            test_loss += loss.item() * data.size(0)

            preds = p_y.argmax(dim=1, keepdim=True)
            correct += preds.eq(labels.view_as(preds)).sum().item()

        test_loss /= len(testloader.dataset)
        accuracy = correct / len(testloader.dataset)

    return test_loss, accuracy

# -----------------------------------------------------------------------------

def positionimage(x, y, ax, ar, zoom=0.5):
    """Place image from file `fname` into axes `ax` at position `x,y`."""
    
    imsize = ar.shape[0]
    if imsize==151: zoom=0.24
    if imsize==51: zoom = 0.75
    im = OffsetImage(ar, zoom=zoom)
    im.image.axes = ax
    
    ab = AnnotationBbox(im, (x,y), xycoords='data')
    ax.add_artist(ab)
    
    return

# -----------------------------------------------------------------------------

def make_linemarker(x,y,dx,col,ax):
    
    xs = [x-0.5*dx,x+0.5*dx]
    for i in range(0,y.shape[0]):
        ys = [y[i],y[i]]
        ax.plot(xs,ys,marker=",",c=col,alpha=0.1,lw=5)
    
    return

# -----------------------------------------------------------------------------

def fr_rotation_test(model, data, target, idx):
    
    T = 100
    rotation_list = range(0, 180, 20)
    #print("True classification: ",target[0].item())
    
    image_list = []
    outp_list = []
    inpt_list = []
    for r in rotation_list:
        
        # make rotated image:
        rotation_matrix = torch.Tensor([[[np.cos(r/360.0*2*np.pi), -np.sin(r/360.0*2*np.pi), 0],
                                         [np.sin(r/360.0*2*np.pi), np.cos(r/360.0*2*np.pi), 0]]])
        grid = F.affine_grid(rotation_matrix, data.size(), align_corners=True)
        data_rotate = F.grid_sample(data, grid, align_corners=True)
        image_list.append(data_rotate)
        
        # get straight prediction:
        model.eval()
        x = model(data_rotate)
        p = F.softmax(x,dim=1)
                                         
        # run 100 stochastic forward passes:
        model.enable_dropout()
        output_list, input_list = [], []
        for i in range(T):
            x = model(data_rotate)
            input_list.append(torch.unsqueeze(x, 0))
            output_list.append(torch.unsqueeze(F.softmax(x,dim=1), 0))
                                         
        # calculate the mean output for each target:
        output_mean = np.squeeze(torch.cat(output_list, 0).mean(0).data.numpy())
                                             
        # append per rotation output into list:
        outp_list.append(np.squeeze(torch.cat(output_list, 0).data.numpy()))
        inpt_list.append(np.squeeze(torch.cat(input_list, 0).data.numpy()))

        #print ('rotation degree', str(r), 'Predict : {} - {}'.format(output_mean.argmax(),output_mean))

    preds = np.array([0,1])
    classes = np.array(["FRI","FRII"])
    
    outp_list = np.array(outp_list)
    inpt_list = np.array(inpt_list)
    rotation_list = np.array(rotation_list)

    colours=["b","r"]

    #fig1, (a0, a1) = pl.subplots(2, 1, gridspec_kw={'height_ratios': [8,1]})
    fig2, (a2, a3) = pl.subplots(2, 1, gridspec_kw={'height_ratios': [8,1]})

    eta = np.zeros(len(rotation_list))
    for i in range(len(rotation_list)):
        x = outp_list[i,:,0]
        y = outp_list[i,:,1]
        eta[i] = overlapping(x, y)

    #a0.set_title("Input")
    if np.mean(eta)>=0.01:
        a2.set_title(r"$\langle \eta \rangle = $ {:.2f}".format(np.mean(eta)))
    else:
        a2.set_title(r"$\langle \eta \rangle < 0.01$")

    dx = 0.8*(rotation_list[1]-rotation_list[0])
    for pred in preds:
        col = colours[pred]
        #a0.plot(rotation_list[0],inpt_list[0,0,pred],marker=",",c=col,label=str(pred))
        a2.plot(rotation_list[0],outp_list[0,0,pred],marker=",",c=col,label=classes[pred])
        for i in range(rotation_list.shape[0]):
        #    make_linemarker(rotation_list[i],inpt_list[i,:,pred],dx,col,a0)
            make_linemarker(rotation_list[i],outp_list[i,:,pred],dx,col,a2)
        
    #a2.plot(rotation_list, eta)
    
    #a0.legend()
    a2.legend(loc='center right')
    #a0.axis([0,180,0,1])
    #a0.set_xlabel("Rotation [deg]")
    a2.set_xlabel("Rotation [deg]")
    #a1.axis([0,180,0,1])
    a3.axis([0,180,0,1])
    #a1.axis('off')
    a3.axis('off')
    
    imsize = data.size()[2]
    mask = build_mask(imsize, margin=1)
            
    for i in range(len(rotation_list)):
        inc = 0.5*(180./len(rotation_list))
        #positionimage(rotation_list[i]+inc, 0., a1, image_list[i][0, 0, :, :].data.numpy(), zoom=0.32)
        positionimage(rotation_list[i]+inc, 0., a3, mask[0,0,:,:]*image_list[i][0, 0, :, :].data.numpy(), zoom=0.32)
        
    
    #fig1.tight_layout()
    fig2.tight_layout()

    #fig1.subplots_adjust(bottom=0.15)
    fig2.subplots_adjust(bottom=0.15)

    #pl.show()
    fig2.savefig("./rotations/rotationtest_"+str(idx)+".png")
    
    pl.close()
    
    return np.mean(eta), np.std(eta)

# -----------------------------------------------------------------------------

def overlapping(x, y, beta=0.1):

    n_z = 100
    z = np.linspace(0,1,n_z)
    dz = 1./n_z
    
    norm = 1./(beta*np.sqrt(2*np.pi))
    
    n_x = len(x)
    f_x = np.zeros(n_z)
    for i in range(n_z):
        for j in range(n_x):
            f_x[i] += norm*np.exp(-0.5*(z[i] - x[j])**2/beta**2)
        f_x[i] /= n_x
    
    
    n_y = len(y)
    f_y = np.zeros(n_z)
    for i in range(n_z):
        for j in range(n_y):
            f_y[i] += norm*np.exp(-0.5*(z[i] - y[j])**2/beta**2)
            
        f_y[i] /= n_y
    
    
    eta_z = np.zeros(n_z)
    eta_z = np.minimum(f_x, f_y)
        
    #pl.subplot(111)
    #pl.plot(z, f_x, label=r"$f_x$")
    #pl.plot(z, f_y, label=r"$f_y$")
    #pl.plot(z, eta_z, label=r"$\eta_z$")
    #pl.legend()
    #pl.show()

    return np.sum(eta_z)*dz

# -----------------------------------------------------------------------------

def eval_overlap():

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
    
    diff = olap1 - olap2

    for i in range(len(targ1)):
        print("{} {} {} {} {:.2f} {:.2f} {:.2f} {:.2f}".format(i,targ1[i],np.argmax(p1[i,:]),np.argmax(p2[i,:]),olap1[i],olap2[i],slap1[i],slap2[i]))

    better = diff[np.where(diff>0.01)]
    better_class = targ1[np.where(diff>0.01)]
    
    worse = diff[np.where(diff<-0.01)]
    worse_class = targ1[np.where(diff<-0.01)]
    
    print("Improved: ",len(better),np.mean(better),np.mean(better_class))
    print("Worse: ",len(worse),np.mean(worse),np.mean(worse_class))
    
    print("Better:")
    for i in range(len(targ1)):
        if diff[i]>0.01:
            print(i, targ1[i], diff[i])
            
    print("Worse:")
    for i in range(len(targ1)):
        if diff[i]<-0.01:
            print(i, targ1[i], diff[i])
    
    print("Low D16:")
    n=0
    for i in range(len(targ1)):
        if olap2[i]<=0.01:
            #print(i, targ2[i], olap2[i])
            n+=1
    print(n)
   
    print("Low e:")
    n=0
    for i in range(len(targ1)):
        if olap1[i]<=0.01:
            #print(i, targ1[i], olap1[i])
            n+=1
    print(n)
    
    return
