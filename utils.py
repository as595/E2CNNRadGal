import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# -----------------------------------------------------------

def get_lr(epoch, lr0, gamma):

    return lr0*gamma**epoch
    
# -----------------------------------------------------------
    
def get_momentum(epoch, p_i, p_f, T):

    if epoch<T:
        p = (epoch/T)*p_f + (1 - (epoch/T))*p_i
    else:
        p = p_f
    
    return p

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
