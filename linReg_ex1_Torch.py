# -*- coding: utf-8 -*-
import torch

# general helper libraries
import pathlib
import os
import pandas as pd
import numpy as np
from numpy.random import seed # numpy random number set function

# read the data - note read in using pandas then convert from dataframe to numpy array then torch tensor
features=pd.read_csv("features.csv",delimiter=",",header=None)
featuresnp=(features.to_numpy())
x = torch.from_numpy(featuresnp).double() # the cast to double is needed

labels=pd.read_csv("labelsL1.csv",delimiter=",",header=None)
labelsnp=(labels.to_numpy())
y = torch.from_numpy(labelsnp).double()

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, D_out = 1000, 9, 1


# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, D_out)
    #torch.nn.ReLU(),
    #torch.nn.Linear(H, D_out)
)
loss_fn = torch.nn.MSELoss(reduction='sum')

model=model.double()

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
curloss=1e+300
abserror=1e-03

for t in range(5001):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    
    if np.absolute(curloss-loss.item()) <abserror:
        # have good enough solution so stop
        print("iter=",t," ","loss=",loss.item(),"\n")
        break
    else: 
        curloss=loss.item() # copy loss

    if ((t%100)==0):
        print(t, loss.item())

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()
    

# print out parameters
print("---PARAMETERS-----\n")
for name, param in model.named_parameters():
    if param.requires_grad:
        print (name, param.data)
