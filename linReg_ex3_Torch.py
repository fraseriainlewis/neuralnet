# -*- coding: utf-8 -*-
import torch
import torch.utils.data as utils

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

my_dataset = utils.TensorDataset(x,y) # create your datset

dataset = utils.DataLoader(my_dataset,batch_size=50) # create your dataloader

#x = utils.DataLoader(x1, batch_size=32, shuffle=False)

#y = utils.DataLoader(y1, batch_size=32, shuffle=False)

#print(x.shape)

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
D_in, H, D_out = 9, 1, 1


# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    #torch.nn.Sigmoid(),
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
abserror=1e-05
maxiters=500

for t in range(maxiters):
    running_loss=0.0
    i=0
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 20 == (20-1):    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (t + 1, i + 1, running_loss))
            running_loss = 0.0
        i=i+1  


# print out parameters
print("---PARAMETERS-----\n")
for name, param in model.named_parameters():
    if param.requires_grad:
        print (name, param.data)
