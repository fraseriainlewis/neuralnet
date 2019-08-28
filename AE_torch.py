# -*- coding: utf-8 -*-
import torch
import torch.utils.data as utils
from torch.autograd import Variable
# general helper libraries
import pathlib
import os
import pandas as pd
import numpy as np
from numpy.random import seed # numpy random number set function

np.random.seed(9990)# 9990
torch.manual_seed(9990) # 9990

# read the data - note read in using pandas then convert from dataframe to numpy array then torch tensor
features=pd.read_csv("data/AEdata.csv",delimiter=",",header=None) 
featuresnp=(features.to_numpy())
x = torch.from_numpy(featuresnp).double() # the cast to double is needed

labels=pd.read_csv("data/AEdata.csv",delimiter=",",header=None) # 1000 observations
labelsnp=(labels.to_numpy())
y = torch.from_numpy(labelsnp).double()

#print(y)
print(y.shape)
#print(x)
print(x.shape)

my_dataset = utils.TensorDataset(x,y) # create your datset

dataset = utils.DataLoader(my_dataset,batch_size=438) # create your dataloader


model = torch.nn.Sequential(
	torch.nn.Linear(8, 1),
	torch.nn.Identity(), # also Identity
	torch.nn.Linear(1, 8),
	torch.nn.Identity() # also Identity
      )

print(model)
print(model[0:2:1])

loss_fn = torch.nn.MSELoss(reduction='mean')

model=model.double()

learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

curloss=1e+300
abserror=1e-05
maxiters=100000


minLossOverall=1e+300

for t in range(maxiters): # for each epoch - all training data run through once
    running_loss=0.0
    i=0
    for input, target in dataset: # for each batch of training data update the current weights
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        #if i % 25 == (25-1):    # print every 25 mini-batches
        #    print('[%d, %5d] loss: %.3f' %
        #          (t + 1, i + 1, running_loss))
        #    #running_loss = 0.0
        i=i+1  
    #print("t=",t," ",i," ",running_loss)
    if np.absolute(running_loss-curloss) <abserror:
        # have good enough solution so stop
        print("BREAK: iter=",t," ","loss=",running_loss,"\n")
        break
    else: 
        curloss=running_loss # copy loss

    if ((t%100)==0):
        print(t, curloss)


# print out parameters
print("---PARAMETERS-----\n")
for name, param in model.named_parameters():
    if param.requires_grad:
        print (name, param.data)


#modelNew=model[]
preds = model(x)
#print(preds.shape)
prednp=preds.detach().numpy()
#print("first 10 and last 10 probabilities output from model\n")
#print(prednp[0:10:1,:])

nrows=prednp.shape[0]
ncols=prednp.shape[1]
myloss=0.0
for i in range(nrows):
	for j in range(ncols):
		myloss+= (prednp[i,j]-labelsnp[i,j])*(prednp[i,j]-labelsnp[i,j])

print("MANUAL LOSS=",myloss/y.shape[0],"\n")

## print encoding of 8-dim vector into
new_model = torch.nn.Sequential(*list(model.children())[0:2:1]) ## only keep first two layers
#print(new_model(x))
encodepreds2dim=new_model(x)


# print out parameters
print("---PARAMETERS-----\n")
for name, param in new_model.named_parameters():
    if param.requires_grad:
        print (name, param.data)

new_model2 = torch.nn.Sequential(*list(model.children())[2:4:1]) ## only keep layers 2 and 3
#print(new_model2(encodepreds2dim))
decodepreds8dim=new_model2(encodepreds2dim);

# print out parameters
print("---PARAMETERS-----\n")
for name, param in new_model2.named_parameters():
    if param.requires_grad:
        print (name, param.data)

print("-----------\n")
# get encoded values in np
encodepreds2dimnp=encodepreds2dim.detach().numpy()
print(encodepreds2dimnp[0:10:1,:])

encodepd=pd.DataFrame(data=encodepreds2dimnp)

# get decoded values in np
decodepreds8dimnp=decodepreds8dim.detach().numpy()
print(decodepreds8dimnp[0:10:1,:])

# this works arithmetically just fine!
encodepd=pd.DataFrame(data=encodepreds2dimnp)
decodepd=pd.DataFrame(data=decodepreds8dimnp)
encodepd.to_csv("torch_encoded.csv", encoding='utf-8', index=False)
decodepd.to_csv("torch_decoded.csv", encoding='utf-8', index=False)


