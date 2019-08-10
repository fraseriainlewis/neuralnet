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

np.random.seed(9999)
torch.manual_seed(9999)

# read the data - note read in using pandas then convert from dataframe to numpy array then torch tensor
features=pd.read_csv("features.csv",delimiter=",",header=None) 
featuresnp=(features.to_numpy())
x = torch.from_numpy(featuresnp).double() # the cast to double is needed

labels=pd.read_csv("labelsBNL1.csv",delimiter=",",header=None) # 1000 observations
labelsnp=(labels.to_numpy())
y = torch.from_numpy(labelsnp).double()

#print(y)

my_dataset = utils.TensorDataset(x,y) # create your datset

dataset = utils.DataLoader(my_dataset,batch_size=1000) # create your dataloader

#x = utils.DataLoader(x1, batch_size=32, shuffle=False)

#y = utils.DataLoader(y1, batch_size=32, shuffle=False)

#print(x.shape)

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
D_in, H, D_out = 9, 2, 1


# Use the nn package to define our model and loss function.

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.LogSoftmax(dim=1) #uncomment alternative to crossentropy
    )
loss_fn = torch.nn.NLLLoss() #uncomment alternative to crossentropy


model=model.double()

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update. 
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
curloss=1e+300
abserror=1e-05
maxiters=100000

#data   = Variable(data,requires_grad=False)
 #   target = Variable(target.long(),requires_grad=False)
minLossOverall=1e+300
for t in range(maxiters): # for each epoch - all training data run through once
    running_loss=0.0
    i=0
    for input, target in dataset: # for each batch of training data update the current weights

        input   = Variable(input,requires_grad=False)
        target = Variable(target.long(),requires_grad=False)

        optimizer.zero_grad()
        output = model(input)
        
        loss = loss_fn(output, target.view(-1))# the target.view(-1) is same as flatten(), converts 32x1 matrix into 32 length vector - essential here
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()*1000 # data set size
        #if i % 25 == (25-1):    # print every 25 mini-batches
        #    print('[%d, %5d] loss: %.3f' %
        #          (t + 1, i + 1, running_loss))
        #    #running_loss = 0.0
        i=i+1  
        #print("t=",t," ",i," ",running_loss)
        #if running_loss<minLossOverall
    if np.absolute(running_loss-curloss) <abserror:
        # have good enough solution so stop
        print("BREAK: iter=",t," ","current loss=",running_loss,"\t previous",curloss,"\t",running_loss-curloss,"\n")
        break
    else: 
        #print("No BREAK: iter=",t," ","loss=",running_loss,"\t",curloss,"\t",running_loss-curloss,"\n")
        curloss=running_loss # copy loss

    if ((t%100)==0):
        print(t, curloss)
    #print(t, curloss)

# print out parameters
print("---PARAMETERS-----\n")
for name, param in model.named_parameters():
    if param.requires_grad:
        print (name, param.data)




preds = model(x)
print(preds.shape)
prednp=preds.detach().numpy()
print("first 10 and last 10 probabilities output from model\n")
print(prednp[0:10:1,:])
print("---")
print(prednp[990:1000:1,:])

nrows=prednp.shape[0]
myloss=0.0
for i in range(nrows):
    myloss+= -prednp[i,(labelsnp[i])]


print("NLL on full data set=",myloss,"\n")
#print(prednp[0:100:1,:])
#print("---")
#print(prednp[2000:2100:1,:])




