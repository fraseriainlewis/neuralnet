<img src="https://raw.githubusercontent.com/fraseriainlewis/neuralnet/master/neural_network_brain1.png" alt="drawing" width="200"/><img src="https://raw.githubusercontent.com/fraseriainlewis/neuralnet/master/neural_network_brain2.png" alt="drawing" width="200"/><img src="https://raw.githubusercontent.com/fraseriainlewis/neuralnet/master/neural_network_brain3.png" alt="drawing" width="200"/>
## Data Analyses with Neural Networks
This repository contains introductory step-by-step examples detailing the basic and essential tasks needed to fit and assess neural networks applied to data using C++ library [mlpack](http://mlpack.org), with [R](https://r-project.org) and [pytorch](https://pytorch.org) used for selected comparisons. 


 ## **[Setup and installation](#setup)**
 
 ## Examples 

1. **[Example 1. Linear Regression](#lr)** 

   1.1 **[Comparison with R](#lr1)** 
   
   1.2 **[with custom initial weights ](#lr2)** 
   
   1.3 **[with random initial weights](#lr3)**
   
2. **[Example 2. Two-layer forward feed network - Continuous response, MSE Loss](#ffn1)** 

   2.1 **[mlpack code](#ffn11)** 
   
   2.2 **[comparison with PyTorch](#ffn12)**  
   
3. **[Example 3. Two-layer forward feed network - Categorical response, NegLogLike Loss](#ffn2)** 

   3.1 **[mlpack code](#ffn21)** 
   
   3.2 **[comparison with PyTorch](#ffn22)**  

<a name="setup"></a>
## Setup
## Installation of [mlpack 3.1.1](http://mlpack.org)
We install [mlpack 3.1.1](http://mlpack.org) from source. The steps given here are self-contained and specific to the versions stated, additional instructions are available on the [mlpack](http://mlpack.org) website. A stock Linux docker image of [Ubuntu 18.04](https://hub.docker.com/_/ubuntu) is used. This is to allow full repeatability of the [mlpack](http://mlpack.org) installation on a clean Linux OS. It is assumed docker is already installed on the host OS ([see Docker Desktop Community Edition)](https://www.docker.com/products/docker-desktop). 

The code below assumes the top-level folder where [mlpack](http://mlpack.org) will be downloaded to, and also where this repo will be cloned to, is *$HOME/myrepos*. The simplest way to execute the code below is to open up two terminal windows, one where we will run commands on the host (e.g. macOS) and a second where we will run commands on the guest (Ubuntu 18.04 via docker). We switch between both of these, the guest terminal is where [mlpack](http://mlpack.org) is used, the host terminal for non-mlpack activities. 

```bash
# at a terminal prompt on the host (e.g. macOS)
docker pull ubuntu:18.04
# pull down docker image
docker run -it -v ~/myrepos:/files ubuntu:18.04 
# start a terminal on the guest OS - ubuntu linux with a mapping into the host OS filesystem
# /files in Ubuntu is mapped into local folder ~/myrepos
# at guest/Ubuntu 18.04 terminal prompt
> apt update
> apt install cmake clang libarmadillo-dev libboost-dev \
libboost-math-dev libboost-program-options-dev libboost-test-dev \
libboost-serialization-dev wget graphviz doxygen vim

cd /files
wget https://www.mlpack.org/files/mlpack-3.1.1.tar.gz
tar -xvzpf mlpack-3.1.1.tar.gz
mkdir -p mlpack-3.1.1/build/ && cd mlpack-3.1.1/build/
cmake ../
make -j2 
# can take a while 15-20 mins depending on system
make install
# mlpack is now installed
```
To check that [mlpack](http://mlpack.org) is installed correctly:
```bash
# in the guest terminal window
export LD_LIBRARY_PATH=/usr/local/lib
# needed so that c++ can find mlpack or else put into bash profile 
# to test the installation try this
mlpack_random_forest --help
# if this works then the installation was successful
```
## Installation of [PyTorch](https://pytorch.org)
[PyTorch](https://pytorch.org) can be installed into either a new docker container or added to the same container as [mlpack](http://mlpack.org). The additional installation commands needed are the same in each case:
```bash
# at a terminal prompt on the host (e.g. macOS)
docker run -it -v ~/myrepos:/files ubuntu:18.04 # only do this if installing into new docker container
apt update
apt install python3-pip
pip3 install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
pip3 install https://download.pytorch.org/whl/cpu/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl
pip3 install pandas
```

## Clone this reposoitory 
```bash
# open up a terminal on the host (e.g macOS) and navigate to where you want the repo to be located
# e.g. $HOME/myrepos
# create a local version of the repo in the current folder
git clone https://github.com/fraseriainlewis/neuralnet.git
```

## Test compilation of a neural network C++ program 
```bash
# in the bash terminal in the guest (Ubuntu OS)
cd /files/neuralnet
c++ linReg_ex1.cpp -o linReg_ex1 -std=c++11 -lboost_serialization -larmadillo -lmlpack -fopenmp
# this should create executable linReg1 with no errors or warnings
./linReg_ex1
# which should print output to the terminal and the last part should be as below
....
-------re-start final params------------------------
   0.4838
  -0.8266
   1.4217
  -0.2708
   0.3052
  -0.0655
   1.4153
  -0.0805
   0.4420
   1.2825
# if the above was successful then we can compile and run neural networks using mlpack
```

<a name="lr"></a>
# 1. Example 1 - Linear regression 
Linear regression is a simple special case of a neural network comprising of only one layer and an identity activation function. This is a useful starting point for learning mlpack because rather than focus on the model structure we can learn and test how the functions which fit the model operate, without being concerned about complex numerical behaviour from the model. This example fits a single model to data and focuses on the optimizer options and how to ensure we have repeatable results.

* **Optimizer configuration** Many different options

   *Parameter estimates* (weights) -  for linear regression these can easily be compared with those from [R](http://www.r-project.org) as a check that the optimization is operating as expected.  
   *Optimizer options* - such as type of optimizer, error tolerances and start/re-start conditions

* **Repeatability** It is essential to be able to repeat analyses exactly

   *Initial estimates/conditions* - for example initial starting values for weights may be set randomly or can be user-specified 
   
   *Additional iterations* - controlling what happens with future calls to the optimizer, e.g. does it start from current best estimates or else re-start from fresh estimates. 
<a name="lr1"></a>  
## 1.1 Code run through and check with R
This example uses **linReg_ex1.cpp** which is in the repo, only relevant snippets are given below.

```c++
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/init_rules/const_init.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <ensmallen.hpp>
#include <mlpack/prereqs.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;
using namespace std;

int main()
{
// set the random number seed - e.g. for random starting points in optimizer or shuffling data points
arma::arma_rng::set_seed(100); // hard code the seed for repeatability of random numbers
//arma::arma_rng::set_seed_random(); // uncomment to allow new random stream on each run of the program
```

Define a simple linear regression model as an instance of the class FNN - (FNN=Forward Feed Neural Network). This class belongs inside the ANN namespace in mlpack (ANN=Artificial Neural Network). 

```c++
/*************************************************************************************************/
/* Loss criteria used is mean squared error(MSE) - a regression problem,                         */
/* and the weights are initialized all to constant=0.9.                                          */
/*************************************************************************************************/
FFN<MeanSquaredError<>,ConstInitialization> model1(MeanSquaredError<>(),ConstInitialization(0.9));
// build layers - one linear layer and then the identity activation.
model1.Add<Linear<> >(trainData.n_rows,1);// trainData.n_rows is the no. of variables in the regression
// note mlpack matrices (armadillo library) are column major -> rows are variables, cases are columns 
model1.Add<IdentityLayer<> >();// needed = final output value is sum of weights and data
```

We have defined the model, we now define the optimizer to be used to fit the model to the data. The [ensmallen](https://www.ensmallen.org) library is used and the documentation of options can be found there. The options chosen here, for example shuffle=false and restartPolicy=true are explained later and changed in later examples.

```c++
// set up optimizer - we create an object called opt of class RMSProp
// doc at https://ensmallen.org/docs.html#rmsprop
ens::RMSProp opt(0.01, trainData.n_cols, 0.99, 1e-8, 0, 1e-8,false,true); 
                 // 1st stepSize = default
                 // 2nd batch size = all data points at once (so no batching)
                 // 3rd alpha, smoothing constant = default
                 // 4th epsilon, initialise the mean squared gradient parameter = default
                 // 5th max iterations (0 = no limit) 
                 // 6th max absolute tolerance to terminate algorithm.
                 // 7th function order of evaluation is shuffled (diff separable functions) = false 
                 // 8th resetPolicy if true parameters are reset on each call to opt() = true
```
Now fit the model and example output - the estimated regression parameters
``` c++
arma::cout<<"-------empty params------------------------"<<arma::endl;//empty params as not yet allocated
arma::cout << model1.Parameters() << arma::endl;

model1.Train(trainData, trainLabels,opt);// this is the line which fits the model

arma::cout<<"-------final params------------------------"<<arma::endl;
arma::cout << model1.Parameters() << arma::endl;
```
output from the terminal is below. The first call to model1.Parameters() is empty as the parameters are initialized as part of the later call in model1.Train(). The second call to model1.Parameters() is after model1.Train() and prints the final estimated point estimates after the training has completed using RMSprop optimizer opt(). The last estimate - 2.1024 is the bias (i.e. intercept).   

```bash
-------empty params------------------------
[matrix size: 0x0]

-------final params------------------------
    1.0222
   -2.5820
    2.9369
    0.5715
   -5.8861
   -0.8739
   20.1051
   -1.5223
    0.3459
    2.1024
```   
If we now fit the same model using [R](https://www.r-project.org) 
```R
setwd("~/myrepos/neuralnet")
features<-read.csv("features.csv",header=FALSE)
labels<-read.csv("labelsL1.csv",header=FALSE)
dat<-data.frame(v0=labels$V1,features)
summary(m1<-lm(v0~.,data=dat))
```
we get the following output
```R
Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept)  2.10739    0.03086   68.29   <2e-16 ***
V1           1.02055    0.03102   32.90   <2e-16 ***
V2          -2.58056    0.03100  -83.24   <2e-16 ***
V3           2.93324    0.03100   94.61   <2e-16 ***
V4           0.57531    0.03098   18.57   <2e-16 ***
V5          -5.88210    0.03098 -189.89   <2e-16 ***
V6          -0.86965    0.03101  -28.05   <2e-16 ***
V7          20.10959    0.03106  647.39   <2e-16 ***
V8          -1.51738    0.03101  -48.93   <2e-16 ***
V9           0.34401    0.03108   11.07   <2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 0.9758 on 990 degrees of freedom
Multiple R-squared:  0.9978,	Adjusted R-squared:  0.9978 
F-statistic: 5.035e+04 on 9 and 990 DF,  p-value: < 2.2e-16
```
The point estimates from R are almost identical to those from mlpack. They will not be identical as different optimizers are used (with different error tolerances and parameters) in addition to the usual caveats of comparing floating point estimates between programs and architectures. The R-squared is very high because this is simulated data and the additive model used here is the true generating model. 

As a final part of the above code if we want to repeat the same model fitting - exactly - from same initial starting conditions as above then we need to reset the model parameters to the same initial condition rule *ConstInitialization(0.9)* and we must ensure that the resetPolicy argument in the call to the *ens::RMSProp* object opt() is set to true (it is in the above code). 

``` c++
model1.ResetParameters();// reset parameters to their initial values - should be used with clean re-start policy = true
                         // to continue with optim from previous solution use clean re-start policy = false and do not reset
arma::cout<<"-------re-start params------------------------"<<arma::endl;
arma::cout << model1.Parameters() << arma::endl;
model1.Train(trainData, trainLabels,opt);
arma::cout<<"-------re-start final params------------------------"<<arma::endl;
arma::cout << model1.Parameters() << arma::endl;
```
which gives output identical to above but this time we also see the initialize parameters before fitting
```bash
-------re-start params------------------------
   0.9000
   0.9000
   0.9000
   0.9000
   0.9000
   0.9000
   0.9000
   0.9000
   0.9000
   0.9000

-------re-start final params------------------------
    1.0222
   -2.5820
    2.9369
    0.5715
   -5.8861
   -0.8739
   20.1051
   -1.5223
    0.3459
    2.1024
```
<a name="lr2"></a> 
## 1.2 Start model fit optimization from matrix of parameters
This example again uses **linReg_ex1.cpp** same as above only relevant snippets are given below. A key part in the snipper below is the *model2.Evaluate(trainData, trainLabels)* which allocates a matrix of initial values which can then be over-written with new custom value before any training commences.  

``` c++
// PART 2 - manually starting set weights
std::cout<<"----- PART 2 ------"<<std::endl;
// initialise weights to a constant and use MSE as the loss function
FFN<MeanSquaredError<>,ConstInitialization> model2(MeanSquaredError<>(),ConstInitialization(0.9));
// build layers
Linear<>* linearModule = new Linear<>(trainData.n_rows,1);
model2.Add(linearModule);
model2.Add<IdentityLayer<> >();

model2.Evaluate(trainData, trainLabels);// allocated matrix which can then be updated with custom starting values
arma::mat inits(trainData.n_rows+1,1);
for(i=0;i<10;i++){inits(i,0)=0.05;} //create starting values for the weights, could be all different

linearModule->Parameters() = std::move(inits); //using move to avoid memory waste

arma::cout<<"-------manual set init params------------------------"<<arma::endl;//empty params as not yet allocated
arma::cout << model2.Parameters() << arma::endl;

// set up optimizer 
ens::RMSProp opt2(0.01, 1060, 0.99, 1e-8, 0,1e-8,false,true); //https://ensmallen.org/docs.html#rmsprop.
                 
model2.Train(trainData, trainLabels,opt2);
arma::cout<<"-------final params------------------------"<<arma::endl;
arma::cout << model2.Parameters() << arma::endl;
```
Which gives output
```bash
-------manual set init params------------------------
   0.0500
   0.0500
   0.0500
   0.0500
   0.0500
   0.0500
   0.0500
   0.0500
   0.0500
   0.0500

-------final params------------------------
    1.0157
   -2.5864
    2.9313
    0.5778
   -5.8802
   -0.8699
   20.1108
   -1.5202
    0.3399
    2.1024
```
The final solution is similar but not absolutely identical to above, rounding errors.
<a name="lr3"></a> 
## 1.3 Start model fit optimization from random parameters
This example again uses **linReg_ex1.cpp** same as above only relevant snippets are given below. This uses a different initialization rule when creating the FNN object, and uses a seed reset to show how to get repeatable results

```c++
// PART 3 - random starting weights
std::cout<<"----- PART 3 ------"<<std::endl;
// initialise weights to a constant and use MSE as the loss function
FFN<MeanSquaredError<>,RandomInitialization> model3(MeanSquaredError<>(),RandomInitialization(-1.0,1.0));
// build layers
model3.Add<Linear<> >(trainData.n_rows,1);
model3.Add<IdentityLayer<> >();// needed = final output value is sum of weights and data
// set up optimizer 
ens::RMSProp opt3(0.01, 1060, 0.99, 1e-8, 10000,1e-8,false,true); //https://ensmallen.org/docs.html#rmsprop.

arma::cout<<"-------empty params------------------------"<<arma::endl;//empty params as not yet allocated
arma::cout << model3.Parameters() << arma::endl;
model3.Train(trainData, trainLabels,opt3);
arma::cout<<"-------final params------------------------"<<arma::endl;
arma::cout << model3.Parameters() << arma::endl;

arma::arma_rng::set_seed(100);
model3.ResetParameters();// reset parameters to their initial values - should be used with clean re-start policy = true
                         // to continue with optim from previous solution use clean re-start policy = false and do not reset
arma::cout<<"-------re-start params------------------------"<<arma::endl;
arma::cout << model3.Parameters() << arma::endl;
model3.Train(trainData, trainLabels,opt3);
arma::cout<<"-------re-start final params------------------------"<<arma::endl;
arma::cout << model3.Parameters() << arma::endl;

arma::arma_rng::set_seed(100);
model3.ResetParameters();// reset parameters to their initial values - should be used with clean re-start policy = true
                         // to continue with optim from previous solution use clean re-start policy = false and do not reset
arma::cout<<"-------re-start params------------------------"<<arma::endl;
arma::cout << model3.Parameters() << arma::endl;
model3.Train(trainData, trainLabels,opt3);
arma::cout<<"-------re-start final params------------------------"<<arma::endl;
arma::cout << model3.Parameters() << arma::endl;
```
Which produces output

```bash
----- PART 3 ------
-------empty (use random starts) params------------------------
[matrix size: 0x0]

-------final params------------------------
   0.4838
  -0.8266
   1.4217
  -0.2708
   0.3052
  -0.0655
   1.4153
  -0.0805
   0.4420
   1.2825

-------re-start params random start------------------------
   0.9233
  -0.3528
   0.9464
  -0.7423
   0.8032
  -0.0079
   0.9113
   0.3833
   0.7071
   0.8185

-------re-start final params------------------------
   0.4838
  -0.8266
   1.4217
  -0.2708
   0.3052
  -0.0655
   1.4153
  -0.0805
   0.4420
   1.2825

-------re-start params random start------------------------
   0.9233
  -0.3528
   0.9464
  -0.7423
   0.8032
  -0.0079
   0.9113
   0.3833
   0.7071
   0.8185

-------re-start final params------------------------
   0.4838
  -0.8266
   1.4217
  -0.2708
   0.3052
  -0.0655
   1.4153
  -0.0805
   0.4420
   1.2825
```
<a name="ffn1"></a>
# 2. Example 2. Two-layer forward feed network - Continuous response, MSE Loss
We fit a simple neural network comprising on one hidden layer with two nodes, and a sigmoid activation function. The code here is to give a simple template for a forward feed network and compares the results between mlpack and PyTorch. 

<a name="ffn11"></a> 
## 2.1 mlpack version
This example uses **ffn_ex1.cpp** which is broadly similar to **linReg_ex1.cpp** but with slight changes to the model definition - to give a hidden layer - rather than linear regression, and the additional code to provide repeated results using different starting weights has been removed. This would work exactly as in the linear regression case. The code snippet below shows the model definition.

```c++
// code above this is similar to linear reg example and reads in the raw data etc
FFN<MeanSquaredError<>,RandomInitialization> model1(MeanSquaredError<>(),RandomInitialization(-1,1));
// build layers
const size_t inputSize=trainData.n_rows;// 9 
const size_t outputSize=1;
const size_t hiddenLayerSize=2;

model1.Add<Linear<> >(trainData.n_rows, hiddenLayerSize); //hidden layer
model1.Add<SigmoidLayer<> >(); //activation in hidden layer
model1.Add<Linear<> >(hiddenLayerSize, outputSize); // output - layer - sum to single value plus bias

// set up optimizer - use Adam optimizer t
ens::Adam opt(0.001, 32, 0.9, 0.999, 1e-8, 0, 1e-5,false,true); 
                 // see https://ensmallen.org/docs.html#adam - these are largely defaults and similar to pyTorch

model1.Train(trainData, trainLabels,opt);
arma::cout<<"-------final params------------------------"<<arma::endl;
arma::cout << model1.Parameters() << arma::endl;

// Use the Predict method to get the assignments.
arma::mat assignments;
model1.Predict(trainData, assignments);

// print out the goodness of fit. Note Total error not mean error
double loss=0;
for(i=0;i<assignments.n_cols;i++){
        loss+= (assignments(0,i)-trainLabels(0,i))*(assignments(0,i)-trainLabels(0,i));
}
//loss=loss/ (double)assignments.n_cols;
arma::cout<<"MSE(sum)="<<loss<<arma::endl;// note. sum of error not divided by sample size 
arma::cout << "n rows="<<assignments.n_rows <<" n cols="<< assignments.n_cols << arma::endl;
```
The output in the terminal should end as below:
```bash
-------final params------------------------
   -0.2245
   -0.1361
    0.6666
    0.2905
   -0.7597
   -0.3231
   -0.1226
   -0.0571
    1.5225
    0.6417
    0.2516
    0.0757
   -5.0853
   -2.2369
    0.4244
    0.1586
   -0.0464
   -0.0471
   -3.0177
   -4.2481
   15.3691
   42.9384
   -0.3416

MSE(sum)=683.55
n rows=1 n cols=1000
```
The parameters are all the weights and biases from each of the hidden and output layers. These are not so easy to assign to specific place in the network structure. See next example using Torch which allocated parameters into tensors and we can compare back with these. 

<a name="ffn12"></a> 
## 2.2 PyTorch version
This example uses **ffn_ex1_torch.py** to repeat the same neural network as in Section 3.1 but using PyTorch. The complete code listing is given below. The optimizer used is Adam, same as in 3.1, and in particular this using batching of results and batch size=32. Implementing batching requires some care and the *DataLoader* class was used. Agruably the simplest option for reading in data from csv is to use pandas, then coerce to numpy array then coerce into a PyTorch tensor, as functions exist for each of these coercions. The disadvantage of this approach is that the assumptions then used by PyTorch as to batch size, specifically how many data points are processed in a batch and therefore how much data is used to do weight updating during the optimization/training stage is implicity and unclear. Using *DataLoader* allows a specific batch size to be specified. 

One other important aspect in the below code is that two loops are used when training the model, the outer loop is per epoch - one complete run of the data through the model, and the inner loop is the batching, weight updates every 32 data points. The code also has a break statement to stop early when the value of the objective function ceases to change by a sufficiently large amount. The parameters in the Adam optimizer are the same as those used in mlpack but with different starting conditions for the weights, each are started randomly with a fixed seed. Note this code takes quite some minutes to run. 

```python
# -*- coding: utf-8 -*-
import torch
import torch.utils.data as utils

# general helper libraries
import pathlib
import os
import pandas as pd
import numpy as np
from numpy.random import seed # numpy random number set function

np.random.seed(999)
torch.manual_seed(999)

# read the data - note read in using pandas then convert from dataframe to numpy array then torch tensor
features=pd.read_csv("features.csv",delimiter=",",header=None)
featuresnp=(features.to_numpy())
x = torch.from_numpy(featuresnp).double() # the cast to double is needed

labels=pd.read_csv("labelsNL1.csv",delimiter=",",header=None)
labelsnp=(labels.to_numpy())
y = torch.from_numpy(labelsnp).double()

my_dataset = utils.TensorDataset(x,y) # create your datset

dataset = utils.DataLoader(my_dataset,batch_size=32) # create your dataloader

#x = utils.DataLoader(x1, batch_size=32, shuffle=False)

#y = utils.DataLoader(y1, batch_size=32, shuffle=False)

#print(x.shape)

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
D_in, H, D_out = 9, 2, 1


# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.Sigmoid(),
    torch.nn.Linear(H, D_out)
)
loss_fn = torch.nn.MSELoss(reduction='sum')

model=model.double()

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
curloss=1e+300
abserror=1e-05
maxiters=30000

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
```
The last part of the output at the terminal from **ffn_ex1_torch.py** is 
```bash
30400 688.9507893603051
30500 688.9496733402435
30600 688.9485942419802
30700 688.9475507906353
BREAK: iter= 30778   loss= 688.94676085179 

---PARAMETERS-----

0.weight tensor([[-0.1345,  0.2795, -0.3171, -0.0566,  0.6329,  0.0782, -2.1893,  0.1507,
         -0.0526],
        [-0.2268,  0.6850, -0.7754, -0.1326,  1.5481,  0.2504, -5.1859,  0.4391,
         -0.0410]], dtype=torch.float64)
0.bias tensor([-4.1436, -3.0621], dtype=torch.float64)
2.weight tensor([[43.5959, 14.9405]], dtype=torch.float64)
2.bias tensor([-0.3613], dtype=torch.float64)
```
The loss in the PyTorch run is close but not identical to that from mlpack (689 v 683) and the parameter estimates are all very similar but are outputted in a diffferent order in each program. To see the similarity, note that mlpack is column major. The output below compares the results from mlpack and matches these to those from PyTorch. The results are very similar, and we would not expect them to be identical, and they are close enough to confirm the same formulation of neural network model is being fitted in each case. 

```bash
compare results below with the torch tensor outputs above:
   -0.2245 0.weight tensor, first col read from the bottom up - entries [1,0] and [0,0] 
   -0.1361 

    0.6666 0.weight tensor, 2nd col read from the bottom up - entries [1,1] and [0,1] 
    0.2905

   -0.7597 0.weight tensor, 3rd col read from the bottom up - entries [1,2] and [0,2] 
   -0.3231

   -0.1226 0.weight tensor, 4th col read from the bottom up - entries [1,3] and [0,3]
   -0.0571

    1.5226 0.weight tensor, 5th col read from the bottom up - entries [1,4] and [0,4]
    0.6417

    0.2516 0.weight tensor, 6th col read from the bottom up - entries [1,5] and [0,5]
    0.0757

   -5.0853 0.weight tensor, 7th col read from the bottom up - entries [1,6] and [0,6]
   -2.2369

    0.4244 0.weight tensor, 8th col read from the bottom up - entries [1,7] and [0,7]
    0.1586

   -0.0464 0.weight tensor, 9th col read from the bottom up - entries [1,8] and [0,8]
   -0.0471
   
   -3.0177 0.bias
   -4.2481 

   15.3691 2.weight
   42.9384 

   -0.3416 2.bias tensor
   
```

<a name="ffn2"></a>
# 3. Example 3. Two-layer forward feed network - Categorical response, NegLogLike Loss
We fit a linear layer with two nodes (corresponding to an output/response with two levels, i.e. binary) and a LogSoftMax activation function which maps the input values to log probabilities denoting the prediction of the input case being in output class 1 or 2. The loss function used is negative log likelihood. 

<a name="ffn21"></a> 
## 3.1 mlpack version
This example uses **ffn_ex2_bin.cpp**. Note that mlpack expects the categorical output (labels) to be coded from 1 through to N, where N is the number of unique classes. The C++ code also manually computes accuracy metrics, *sensitivity* (proportion of true positives), *specificity* (proportion of true negatives) and the overall *accuracy* (proportion of correctly predicted cases). Note as with above examples, these metrics are *not* out of sample (they use all the data, not split into training and test sets), and so not meaningful other than for comparison with the PyTorch example output below. Later examples use cross-validation. 

```c++
// snippets - with gaps see full source file - of differences from above mlpack examples

data::Load("features.csv", featureData, true);// last arg is transpose - needed as col major
data::Load("labelsBNL1.csv", labels01, true);// NOTE: these are coded 0/1

const arma::mat labels = labels01.row(0) + 1;// add +1 to all response values, mapping from 0-1 to 1-2
                                             // NegativeLogLikelihood needs 1...number of classes                                            
// Model definition
FFN<NegativeLogLikelihood<>,RandomInitialization> model1(NegativeLogLikelihood<>(),RandomInitialization(-1,1));//default

// build layers
const size_t inputSize=featureData.n_rows;// number of predictors
const size_t hiddenLayerSize=2;// 2 as binary outcome and this is the last layer fed into LogSoftMax

model1.Add<Linear<> >(inputSize, hiddenLayerSize);
model1.Add<LogSoftMax<> >();//output

// fit model
model1.Train(featureData, labels,opt);
lossAuto=model1.Evaluate(featureData, labels);// lossAuto is neg log like

// compute predictions = log probabilities for each input case
model1.Predict(featureData, assignments);

// compute the negative log like loss manually and also the Se, Sp, Acc
double lossManual=0;
double predp2,predp1,sen,spec,acc;
uword P,N,TP,TN;
P=0;
N=0;
TP=0;
TN=0;
for(i=0;i<assignments.n_cols;i++){
       lossManual+= -(assignments(labels(0,i)-1,i));// -1 as maps from 1,2 to 0,1 indexes

       predp2=assignments(1,i);//log prob of being in class 2
       predp1=assignments(0,i);//log prob of being in class 1
       
       if(labels(0,i)==2){//truth is class 2 - positives
         P++;//count number of class 2
         if(predp2>=predp1){//truth is class 2 and predict class 2
                                           TP++;//increment count of true positives TP
            }
         } //end of count positives
       if(labels(0,i)==1){//truth is class 1
         N++;// count number of class 1 - negatives
         if(predp1>=predp2){//truth is class 1 and predict class 1
                                           TN++;//increment count of true negative FN
       }
       }
                
}
sen=(double)TP/(double)P;
spec=(double)TN/(double)N;
acc=((double)TP+(double)TN)/((double)P+(double)N);//overall accuracy
  
std::cout<<"NLL (from Evaluate()) on full data set="<<lossAuto<<std::endl;
std::cout<<"NLL manual - and correct - on full data set="<<lossManual<<std::endl;
std::cout<<"P= "<<P<<"  TP="<<TP<<"  N= "<<N<<"  TN= "<<TN<<std::endl;
std::cout<<"sensitivity = "<<std::setprecision(5)<<fixed<<sen<<" specificity = "<<spec<<" accuracy= "<<acc<<std::endl;

```
This gives the following output:

```bash
#-------------------------------------------------------------------------#
#--------- log probabilities debugging on --------------------------------#
#-------------------------------------------------------------------------#

Predictions shape rows  : 2
Predictions shape cols : 1000

	1. selected output records, first 10 records
	0	[-0.000267637,-8.25953]
	1	[-0.000100894,-9.2551]
	2	[-3.23961e-05,-10.424]
	3	[-11.0338,-1.79943e-05]
	4	[-0.0597317,-2.84787]
	5	[-2.95479,-0.0535124]
	6	[-0.272727,-1.43255]
	7	[-1.01258,-0.451426]
	8	[0,-17.3028]
	9	[-0.00065943,-7.3449]

	2. selected output records, first 10 records which correspond to true label
	0	[-0.000267637]
	1	[-0.000100894]
	2	[-3.23961e-05]
	3	[-1.79943e-05]
	4	[-0.0597317]
	5	[-0.0535124]
	6	[-1.43255]
	7	[-1.01258]
	8	[0]
	9	[-0.00065943]

	3. selected output records, last 10 records
	990	[-0.68687,-0.699464]
	991	[-0.147277,-1.98821]
	992	[-2.02707e-05,-10.91]
	993	[-3.83821,-0.0217917]
	994	[-0.26634,-1.4532]
	995	[-0.00886764,-4.73271]
	996	[-10.0142,-4.81877e-05]
	997	[-2.63317e-05,-10.6386]
	998	[0,-15.5898]
	999	[-4.2746,-0.0140414]

	4. selected output records, last 10 records which correspond to true label
	990	[-0.699464]
	991	[-0.147277]
	992	[-2.02707e-05]
	993	[-0.0217917]
	994	[-0.26634]
	995	[-0.00886764]
	996	[-4.81877e-05]
	997	[-2.63317e-05]
	998	[0]
	999	[-0.0140414]
NLL (from Evaluate()) on full data set=138.496
NLL manual - and correct - on full data set=138.496
P= 419  TP=381  N= 581  TN= 549
sensitivity = 0.90931 specificity = 0.94492 accuracy= 0.93000
```



<a name="ffn32"></a> 
## 3.2 PyTorch version
This example uses **ffn_ex2_torch.py** to

```python

```
