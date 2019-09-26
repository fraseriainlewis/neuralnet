<img src="https://raw.githubusercontent.com/fraseriainlewis/neuralnet/master/neural_network_brain1.png" alt="drawing" width="200"/><img src="https://raw.githubusercontent.com/fraseriainlewis/neuralnet/master/neural_network_brain2.png" alt="drawing" width="200"/><img src="https://raw.githubusercontent.com/fraseriainlewis/neuralnet/master/neural_network_brain3.png" alt="drawing" width="200"/>
## Introduction to Machine Learning with Neural Networks using [mlpack](http://mlpack.org)
This repository contains introductory step-by-step examples detailing the basic and essential tasks needed to fit and assess neural networks applied to data using C++ library [mlpack](http://mlpack.org), with [R](https://r-project.org), [pytorch](https://pytorch.org) and [Tensorflow](https://tensorflow.org) used for selected comparisons. 


 ## **[Setup and installation](#setup)**
 
 ## Examples 

1. **[Example 1. Linear Regression](#lr)** 

   1.1 *[Comparison with R](#lr1)* 
   
   1.2 *[with custom initial weights ](#lr2)* 
   
   1.3 *[with random initial weights](#lr3)*
   
2. **[Example 2. Two-layer forward feed network - Continuous response, MSE Loss](#ffn1)** 

   2.1 *[mlpack code](#ffn11)*
   
   2.2 *[comparison with PyTorch](#ffn12)*  
   
3. **[Example 3. Two-layer forward feed network - Categorical response, NegLogLike Loss](#ffn2)** 

   3.1 *[mlpack code](#ffn21)* 
   
   3.2 *[comparison with PyTorch](#ffn22)*
   
   3.3 *[k-fold cross-validation with mlpack](#ffn23)*

4. **[Example 4. Simple autoencoder](#ae)** 

   4.1 *[mlpack code](#ae1)* 
   
   4.2 *[comparison with PyTorch](#ae2)*
   

<a name="setup"></a>
## Setup
## Installation of [mlpack](http://mlpack.org)
We install [mlpack](http://mlpack.org) from source by cloning the mlpack [github repository](https://github.com/mlpack/mlpack.git). Note that this is a live repository and so is subject to constant change. The steps below have been tested on the repo cloned on 24th Aug 2019. A stock Linux docker image of [Ubuntu 19.10](https://hub.docker.com/_/ubuntu) is used. This is to allow full repeatability of the [mlpack](http://mlpack.org) installation on a clean Linux OS. It is assumed docker is already installed on the host OS ([see Docker Desktop Community Edition)](https://www.docker.com/products/docker-desktop). 

The code below assumes the top-level folder where [mlpack](http://mlpack.org) will be downloaded to, and also where this repo will be cloned to, is *$HOME/myrepos*. The simplest way to execute the code below is to open up two terminal windows, one where we will run commands on the host (e.g. macOS) and a second where we will run commands on the guest (Ubuntu 18.04 via docker). We switch between both of these, the guest terminal is where [mlpack](http://mlpack.org) is used, the host terminal for non-mlpack activities. 

```bash
# go to top level directory 
cd $HOME/myrepos
# at a terminal prompt on the host (e.g. macOS) clone the repo to a local folder
git clone https://github.com/mlpack/mlpack.git
# pull down docker image
docker pull ubuntu:19.10
# start a terminal on the guest OS - ubuntu linux with a mapping into the host OS filesystem
docker run -it -v ~/myrepos:/files ubuntu:19.10 
# start a terminal on the guest OS - ubuntu linux with a mapping into the host OS filesystem
# /files in Ubuntu is mapped into local folder ~/myrepos
# at guest/Ubuntu 19.10 terminal prompt
> apt update
> apt install cmake g++ clang libarmadillo-dev libboost-dev \
libboost-math-dev libboost-program-options-dev libboost-test-dev \
libboost-serialization-dev wget graphviz doxygen vim

cd /files
# change folder name so if later download a newer repo it does not overwrite - use date of download, here 24thAug2019
mv mlpack mlpack26Sep2019
mkdir -p mlpack26Sep2019/build/ && cd mlpack26Sep2019/build/
cmake ../
make 
# can take a while 15-20 mins depending on system. note "make -j2" is possible to parallelize but gave compiler error
# with this particular snapshot build, mileage may vary, serial compilation worked with no errors 
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
docker run -it -v ~/myrepos:/files ubuntu:19.10 # only do this if installing into new docker container
apt update
apt install python3-pip
pip3 install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip3 install pandas
```
## Installation of [Tensorflow](https://tensorflow.org)
[Tensorflow](https://tensorflow.org) can be installed into either a new docker container or added to the same container as [mlpack](http://mlpack.org). The additional installation commands needed are the same in each case. Note that this does not work for the latest snapshot of v2.0.0+ API, it does work with the latest stable release (which is used in the installation code below). 
```bash
# at a terminal prompt on the host (e.g. macOS)
docker run -it -v ~/myrepos:/files ubuntu:19.10 # only do this if installing into new docker container
apt update
apt install python3-pip
pip3 install --upgrade tensorflow
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
c++ linReg_ex1.cpp -o linReg_ex1 -std=c++11 -lboost_serialization -larmadillo -lmlpack -fopenmp -Wall
# this should create executable linReg_ex1 with no errors or warnings
./linReg_ex1
# which should print output to the terminal and the last part should be as below
....
-------re-start final params------------------------
    1.0010
   -2.4073
    2.8359
    0.4678
   -5.9357
   -0.9726
   19.5992
   -1.5430
    0.2949
    2.8393
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
[matrix size: 0x0]
-------final params------------------------
    1.0095
   -2.3980
    2.8343
    0.4778
   -5.9454
   -0.9827
   19.5992
   -1.5334
    0.3052
    2.8292
```   
If we now fit the same model using [R](https://www.r-project.org) 
```R
setwd("~/myrepos/neuralnet")
features<-read.csv("data/features.csv",header=FALSE)
labels<-read.csv("data/labelsL1.csv",header=FALSE)
dat<-data.frame(v0=labels$V1,features)
summary(m1<-lm(v0~.,data=dat))
```
we get the following output
```R
Coefficients:
            Estimate Std. Error  t value Pr(>|t|)    
(Intercept)  2.83425    0.03196   88.695   <2e-16 ***
V1           1.00577    0.03205   31.385   <2e-16 ***
V2          -2.40246    0.03211  -74.812   <2e-16 ***
V3           2.83563    0.03201   88.588   <2e-16 ***
V4           0.47255    0.03217   14.690   <2e-16 ***
V5          -5.94067    0.03208 -185.155   <2e-16 ***
V6          -0.97756    0.03204  -30.506   <2e-16 ***
V7          19.59917    0.03202  612.097   <2e-16 ***
V8          -1.53827    0.03206  -47.976   <2e-16 ***
V9           0.29987    0.03207    9.352   <2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 1.011 on 990 degrees of freedom
Multiple R-squared:  0.9977,	Adjusted R-squared:  0.9977 
F-statistic: 4.762e+04 on 9 and 990 DF,  p-value: < 2.2e-16
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
    1.0095
   -2.3980
    2.8343
    0.4778
   -5.9454
   -0.9827
   19.5992
   -1.5334
    0.3052
    2.8292
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
ens::RMSProp opt2(0.01, trainLabels.n_cols, 0.99, 1e-8, 0,1e-8,false,true); //https://ensmallen.org/docs.html#rmsprop.
                 
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
    1.0105
   -2.3975
    2.8359
    0.4775
   -5.9458
   -0.9827
   19.5979
   -1.5334
    0.3050
    2.8394
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
ens::RMSProp opt3(0.01, trainLabels.n_cols, 0.99, 1e-8, 10000,1e-8,false,true); //https://ensmallen.org/docs.html#rmsprop.

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
    1.0010
   -2.4073
    2.8359
    0.4678
   -5.9357
   -0.9726
   19.5992
   -1.5430
    0.2949
    2.8393

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
    1.0010
   -2.4073
    2.8359
    0.4678
   -5.9357
   -0.9726
   19.5992
   -1.5430
    0.2949
    2.8393

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
    1.0010
   -2.4073
    2.8359
    0.4678
   -5.9357
   -0.9726
   19.5992
   -1.5430
    0.2949
    2.8393
```
<a name="ffn1"></a>
# 2. Example 2. Two-layer forward feed network

**Continuous response, MSE Loss**

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

// set up optimizer - use Adam optimizer 
ens::Adam opt(0.01, trainData.n_cols, 0.9, 0.999, 1e-8, 0, 1e-5,false,true); 
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
   -0.2217
   -0.1183
    0.6000
    0.2821
   -0.6739
   -0.3516
   -0.1108
   -0.0499
    1.4475
    0.6751
    0.2419
    0.1248
   -4.8933
   -2.2408
    0.4083
    0.1457
   -0.0557
   -0.0401
   -3.1662
   -4.4231
   15.7721
   41.6386
   -0.2959

MSE=635.869
n rows=1 n cols=1000
```
The parameters are all the weights and biases from each of the hidden and output layers. These are not so easy to assign to specific place in the network structure. See next example using Torch which allocated parameters into tensors and we can compare back with these. 

<a name="ffn12"></a> 
## 2.2 PyTorch version
This example uses **ffn_ex1_torch.py** to repeat the same neural network as in Section 3.1 but using PyTorch. The complete code listing is given below. The optimizer used is Adam, same as in 3.1, and in particular this using batching of results and batch size=32. Implementing batching requires some care and the *DataLoader* class was used. Agruably the simplest option for reading in data from csv is to use pandas, then coerce to numpy array then coerce into a PyTorch tensor, as functions exist for each of these coercions. The disadvantage of this approach is that the assumptions then used by PyTorch as to batch size, specifically how many data points are processed in a batch and therefore how much data is used to do weight updating during the optimization/training stage is implicity and unclear. Using *DataLoader* allows a specific batch size to be specified. 

One other important aspect in the below code is that two loops are used when training the model, the outer loop is per epoch - one complete run of the data through the model, and the inner loop is the batching, weight updates every X data points. Given our data set is small we use all data points in each batch update. The code also has a break statement to stop early when the value of the objective function ceases to change by a sufficiently large amount. The parameters in the Adam optimizer are the same as those used in mlpack but with different starting conditions for the weights, each are started randomly with a fixed seed. Note this code takes quite some minutes to run. 

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

dataset = utils.DataLoader(my_dataset,batch_size=1000) # create your dataloader

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
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
curloss=1e+300
abserror=1e-08
maxiters=100000

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
16900 635.5367461940899
17000 635.5431125909644
17100 635.5367082955127
BREAK: iter= 17137   loss= 635.5367051535782 

---PARAMETERS-----

0.weight tensor([[-0.1164,  0.2779, -0.3455, -0.0490,  0.6642,  0.1228, -2.2044,  0.1436,
         -0.0397],
        [-0.2252,  0.6089, -0.6836, -0.1133,  1.4716,  0.2456, -4.9758,  0.4161,
         -0.0561]], dtype=torch.float64)
0.bias tensor([-4.3438, -3.1991], dtype=torch.float64)
2.weight tensor([[42.2919, 15.3779]], dtype=torch.float64)
2.bias tensor([-0.3060], dtype=torch.float64)
torch.Size([1000, 1])
...
MSE on full data set= [635.53670518]
```
The loss in the PyTorch run is close but not identical to that from mlpack (635.5 v 635.9) and the parameter estimates are all very similar but are outputted in a diffferent order in each program. To see the similarity, note that mlpack is column major. The output below compares the results from mlpack and matches these to those from PyTorch. The results are very similar, and we would not expect them to be identical, and they are close enough to confirm the same formulation of neural network model is being fitted in each case. 

```bash
compare results below with the torch tensor outputs above:
   -0.2217 0.weight tensor, first col read from the bottom up - entries [1,0] and [0,0] 
   -0.1183 

    0.6000 0.weight tensor, 2nd col read from the bottom up - entries [1,1] and [0,1] 
    0.2821 

   -0.6739 0.weight tensor, 3rd col read from the bottom up - entries [1,2] and [0,2] 
   -0.3516 

   ...

   -0.0557 0.weight tensor, 9th col read from the bottom up - entries [1,8] and [0,8]
   -0.0401
   
   -3.1662 0.bias
   -4.4231

   15.7721 2.weight
   41.6386 

   -0.2959 2.bias tensor
   
```

<a name="ffn2"></a>
# 3. Example 3. Two-layer forward feed network
**Categorical response, NegLogLike Loss**

We fit a linear layer with two nodes (corresponding to an output/response with two levels, i.e. binary) and a LogSoftMax activation function which maps the input values to log probabilities denoting the prediction of the input case being in output class 1 or 2. The loss function used is negative log likelihood. Note that the predicted outcome class is taken to be that with the higher probability of the two, i.e. an implicit cut-off of 0.5, with real data this could potentially be tuned using ROC etc. 

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

<a name="ffn22"></a> 
## 3.2 PyTorch version
This example uses **ffn_ex2_bin_torch.py** and fits the same model as in **ffn_ex2_bin.cpp** although without the extra code to compute accuracy measures. This example uses an Adam() optimizer with default parameters, and the mlpack version uses the same type of optimizer and parameters (although the starting conditions are different between these two versions as randomly set within each of C++ and python). 

```python
# snippet for the model definition, linear layer with two nodes and LogSoftMax 
# activation to produce log probs

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.LogSoftmax(dim=1) 
    )
loss_fn = torch.nn.NLLLoss() # neg log like loss

model=model.double()
```
This gives the following output:

```bash
2600 138.4964831651668
2700 138.49194628939256
2800 138.48913947560112
2900 138.48744828558551
BREAK: iter= 2947   current loss= 138.48691488261667 	 previous 138.4869248666776 	 -9.98406093799531e-06 

---PARAMETERS-----

0.weight tensor([[-1.2315e+00, -1.2979e-01,  4.2458e+00, -3.5071e-01, -9.1897e-02,
         -7.4486e-04,  1.5494e-01, -2.5916e-02,  1.5279e-01],
        [ 1.3023e+00,  3.3805e-01, -4.2129e+00,  1.7719e-01, -3.4586e-01,
         -2.1554e-01, -5.9367e-02, -1.4222e-01,  2.0074e-01]],
       dtype=torch.float64)
0.bias tensor([ 1.1435, -0.9950], dtype=torch.float64)
torch.Size([1000, 2])
first 10 and last 10 probabilities output from model

[[-2.58633893e-04 -8.26022635e+00]
 [-9.55498697e-05 -9.25591003e+00]
 [-2.97088629e-05 -1.04240800e+01]
 [-1.10340875e+01 -1.61421062e-05]
 [-5.97058091e-02 -2.84803034e+00]
 [-2.95487716e+00 -5.34905048e-02]
 [-2.72809440e-01 -1.43228735e+00]
 [-1.01250811e+00 -4.51467092e-01]
 [-3.05603134e-08 -1.73035636e+01]
 [-6.45924837e-04 -7.34515036e+00]]
---
[[-6.86800010e-01 -6.99534895e-01]
 [-1.47285295e-01 -1.98812273e+00]
 [-1.82633813e-05 -1.09106217e+01]
 [-3.83790538e+00 -2.17740121e-02]
 [-2.66330828e-01 -1.45322768e+00]
 [-8.83961060e-03 -4.73292900e+00]
 [-1.00143687e+01 -4.47532593e-05]
 [-2.39585089e-05 -1.06391990e+01]
 [-1.69524888e-07 -1.55902662e+01]
 [-4.27402402e+00 -1.40235045e-02]]
NLL on full data set= [138.48690495] 

```
The predicted probabilites and negloglik values produced by mlpack and pytorch are almost identical.  

<a name="ffn23"></a> 
## 3.3 k-fold cross-validation with mlpack
This example uses **ffn_ex3_cv_bin.cpp** and uses several custom functions defined and implemented in **cv_header.hpp**. This examples repeats the example in 3.1 but additionally compute k-fold cross-validation estimates of *sensitivity*, *specificity* and *accuracy*. The example uses 10 folds. The CV code simply partitions the total data set into k-folds, and if the total data set size is not a multiple of k then the last fold is ragged (e.g. n=105 will have 9 folds of size 10 and 1 fold of 15). the snippet below contains some of the key new lines. We do some setup for the k-fold data and then loop through, grabbing a matrix view of the features and a vector view of the labels, associated with each fold.   

```c++
...
/**************************************************************************************************/
/** we now reset the model parameters and use 10-fold cross validation to estimate the out of    **/
/** sample accuracy                                                                              **/
/*******************************CV folds **********************************************************/
/**************************************************************************************************/
uvec foldinfo;
umat indexes;
uword verbose=0;
setupFolds(totalDataPts,nFolds,&foldinfo,&indexes,verbose);// pass pointers as call constructor in function

uvec trainIdx,testIdx;
uword curFold;

//curFold=2;//1-based
//arma::mat assignments2;
for(curFold=1;curFold<=nFolds;curFold++){
  std::cout<<"Processing fold "<<curFold<<" of "<<nFolds<<std::endl;
  getFold(nFolds,curFold,indexes,foldinfo,&trainIdx,&testIdx,verbose);// note references and pointers, refs for

  model1.ResetParameters();// reset parameters to their initial values - should be used with clean re-start policy = true
  arma::cout<<"-------re-start empty params------------------------"<<arma::endl;//empty params as not yet allocated
  arma::cout << model1.Parameters() << arma::endl;
  model1.Train(featureData.cols(trainIdx), labels.cols(trainIdx),opt);
  arma::cout<<"-------re-start final params------------------------"<<arma::endl;
  arma::cout << model1.Parameters() << arma::endl;

  // Use the Predict method to get the assignments.
  //arma::mat assignments2;
  model1.Predict(featureData.cols(testIdx), assignments);

...
} // end of k-fold loop

```
This gives the following output:

```bash
data is exact multiple of number of folds 100 lastFoldSize 100
Processing fold 1 of 10
-------re-start empty params------------------------
  -0.9569
   0.5090
   0.0497
  -0.7270
  -0.1949
  -0.3512
  -0.4596
  -0.1957
   0.2470
  -0.3391
  -0.2576
   0.3777
   0.3602
   0.9576
   0.7043
   0.6197
  -0.1943
   0.2445
  -0.8926
  -0.5218

-------re-start final params------------------------
  -1.5445
   1.0989
  -0.5155
  -0.1618
   4.1959
  -4.7443
  -0.5882
  -0.0736
   0.0646
  -0.1591
   0.1740
  -0.0517
   0.7355
   0.5857
   0.7564
   0.5620
   0.0151
   0.0358
   0.4289
  -1.8019

SIZE=100
P= 38  TP=33  N= 62  TN= 59
sensitivity = 0.86842 specificity = 0.95161 accuracy= 0.92000
Processing fold 2 of 10
-------re-start empty params------------------------
  -0.7096
  -0.0173

...

Processing fold 10 of 10
-------re-start empty params------------------------
   0.9011
  -0.1312
  -0.1270
   0.7684
   0.1311
  -0.3305
  -0.6499
  -0.6583
  -0.9079
   0.3433
  -0.7067
  -0.3934
  -0.4293
  -0.1467
  -0.3419
  -0.6472
  -0.3377
  -0.1410
   0.3335
   0.6602

-------re-start final params------------------------
  -0.8368
   1.6075
   0.1005
   0.5396
   4.0246
  -4.2259
  -0.9253
  -0.3853
  -0.1550
  -0.4102
  -0.4851
  -0.6120
  -0.1760
  -0.3992
  -0.4511
  -0.5405
  -0.2985
  -0.1840
   1.5897
  -0.5639

SIZE=100
P= 44  TP=38  N= 56  TN= 54
sensitivity = 0.86364 specificity = 0.96429 accuracy= 0.92000
in-sample Se= 0.90931 in-sample Sp = 0.94492
out-sample 10-fold mean Se = 0.90761 out-sample 10-fold mean Sp = 0.94405
```
The last two lines show the accuracy metrics on the performance of the model on the same data it was fitted to (all the data), this is the in-sample estimates, and then compares this with the out of sample estimate which is the mean performance over the 10 folds. 

<a name="ae"></a>
# 4. Example 4. Simple autoencoder
**Single hidden layer with one node**
An encoder - at its simplest is analogous to linear principle components analysis (PCA) - it uses a bottleneck (a set of hidden layers) which reduces the dimension of the data. The general process is that high dimensional data is passed through the neural network and pushed through the bottleneck and the output dimension is the same as the input dimension, and is reconstructed data. We fit the model so the reconstructed data is as close as possible to the original data, but there will be information loss due to the bottlneck. The intuitive idea is that the key features of the data will be retained, giving a similar - but lower dimensional - representation of the original data. By capturing the encoded data, this is analogous to PCA components (if the activation function is linear), at least up to a rotation, reflection or translation. Non-linear activations, and multiple hidden layers make this a very rich methodology for dimension reduction. The examples here are the simplest possible: one hidden layer with one node and linear activation. The code demonstrates how to capture the encoded data and compute the loss.  

<a name="ae1"></a> 
## 4.1 mlpack version
This example uses **AE.cpp**. Only key parts are shown below, for example how to set up the network using sequential(), which is more natural than the raw method in the previous examples, and how to manually extract the encoded and decoded data. Input dimension = 8, hidden layer dimension =1, output dimension = 8. We use the .Forward() method to push the data through the layers (using the estimated weights), which is one way to get the encoded data.

```c++
...
/**************************************************************************************************/
/** Load the training set - separate files for features and lables (regression) **/
/** note - data is read into matrix in column major, e.g. each new data point is a column - opposite from data file **/
arma::mat trainData, trainLabels;
uword i,j;
data::Load("AEdata.csv", trainData, true);//  
data::Load("AEdata.csv", trainLabels, true);// use same data as train and test is the same in AE

// print out to check the data is read in correctly
arma::cout << "n rows="<<trainData.n_rows <<" n cols="<< trainData.n_cols << arma::endl;
//arma::cout << "n rows="<< trainLabels.n_rows <<" n cols"<< trainLabels.n_cols << arma::endl;

/**************************************************************************************************/
/** MODELS - linear autoencoder with one hidden layer with one node **/

const size_t inputSize=trainData.n_rows;// =8 
const size_t outputSize=trainData.n_rows;// =8
const size_t hiddenLayerSize=1;//

/** could build model direct in layers - this works fine but more elegant using sequential here see below **/
/* not using this code *//*
FFN<MeanSquaredError<>,RandomInitialization> model0(MeanSquaredError<>(),RandomInitialization(-1,1));
model0.Add<Linear<> >(inputSize, hiddenLayerSize);
model0.Add<IdentityLayer<> >();
model0.Add<Linear<> >(hiddenLayerSize, outputSize);
model0.Add<IdentityLayer<> >();
*/

FFN<MeanSquaredError<>,RandomInitialization> model1(MeanSquaredError<>(),RandomInitialization(-1,1));
// Encoder
Sequential<>* encoder = new Sequential<>();
encoder->Add<Linear<> >(inputSize, hiddenLayerSize);
encoder->Add<IdentityLayer<> >();
// Decoder
Sequential<>* decoder = new Sequential<>();
decoder->Add<Linear<> >(hiddenLayerSize,outputSize);
decoder->Add<IdentityLayer<> >();

// now link all the layers together
model1.Add<IdentityLayer<> >(); // layer-0 this is needed a can't start with sequential as first layer
model1.Add(encoder);            // layer-1
model1.Add(decoder);            // layer-2

// set up optimizer 
ens::Adam opt(0.001,trainData.n_cols, 0.9, 0.999, 1e-8, 0, 1e-5,false,true); //https://ensmallen.org/docs.html#adam
                 

arma::cout<<"-------empty params------------------------"<<arma::endl;//empty params as not yet allocated
arma::cout << model1.Parameters() << arma::endl;
model1.Train(trainData, trainLabels,opt);
arma::cout<<"-------final params------------------------"<<arma::endl;
arma::cout << model1.Parameters() << arma::endl;

// Use the Predict method to get the assignments.
double loss1=model1.Evaluate(trainData,trainLabels);
arma::mat assignments;
model1.Predict(trainData, assignments);

//check forward - full forward, all layers, should be same as Predict
arma::mat assignments2;
model1.Forward(trainData, assignments2,0,2);// start layer 0, stop layer 2

// manual computation of MSE loss - from predict
double loss2=0;
for(i=0;i<assignments.n_cols;i++){
  for(j=0;j<assignments.n_rows;j++){
    loss2+= (assignments(j,i)-trainLabels(j,i))*(assignments(j,i)-trainLabels(j,i));
  }
}

// manual computation of MSE loss - from predict
double loss3=0;
for(i=0;i<assignments2.n_cols;i++){
  for(j=0;j<assignments2.n_rows;j++){
    loss3+= (assignments2(j,i)-trainLabels(j,i))*(assignments2(j,i)-trainLabels(j,i));
  }
}


// check loss manually
loss1=loss1;//get mean error from predict
loss2=loss2/ (double)assignments.n_cols;//get mean error from forward
loss3=loss3/ (double)assignments2.n_cols;//get mean error from forward
arma::cout<<"MSE auto="<<loss1<<arma::endl;
arma::cout<<"MSE manual predict="<<loss2<<arma::endl;
arma::cout<<"MSE manual forward(0,2)="<<loss3<<arma::endl;


// now compute just encoded data - push into layer 0 and out of layer 1
arma::mat assignments3;
model1.Forward(trainData, assignments3,0,1);// this works 
arma::cout<<"FORWARD SEQ 0 1 ncol"<<assignments3.n_cols<<" FORWARD SEQ 0 1 nrow"<<assignments3.n_rows<<arma::endl;

// now compute decoded data - push encoded data into layer 2 and out of layer 2
arma::mat assignments4;
model1.Forward(assignments3, assignments4,2,2);// this works but 
arma::cout<<"FORWARD 0 1 ncol"<<assignments4.n_cols<<" FORWARD 1 nrow"<<assignments4.n_rows<<arma::endl;

//now send to disk - encoded data
mat ass4T = assignments3.t();
ass4T.save("c_encoded1.csv", csv_ascii); 

//now send to disk - decoded data
mat ass5T = assignments4.t();
ass5T.save("c_decoded8.csv", csv_ascii); 

...

```

```bash
n rows=8 n cols=438
-------empty params------------------------
[matrix size: 0x0]

-------final params------------------------
   0.3450
   0.2518
   0.1047
  -0.3409
  -0.3545
  -0.3687
  -0.3338
  -0.3096
   1.1270
   0.4377
   0.3147
   0.1308
  -0.4432
  -0.4577
  -0.4753
  -0.4313
  -0.4001
   0.8588
   0.3215
   0.2091
   1.0491
   0.7900
   0.9175
   0.8754
   0.8350

MSE auto=5.91827
MSE manual predict=5.91827
MSE manual forward(0,2)=5.91827
FORWARD SEQ 0 1 ncol438 FORWARD SEQ 0 1 nrow1
FORWARD 0 1 ncol438 FORWARD 1 nrow8

```

<a name="ae2"></a> 
## 4.2 Pytorch version
This example uses **AE_torch.py**. The general idea is the same as in the mlpack code, the parameter estimates are difficult to compare due to the equivalence of the linear transformation of the references axes (rotation etc), but the losses came be compared as an intuitive check, and these are similar, 5.92 in mlpack and 5.95 in pytorch. Note that this loss estimate is manually computed in the code, to ensure it is directly comparable with that from mlpack. To get the encoded data we extract data out of the relevant subset of layers in the torch sequential model framework (rather than push data through layers, it is stored as part of this). 

```python

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

new_model2 = torch.nn.Sequential(*list(model.children())[2:4:1]) ## only keep first two layers
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
```

```bash
torch.Size([438, 8])
torch.Size([438, 8])
Sequential(
  (0): Linear(in_features=8, out_features=1, bias=True)
  (1): Identity()
  (2): Linear(in_features=1, out_features=8, bias=True)
  (3): Identity()
)
Sequential(
  (0): Linear(in_features=8, out_features=1, bias=True)
  (1): Identity()
)
0 1.715515742858413
100 1.4177103432437188
200 1.2447982699915816
300 1.1176322918071444
400 1.0192697696308524
500 0.9403487709692049
600 0.8822497407762648
700 0.8415934698863026
800 0.8133681358925989
900 0.7935171904337369
1000 0.7792329346694701
1100 0.7687454247717685
1200 0.7609888315691693
1300 0.7552784794785009
1400 0.7511161487104931
1500 0.7481098719911505
1600 0.7459483900755002
1700 0.7443912774392966
BREAK: iter= 1789   loss= 0.7433677820221017 

---PARAMETERS-----

0.weight tensor([[ 0.1370,  0.0757,  0.0157, -0.3563, -0.2745, -0.2650, -0.2338, -0.2397]],
       dtype=torch.float64)
0.bias tensor([-0.0510], dtype=torch.float64)
2.weight tensor([[ 0.6091],
        [ 0.3648],
        [ 0.1621],
        [-0.7784],
        [-0.6588],
        [-0.6535],
        [-0.5559],
        [-0.5704]], dtype=torch.float64)
2.bias tensor([1.5363, 0.8159, 0.4138, 0.1911, 0.0172, 0.1509, 0.2036, 0.1740],
       dtype=torch.float64)
MANUAL LOSS= 5.946862542963064 

---PARAMETERS-----

0.weight tensor([[ 0.1370,  0.0757,  0.0157, -0.3563, -0.2745, -0.2650, -0.2338, -0.2397]],
       dtype=torch.float64)
0.bias tensor([-0.0510], dtype=torch.float64)
---PARAMETERS-----

0.weight tensor([[ 0.6091],
        [ 0.3648],
        [ 0.1621],
        [-0.7784],
        [-0.6588],
        [-0.6535],
        [-0.5559],
        [-0.5704]], dtype=torch.float64)
0.bias tensor([1.5363, 0.8159, 0.4138, 0.1911, 0.0172, 0.1509, 0.2036, 0.1740],
       dtype=torch.float64)
-----------

[[ 0.17450165]
 [-0.129184  ]
 [ 0.34673723]
 [ 0.18031122]
 [ 0.34778009]
 [-0.80109972]
 [ 0.25190284]
 [ 0.34042402]
 [-0.20006302]
 [-0.76083321]]
[[ 1.64257051  0.87954356  0.44205581  0.05531233 -0.09776347  0.03681768
   0.1066209   0.07450145]
 [ 1.45759097  0.76875068  0.39282539  0.29170825  0.10231608  0.23527126
   0.27544793  0.24773156]
 [ 1.74748181  0.94237983  0.46997688 -0.07875982 -0.21123876 -0.07573544
   0.01087051 -0.02374615]
 [ 1.6461092   0.88166305  0.44299759  0.05079003 -0.10159103  0.03302122
   0.10339121  0.07118753]
 [ 1.74811703  0.9427603   0.47014594 -0.07957161 -0.21192583 -0.07641694
   0.01029076 -0.02434103]
 [ 1.04831692  0.52361735  0.28390129  0.81474295  0.54499945  0.67435714
   0.64898395  0.63100956]
 [ 1.68971674  0.90778164  0.4546033  -0.00493854 -0.14875829 -0.01376273
   0.0635915   0.03034983]
 [ 1.74363634  0.9400766   0.46895345 -0.07384548 -0.20707938 -0.07160987
   0.01438019 -0.02014493]
 [ 1.41441748  0.74289206  0.38133521  0.34688212  0.14901384  0.28158954
   0.31485148  0.28816278]
 [ 1.07284386  0.53830768  0.29042888  0.78339858  0.51847036  0.64804364
   0.62659871  0.60804051]]

```
