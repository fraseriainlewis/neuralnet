<img src="https://github.com/fraseriainlewis/neuralnet/blob/master/neural_network_brain1.png" alt="drawing" width="200"/><img src="https://github.com/fraseriainlewis/neuralnet/blob/master/neural_network_brain2.png" alt="drawing" width="200"/><img src="https://github.com/fraseriainlewis/neuralnet/blob/master/neural_network_brain3.png" alt="drawing" width="200"/>
## A tutorial using C++ library [mlpack](http://mlpack.org) to build, optimize and assess different formulations of neural networks

**Table of contents**
1. [Setup](#setup)
2. [Example 1 - Linear Regression](#lr) 

   2.1 [Comparison with R](#lr1) 
   
   2.2 [Set custom initial conditions](#lr2) 
   
   2.3 [Set random initial conditions](#lr3)

<a name="setup"></a>
# 1. Setup
## 1.1 Installation of [mlpack 3.1.1](http://mlpack.org)
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
## 1.2 Clone this reposoitory 
```bash
# open up a terminal on the host (e.g macOS) and navigate to where you want the repo to be located
# e.g. $HOME/myrepos
# create a local version of the repo in the current folder
git clone https://github.com/fraseriainlewis/neuralnet.git
```

## 1.3 Test compilation of a neural network C++ program 
```bash
# in the bash terminal in the guest (Ubuntu OS)
cd /files/neuralnet
c++ linReg_ex1.cpp -o linReg_ex1 -std=c++11 -lboost_serialization -larmadillo -lmlpack
# this should create executable linReg1 with no errors or warnings
./linReg_ex1
# which should print output to the terminal.
....
-------re-start final params------------------------
   0.6245
  -0.0483
   0.6581
  -0.3430
   0.5716
  -0.2199
   0.5118
  -0.0134
   0.3289
   0.4951
# if the above was successful then we can compile and run neural networks using mlpack
```

<a name="lr"></a>
# 2. Example 1 - linear regression 
Linear regression is a simple special case of a neural network comprising of only one layer and an identity activation function. This is a useful starting point for learning mlpack because rather than focus on the model structure we can learn and test how the functions which fit the model operate, without being concerned about complex numerical behaviour from the model. This example fits a single model to data and focuses on the optimizer options and how to ensure we have repeatable results.

* **Optimizer configuration** Many different options

   *Parameter estimates* (weights) -  for linear regression these can easily be compared with those from [R](http://www.r-project.org) as a check that the optimization is operating as expected.  
   *Optimizer options* - such as type of optimizer, error tolerances and start/re-start conditions

* **Repeatability** It is essential to be able to repeat analyses exactly

   *Initial estimates/conditions* - for example initial starting values for weights may be set randomly or can be user-specified 
   
   *Additional iterations* - controlling what happens with future calls to the optimizer, e.g. does it start from current best estimates or else re-start from fresh estimates. 
<a name="lr1"></a>  
## 2.1 Code run through and check with R
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
                 // 8th resetPolicy if true parameters are reset on each call to opt() = false
```
Now fit the model and example output - the estimated regression parameters
``` c++
arma::cout<<"-------empty params------------------------"<<arma::endl;//empty params as not yet allocated
arma::cout << model1.Parameters() << arma::endl;

model1.Train(trainData, trainLabels,opt);// this is the line which fits the model

arma::cout<<"-------final params------------------------"<<arma::endl;
arma::cout << model1.Parameters() << arma::endl;
```
output from the terminal is below. The first call to model1.Parameters() is empty as the parameters are initialized as part of the later call in model1.Train(). The second call to model1.Parameters() is after model1.Train() and prints the final estimated point estimates after the training has completed using RMSprop optimizer opt(). The last estimate - 0.4506 is the bias (i.e. intercept).   

```bash
-------empty params------------------------
[matrix size: 0x0]

-------final params------------------------
  -0.0096
  -0.0018
   0.0039
   0.0086
   0.0046
  -0.0045
  -0.0035
  -0.0047
  -0.0009
   0.4506
```   
If we now fit the same model using [R](https://www.r-project.org) 
```R
setwd("~/myrepos/neuralnet")
features<-read.csv("trainfeatures.csv",header=FALSE)
labels<-read.csv("trainlabels.csv",header=FALSE)
dat<-data.frame(v0=labels$V1,features)
summary(m1<-lm(v0~.,data=dat))
```
we get the following output
```R
Coefficients:
  Estimate Std. Error t value Pr(>|t|)    
(Intercept)  0.4505660  0.0005254 857.532  < 2e-16 ***
  V1          -0.0101349  0.0009971 -10.164  < 2e-16 ***
  V2          -0.0017885  0.0007453  -2.400   0.0166 *  
  V3           0.0035829  0.0008267   4.334 1.60e-05 ***
  V4           0.0086919  0.0006017  14.445  < 2e-16 ***
  V5           0.0043800  0.0008810   4.972 7.75e-07 ***
  V6          -0.0044441  0.0006237  -7.126 1.92e-12 ***
  V7          -0.0035639  0.0006712  -5.309 1.34e-07 ***
  V8          -0.0045785  0.0006028  -7.596 6.76e-14 ***
  V9          -0.0008069  0.0007316  -1.103   0.2704    
---
  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 0.01711 on 1050 degrees of freedom
Multiple R-squared:  0.6097,	Adjusted R-squared:  0.6064 
F-statistic: 182.3 on 9 and 1050 DF,  p-value: < 2.2e-16
```
The point estimates from R are almost identical to those from mlpack. They will not be identical as different optimizers are used (with different error tolerances and parameters) in addition to the usual caveats of comparing floating point estimates between programs and architectures. 

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
  -0.0096
  -0.0018
   0.0039
   0.0086
   0.0046
  -0.0045
  -0.0035
  -0.0047
  -0.0009
   0.4506
```
<a name="lr2"></a> 
## 2.2 Start model fit optimization from matrix of parameters
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
  -0.0098
  -0.0018
   0.0037
   0.0087
   0.0047
  -0.0044
  -0.0036
  -0.0046
  -0.0008
   0.4506
```
The final solution is similar but not absolutely identical to above, rounding errors.
<a name="lr3"></a> 
## 2.3 Start model fit optimization from random parameters
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
   0.6245
  -0.0483
   0.6581
  -0.3430
   0.5716
  -0.2199
   0.5118
  -0.0134
   0.3289
   0.4951

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
   0.6245
  -0.0483
   0.6581
  -0.3430
   0.5716
  -0.2199
   0.5118
  -0.0134
   0.3289
   0.4951

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
   0.6245
  -0.0483
   0.6581
  -0.3430
   0.5716
  -0.2199
   0.5118
  -0.0134
   0.3289
   0.4951
```
