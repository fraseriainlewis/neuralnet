# neuralnet
A compilation of simple examples of using C++ to build, optimize and fit different formulations of neural networks to data sets using the [mlpack](http://mlpack.org) library. 

## Installation of MLPACK 3.1.1
We install [mlpack](http://mlpack.org) from source. The steps given here are self-contained and specific to the versions stated, additional instructions are available on the [mlpack](http://mlpack.org) website. A stock Linux docker image of [Ubuntu 18.04](https://hub.docker.com/_/ubuntu) is used. This is to allow full repeatability of the [mlpack](http://mlpack.org) installation on a clean Linux OS. It is assumed docker is already installed on the host OS (in this case macOS Mojave 10.14.5 and [Docker Desktop Community edition 2.0.0.3](https://www.docker.com/products/docker-desktop), but this should work on any host system with a docker installation. It has also been tested on Linux CentOS).  

The code snippet below assumes the top-level folder where mlpack will be downloaded to, and also where this repo will be cloned to, is *$HOME/myrepos*. The simplest way to execute the code below is to open up two terminal windows, one which runs commands on the host (e.g. macOS) and a second which runs commands on the guest (Ubuntu 18.04 via docker). We switch between both of these, the guest terminal is where [mlpack](http://mlpack.org) is used, the host terminal for non-mlpack activities. 

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
make install
# mlpack is now installed - can take a while 15-20 mins depending on system
```
To test [mlpack](http://mlpack.org) is installed correctly try
```bash
# now set up variable so that c++ can find mlpack - needed every time
# or else put into bash profile
export LD_LIBRARY_PATH=/usr/local/lib
# to test the installation try this
mlpack_random_forest --help
# if this works then the installation was successful
```

## Compiled a neural network C++ program 
```bash
# open up a terminal on the host (e.g macOS) and navigate to where
# you want the repo to be located, e.g. $HOME/myrepos
# create a local version of the repo in the current folder
git clone https://github.com/fraseriainlewis/neuralnet.git
# now go to the bash terminal in the guest (Ubuntu OS)
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


```c++
require 'redcarpet'
markdown = Redcarpet.new("Hello World!")
puts markdown.to_html
```
