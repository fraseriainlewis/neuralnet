# neuralnet
A compilation of simple examples of using C++ to build, optimize and fit different formulations of neural networks to data sets using the [mlpack](http://mlpack.org) library. 

## Installation of MLPACK 
We install [mlpack](http://mlpack.org) from source. The steps given here are self-contained and specific to the versions stated, additional instructions are available on the [mlpack](http://mlpack.org) website. A stock Linux docker image of [Ubuntu 18.04](https://hub.docker.com/_/ubuntu) is used. This is to allow full repeatability of the [mlpack](http://mlpack.org) installation on a clean linux OS. It is assumed docker is already installed on the host OS (in this case macOS Mojave 10.14.5 and Docker Desktop Community edition 2.0.0.3, but this should work on any host system with a docker installation. It has also been tested on Linux CentOS).  

```bash
# at a terminal prompt on the host (e.g. macOS)
docker pull ubuntu:18.04
# pull down image
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
To test mlpack is installed correct and programs can be compiled
```bash
# now set up variable so that c++ can find mlpack 
export LD_LIBRARY_PATH=/usr/local/lib
# to test the installation try this
mlpack_random_forest --help
# if this works then the installation was successful
```

# Compiled a test C++ program using the Forward Feed Neural network API 
First clone this repo in a terminal on the host

cd /files/neural
c++ linReg1.cpp -o linReg1 -std=c++11 -lboost_serialization -larmadillo -lmlpack
```


```c++
require 'redcarpet'
markdown = Redcarpet.new("Hello World!")
puts markdown.to_html
```
