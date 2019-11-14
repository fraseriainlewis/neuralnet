rm(list=ls())
library(MASS)
library("BiDAG")
setwd("/Users/fraser/myrepos/neuralnet/matlab/griddag");

####### NETWORK WITH 4 NODES-------
set.seed(100);
n<-4
Sigma <- matrix(rep(0,n*n),ncol=n);
for(i in 1:n){Sigma[i,i]<-2;}
      
Sigma[1,c(2:3)]<-1
Sigma[2,c(3:4)]<-1
Sigma[3,c(4)]<-1
for(i in 1:n){
  for(j in 1:n){
    if(j>i){Sigma[j,i]<-Sigma[i,j];}
  }
}

Means<-rnorm(n,mean=0,sd=2);

thedata<-mvrnorm(n = 1000, mu=Means,Sigma=Sigma)

myScore<-scoreparameters(n,"bge",thedata,bgepar = list(aw=30,am=30))

dag0<-matrix(data=rep(0,n*n),ncol=n);

DAGscore(n,myScore, t(dag0))
dag0[1,c(2:3)]<-1
dag0[2,c(3:4)]<-1
dag0[3,c(4)]<-1

DAGscore(n,myScore, t(dag0))
# -6.616118110322452e+03


library(readr);
write_csv(as.data.frame(thedata),"n4m1000.csv", col_names = FALSE);

if(FALSE){
####### NETWORK WITH 5 NODES-------
set.seed(100);
n<-5
Sigma <- matrix(rep(0,n*n),ncol=n);
for(i in 1:n){Sigma[i,i]<-2;}

Sigma[1,c(2:3)]<-1
Sigma[2,c(3:5)]<-1
Sigma[3,c(4)]<-1
Sigma[4,c(5)]<-1
for(i in 1:n){
  for(j in 1:n){
    if(j>i){Sigma[j,i]<-Sigma[i,j];}
  }
}

Means<-rnorm(n,mean=0,sd=2);

thedata<-mvrnorm(n = 1000, mu=Means,Sigma=Sigma)

myScore<-scoreparameters(n,"bge",thedata,bgepar = list(aw=30,am=30))

dag0<-matrix(data=rep(0,n*n),ncol=n);

DAGscore(n,myScore, t(dag0))
dag0[1,c(2:3)]<-1
dag0[2,c(3:5)]<-1
dag0[3,c(4)]<-1
dag0[4,c(5)]<-1

DAGscore(n,myScore, t(dag0))
# -8071.677

#dyn.load("abn.so")
#dag.m<-as.integer(dag0);## this creates one long vector - filled by cols from dag.m = same as internal C reprentation so fine.
#
#res<-.Call("checkforcycles",dag.m,n
#           #,PACKAGE="abn" ## uncomment to load as package not shlib
#)
setwd("/Users/fraser/myrepos/neuralnet/matlab/griddag");
library(readr);
write_csv(as.data.frame(thedata),"n5m1000.csv", col_names = FALSE);
}

if(FALSE){
  ####### NETWORK WITH 6 NODES-------
  set.seed(100);
  n<-6
  Sigma <- matrix(rep(0,n*n),ncol=n);
  for(i in 1:n){Sigma[i,i]<-2;}
  
  Sigma[1,c(2:3)]<-1
  Sigma[2,c(3:5)]<-1
  Sigma[3,c(4)]<-1
  Sigma[4,c(5)]<-1
  Sigma[5,c(6)]<-1
  Sigma[6,c(1,2,3)]<-1
  for(i in 1:n){
    for(j in 1:n){
      if(j>i){Sigma[j,i]<-Sigma[i,j];}
    }
  }
  
  Means<-rnorm(n,mean=0,sd=2);
  
  thedata<-mvrnorm(n = 1000, mu=Means,Sigma=Sigma)
  
  myScore<-scoreparameters(n,"bge",thedata,bgepar = list(aw=30,am=30))
  
  dag0<-matrix(data=rep(0,n*n),ncol=n);
  
  DAGscore(n,myScore, t(dag0))
  dag0[1,c(2:3)]<-1
  dag0[2,c(3:5)]<-1
  dag0[3,c(4)]<-1
  dag0[4,c(5)]<-1
  dag0[5,c(6)]<-1
  dag0[6,c(1,2,3)]<-1
  
  DAGscore(n,myScore, t(dag0))
  # -9825.029
  
  #dyn.load("abn.so")
  #dag.m<-as.integer(dag0);## this creates one long vector - filled by cols from dag.m = same as internal C reprentation so fine.
  #
  #res<-.Call("checkforcycles",dag.m,n
  #           #,PACKAGE="abn" ## uncomment to load as package not shlib
  #)
  setwd("/Users/fraser/myrepos/neuralnet/matlab/griddag");
  library(readr);
  write_csv(as.data.frame(thedata),"n6m1000.csv", col_names = FALSE);
}



