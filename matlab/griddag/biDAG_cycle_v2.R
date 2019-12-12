rm(list=ls())
library(MASS)
library("BiDAG")

##--------------------------------
## generate simulated data 
##--------------------------------
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
##--------------------------------
## fit DAG - but this has a cycle?
##--------------------------------
dag0<-matrix(data=rep(0,n*n),ncol=n);
dag0[1,]=c(0,1,1,1)
dag0[2,]=c(0,0,0,1)
dag0[3,]=c(0,1,0,0)
dag0[4,]=c(0,0,1,0)

DAGscore(n,myScore, t(dag0));# transpose because dag0 is not adjacency,each row a child
# [1] -6375.535

##--------------------------------
## fit DAG - try with abn
##--------------------------------
library(abn)
thedata2 <- thedata; colnames(thedata2)<-paste("v",1:4,sep="");
mydists <- list(v1="gaussian",v2="gaussian",v3="gaussian",v4="gaussian");
colnames(dag0) <- rownames(dag0) <- colnames(thedata2)
myres <- fitabn(dag.m = dag0, data.df = as.data.frame(thedata2), data.dists = mydists)
# DAG definition is not acyclic!
# also try transpose - since abn does not use adjacency matrix
myres <- fitabn(dag.m = t(dag0), data.df = as.data.frame(thedata2), data.dists = mydists)
#DAG definition is not acyclic!
plotabn(dag0,data.dists=mydists); # there is a cycle between 2,3,4

