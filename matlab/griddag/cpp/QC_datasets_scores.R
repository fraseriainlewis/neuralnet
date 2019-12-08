rm(list=ls())
library(MASS)
library("BiDAG")
library(abn);
setwd("/Users/fraser/myrepos/neuralnet/matlab/griddag/cpp");

####### NETWORK WITH 10 NODES - version a-------
set.seed(100);
n<-10
Sigma <- matrix(rep(0,n*n),ncol=n);
for(i in 1:n){Sigma[i,i]<-2;}
      
Sigma[1,c(2:3)]<-1
Sigma[2,c(3:4)]<-1
Sigma[3,c(4)]<-1
Sigma[5,c(7,1)]<-1
Sigma[6,c(8)]<-1
Sigma[7,c(8)]<-1
Sigma[8,c(1)]<-1

Sigma[9,c(10)]<-1
Sigma[10,c(5,6)]<-1

for(i in 1:n){
  for(j in 1:n){
    if(j>i){Sigma[j,i]<-Sigma[i,j];}
  }
}

Means<-rnorm(n,mean=0,sd=2);

thedata<-mvrnorm(n = 1000, mu=Means,Sigma=Sigma)

myScore<-scoreparameters(n,"bge",thedata,bgepar = list(aw=30,am=30))

dag0<-matrix(data=rep(0,n*n),ncol=n);

print(DAGscore(n,myScore, t(dag0)))
# [1] -17799.74

dag1=dag0;
dag1[1,c(2:3)]<-1
dag1[2,c(3:4)]<-1
dag1[3,c(4)]<-1
dag1[5,c(7,1)]<-1
dag1[6,c(8)]<-1
dag1[7,c(8)]<-1
dag1[8,c(1)]<-1

dag1[9,c(10)]<-1
dag1[10,c(5,6)]<-1

print(DAGscore(n,myScore, t(dag1)))
#[1] -16657.01

print(dag1)


####
library(bnlearn);
a<-hc(x=as.data.frame(thedata),score="bge",iss.mu=30,iss.w=30,restart=0,perturb=0)

# [V1][V4][V5][V6][V9][V2|V1:V4][V3|V1:V4][V7|V5][V10|V9][V8|V5:V6:V7] 
dag2=dag0;
dag2[2,c(1,4)]<-1
dag2[3,c(1,4)]<-1
dag2[7,c(5)]<-1
dag2[8,c(5,6,7)]<-1
dag2[10,c(9)]<-1
#dag2[9,c(10)]<-1
print(DAGscore(n,myScore, t(dag2)))
#[1] -16396.77

library(bnlearn);
a<-hc(x=as.data.frame(thedata),score="bge",iss.mu=30,iss.w=30,restart=10000,perturb=10)

# [V1][V4][V5][V6][V9][V2|V1:V4][V3|V1:V4][V7|V5][V10|V9][V8|V5:V6:V7] 
dag2=dag0;
dag2[2,c(1,4)]<-1
dag2[3,c(1,4)]<-1
dag2[7,c(5)]<-1
dag2[8,c(5,6,7)]<-1
dag2[10,c(8)]<-1
dag2[9,c(10)]<-1
print(DAGscore(n,myScore, t(dag2)))
#[1] -16397.56


# check for cycles
library(abn)
thedata2 <- thedata; colnames(thedata2)<-paste("v",1:10,sep="");
mydists <- list(v1="gaussian",v2="gaussian",v3="gaussian",v4="gaussian",v5="gaussian",
                v6="gaussian",v7="gaussian",v8="gaussian",v9="gaussian",v10="gaussian");
colnames(dag0) <- rownames(dag0) <- colnames(thedata2)
colnames(dag1) <- rownames(dag1) <- colnames(thedata2)
myres <- fitabn(dag.m = dag0, data.df = as.data.frame(thedata2), data.dists = mydists)
myres2 <- fitabn(dag.m = dag1, data.df = as.data.frame(thedata2), data.dists = mydists)
graphics.off()
plotabn(dag1,data.dists=mydists); # there is a cycle between 2,3,4

## add a cycle - a check
dag1[2,9]=1;
#myres2 <- fitabn(dag.m = dag1, data.df = as.data.frame(thedata2), data.dists = mydists)


setwd("/Users/fraser/myrepos/neuralnet/matlab/griddag/cpp");
library(readr);
write_csv(as.data.frame(thedata),"n10m1000a.csv", col_names = FALSE);

###########
####### NETWORK WITH 10 NODES - version b-------
set.seed(100);
n<-10
Sigma <- matrix(rep(0,n*n),ncol=n);
for(i in 1:n){Sigma[i,i]<-2;}

Sigma[1,c(2:3)]<-1
Sigma[2,c(3:4)]<-1
Sigma[3,c(4)]<-1
Sigma[5,c(7,1)]<-1
Sigma[6,c(3)]<-1
Sigma[7,c(3)]<-1
Sigma[8,c(1)]<-1

Sigma[9,c(4)]<-1
Sigma[10,c(2)]<-1

for(i in 1:n){
  for(j in 1:n){
    if(j>i){Sigma[j,i]<-Sigma[i,j];}
  }
}

Means<-rnorm(n,mean=0,sd=2);

thedata<-mvrnorm(n = 1000, mu=Means,Sigma=Sigma)

myScore<-scoreparameters(n,"bge",thedata,bgepar = list(aw=25,am=25))

dag0<-matrix(data=rep(0,n*n),ncol=n);

print(DAGscore(n,myScore, t(dag0)))
# [1] -17772.41

dag1=dag0;
dag1[1,c(2:3)]<-1
dag1[2,c(3:4)]<-1
dag1[3,c(4)]<-1
dag1[5,c(7,1)]<-1
dag1[6,c(3)]<-1
dag1[7,c(3)]<-1
dag1[8,c(1)]<-1

dag1[9,c(4)]<-1
dag1[10,c(2)]<-1

print(DAGscore(n,myScore, t(dag1)))
#[1] -17107.43

print(dag1)

# check for cycles
library(abn)
thedata2 <- thedata; colnames(thedata2)<-paste("v",1:10,sep="");
mydists <- list(v1="gaussian",v2="gaussian",v3="gaussian",v4="gaussian",v5="gaussian",
                v6="gaussian",v7="gaussian",v8="gaussian",v9="gaussian",v10="gaussian");
colnames(dag0) <- rownames(dag0) <- colnames(thedata2)
colnames(dag1) <- rownames(dag1) <- colnames(thedata2)
myres <- fitabn(dag.m = dag0, data.df = as.data.frame(thedata2), data.dists = mydists)
myres2 <- fitabn(dag.m = dag1, data.df = as.data.frame(thedata2), data.dists = mydists)
graphics.off()
plotabn(dag1,data.dists=mydists); # there is a cycle between 2,3,4

## add a cycle - a check
dag1[4,10]=1;
#myres2 <- fitabn(dag.m = dag1, data.df = as.data.frame(thedata2), data.dists = mydists)

setwd("/Users/fraser/myrepos/neuralnet/matlab/griddag/cpp");
library(readr);
write_csv(as.data.frame(thedata),"n10m1000b.csv", col_names = FALSE);




