rm(list=ls())
library(MASS)
library("BiDAG")
library(abn);
setwd("/Users/fraser/myrepos/neuralnet/matlab/griddag/cpp");

###########################################################
# to define DAG we need to define unconditional means and variance for each node
covar<-as.matrix(read.csv("covarChain.csv",header=FALSE));
means<-as.matrix(read.csv("meansChain.csv",header=FALSE));


####### NETWORK WITH 10 NODES - version a-------
set.seed(100);
n<-ncol(covar);
Sigma <- covar;
Means <- means;   
   
thedata<-mvrnorm(n = 10000, mu=Means,Sigma=Sigma)
setwd("/Users/fraser/myrepos/neuralnet/matlab/griddag/cpp");
library(readr);
write_csv(as.data.frame(thedata),"n10Chainm10000.csv", col_names = FALSE);


myScore<-scoreparameters(n,"bge",thedata,bgepar = list(aw=30,am=30))

dag0<-matrix(data=rep(0,n*n),ncol=n);

print(DAGscore(n,myScore, t(dag0)))
# [1] -217166.3

dag1=dag0;
dag1[2,1]<-1
dag1[3,2]<-1
dag1[4,3]<-1
dag1[5,4]<-1
dag1[6,5]<-1
dag1[7,6]<-1
dag1[8,7]<-1
dag1[9,8]<-1
dag1[10,9]<-1

print(DAGscore(n,myScore, t(dag1)))
# [1] -141975.2

print(dag1)


#
set.seed(100);
library(bnlearn);
a<-hc(x=as.data.frame(thedata),score="bge",iss.mu=30,iss.w=30,restart=0,perturb=0)
print(a);

#
set.seed(100);
library(bnlearn);
a<-hc(x=as.data.frame(thedata),score="bge",iss.mu=30,iss.w=30,restart=10000,perturb=100)
print(a);

# [V8][V7|V8][V9|V8][V6|V7][V10|V9][V5|V6][V4|V5][V3|V4][V2|V3][V1|V2] 
dag2=dag0;
dag2[7,8]<-1
dag2[9,8]<-1
dag2[6,7]<-1
dag2[10,9]<-1
dag2[5,6]<-1
dag2[4,5]<-1
dag2[3,4]<-1
dag2[2,3]<-1
dag2[1,2]<-1

print(DAGscore(n,myScore, t(dag2)))
# [1] -141975.2

#################################################################
###########################################################
# to define DAG we need to define unconditional means and variance for each node
covar<-as.matrix(read.csv("covarN20a.csv",header=FALSE));
means<-as.matrix(read.csv("meansN20a.csv",header=FALSE));


####### NETWORK WITH 20 NODES - version a-------
set.seed(100);
n<-ncol(covar);
Sigma <- covar;
Means <- means;   

thedata<-mvrnorm(n = 10000, mu=Means,Sigma=Sigma)
setwd("/Users/fraser/myrepos/neuralnet/matlab/griddag/cpp");
library(readr);
write_csv(as.data.frame(thedata),"n20m10000a.csv", col_names = FALSE);


myScore<-scoreparameters(n,"bge",thedata,bgepar = list(aw=30,am=30))

dag0<-matrix(data=rep(0,n*n),ncol=n);

print(DAGscore(n,myScore, t(dag0)))
# [1] -355259.9

# [a1][a2|a1][a3][a4|a1][a5][a6|a4:a3][a7|a6][a8][a9][a10|a4:a6:a7][a11][a12][a13|a11:a12][a14][a15|a2][a16]
# [a17][a18|a10][a19][a20|a11:a19]

dag1=dag0;
dag1[2,1]<-1;
dag1[4,1]<-1;
dag1[6,c(4,3)]<-1;
dag1[7,6]<-1;
dag1[10,c(4,6,7)]<-1;
dag1[13,c(11,12)]<-1;
dag1[15,2]<-1;
dag1[18,10]<-1;
dag1[20,c(11,19)]<-1;

print(DAGscore(n,myScore, t(dag1)))
# [1] -283273.3

print(dag1)


#
set.seed(100);
library(bnlearn);
a<-hc(x=as.data.frame(thedata),score="bge",iss.mu=30,iss.w=30,restart=0,perturb=0)
print(a);

# [V3][V5][V8][V9][V11][V12][V14][V16][V17][V19][V6|V3][V13|V11:V12][V20|V11:V19][V10|V3:V6][V4|V3:V10][V18|V10]
# [V1|V4][V7|V4:V10][V2|V1][V15|V2]

dag2=dag0;
dag2[6,3]<-1
dag2[13,c(11,12)]<-1
dag2[20,c(11,19)]<-1
dag2[10,c(3,6)]<-1
dag2[4,c(3,10)]<-1
dag2[18,10]<-1
dag2[1,4]<-1
dag2[7,c(4,10)]<-1
dag2[2,1]<-1
dag2[15,2]<-1;

print(DAGscore(n,myScore, t(dag2)))
# [1] -283277

set.seed(100);
library(bnlearn);
a<-hc(x=as.data.frame(thedata),score="bge",iss.mu=30,iss.w=30,restart=10000,perturb=100)
print(a);

# [V1][V3][V5][V8][V9][V11][V12][V14][V16][V17][V19][V2|V1][V4|V1][V13|V11:V12][V20|V11:V19][V6|V3:V4][V15|V2]
# [V7|V6][V10|V4:V6:V7][V18|V10]

dag2=dag0;
dag2[2,1]<-1
dag2[4,1]<-1
dag2[13,c(11,12)]<-1
dag2[20,c(11,19)]<-1
dag2[6,c(3,4)]<-1
dag2[15,2]<-1
dag2[7,6]<-1
dag2[10,c(4,6,7)]<-1
dag2[18,10]<-1


print(DAGscore(n,myScore, t(dag2)))

