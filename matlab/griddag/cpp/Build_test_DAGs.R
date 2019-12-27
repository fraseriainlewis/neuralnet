##############################################################################################
# This file creates the simulated data for usin with rl dag
# Input data created by computeT0.m (matlab/octave)
# output data is CSV for going to C++
#
# Important note - the set.seed() is not platform/version independent
# so data sets created here are not identical to those when run on a different machine
# 
# the generated data is saved as a workspace at the end 
##############################################################################################
rm(list=ls())
library(MASS)
library("BiDAG")
library(readr);
#setwd("/Users/fraser/myrepos/neuralnet/matlab/griddag/cpp");
setwd("/home/lewisfa/myrepos/neuralnet/matlab/griddag/cpp");

###########################################################
# SIMULATED DATA SET 1
#
###########################################################
# 1. read in covar matrix and mean vector from which to generate MVN data
covar<-as.matrix(read.csv("covarN10.csv",header=FALSE));
means<-as.matrix(read.csv("meansN10.csv",header=FALSE));

####### NETWORK WITH 10 NODES - version a-------
set.seed(100);
n<-ncol(covar);
Sigma <- covar;
Means <- means;   

# generate data   
thedata<-mvrnorm(n = 10000, mu=Means,Sigma=Sigma)

# save input parameters and the generate data
thedata.s1<-as.data.frame(thedata); # for saving
covar.s1<-covar; # for saving
means.s1<-means; # for saving

# write to CSV for import into C++
write_csv(as.data.frame(thedata),"n10m10000.csv", col_names = FALSE);

# set up function for fitting model using BIDAG
myScore<-scoreparameters(n,"bge",thedata,bgepar = list(aw=30,am=30))

# null DAG
dag0<-matrix(data=rep(0,n*n),ncol=n);

# score for null dag
print(DAGscore(n,myScore, t(dag0)))
# [1] -217155.2

# now build true generating DAG
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
# [1] -141974.5


# now try single greedy search from null dag
set.seed(100);
library(bnlearn);
a<-hc(x=as.data.frame(thedata),score="bge",iss.mu=30,iss.w=30,restart=0,perturb=0)
print(a);

# correct model right away

#################################################################
###########################################################
# SIMULATED DATA SET 2
#
###########################################################
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

# save data
covar.s2<-covar; # for saving
means.s2<-means; # for saving
thedata.s2<-as.data.frame(thedata); # for saving

write_csv(as.data.frame(thedata),"n20m10000a.csv", col_names = FALSE);

# setup
myScore<-scoreparameters(n,"bge",thedata,bgepar = list(aw=30,am=30))
# null dag
dag0<-matrix(data=rep(0,n*n),ncol=n);

# score for null dag
print(DAGscore(n,myScore, t(dag0)))
# [1] -355254.2

# now create true generating DAG and fit
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

# the true model
print(DAGscore(n,myScore, t(dag1)))
# [1] -283272.4

# now try a greedy search from null dag 
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

# less than true DAG so purely greedy search from null is not sufficient
print(DAGscore(n,myScore, t(dag2)))
# [1] -283276.7

# run 10K random hill climbs
set.seed(100);
library(bnlearn);
a<-hc(x=as.data.frame(thedata),score="bge",iss.mu=30,iss.w=30,restart=10000,perturb=100)
print(a);

#best model found - and this does give the correct optimal score

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
# [1] -283272.4

###########################################################
# SIMULATED DATA SET 3
#
###########################################################
# to define DAG we need to define unconditional means and variance for each node
covar<-as.matrix(read.csv("covarN30b.csv",header=FALSE));
means<-as.matrix(read.csv("meansN30b.csv",header=FALSE));

####### NETWORK WITH 20 NODES - version b-------
set.seed(10000);
n<-ncol(covar);
Sigma <- covar;
Means <- means;   

thedata<-mvrnorm(n = 10000, mu=Means,Sigma=Sigma)

thedata<-thedata[,1:20];
n<-20;

thedata.s3<-as.data.frame(thedata); # for saving
covar.s3<-covar; # for saving
means.s3<-means; # for saving

#write_csv(as.data.frame(thedata),"n20m10000b.csv", col_names = FALSE);


myScore<-scoreparameters(n,"bge",thedata,bgepar = list(aw=32,am=32))

dag0<-matrix(data=rep(0,n*n),ncol=n);

print(DAGscore(n,myScore, t(dag0))) ; # [1] -327597.2
# [1] -328143.7

## [a1][a2][a3|a2][a4][a5|a4][a6][a7][a8][a9][a10][a11|a9][a12|a6][a13|a5:a6][a14][a15][a16|a13][a17]
## [a18][a19|a14:a17][a20|a3:a16]

dag1=dag0;
dag1[3,2]<-1;
dag1[5,4]<-1;
dag1[11,c(9)]<-1;
dag1[12,6]<-1;
dag1[13,c(5,6)]<-1;
dag1[16,c(13)]<-1;
dag1[19,c(14,17)]<-1;
dag1[20,c(3,16)]<-1;
#dag1[30,c(1,13,25)]<-1;



print(DAGscore(n,myScore, t(dag1))) # [1] -283612.3
# [1] -283247.6

#
set.seed(100);
library(bnlearn);
a<-hc(x=as.data.frame(thedata),score="bge",iss.mu=32,iss.w=32,restart=0,perturb=0)
print(a);

# [V1][V2][V4][V6][V7][V8][V9][V10][V14][V15][V17][V18][V3|V2][V5|V4][V11|V1:V9][V12|V6][V19|V14:V17][V13|V5:V6]
# [V16|V13][V20|V3:V16] 

dag2=dag0;
dag2[3,c(2)]<-1;
dag2[5,4]<-1;
dag2[12,c(6)]<-1;
dag2[19,c(14,17)]<-1;
dag2[11,c(9)]<-1;
dag2[7,6]<-1;
dag2[13,c(5,6)]<-1;
dag2[16,c(13)]<-1;
dag2[20,c(3,16)]<-1;

print(DAGscore(n,myScore, t(dag2)))
# [1] -283250.7

set.seed(100);
library(bnlearn);
a<-hc(x=as.data.frame(thedata),score="bge",iss.mu=32,iss.w=32,restart=10,perturb=100)
print(a);

# [V1][V3][V5][V8][V9][V11][V12][V14][V16][V17][V19][V2|V1][V4|V1][V13|V11:V12][V20|V11:V19][V6|V3:V4][V15|V2]
# [V7|V6][V10|V4:V6:V7][V18|V10]

# [V1][V3][V5][V6][V7][V8][V10][V11][V14][V15][V17][V18][V2|V3][V4|V5][V9|V11][V12|V6][V13|V5:V6][V19|V14:V17]
# [V16|V13][V20|V3:V16]

dag2=dag0;
dag2[2,3]<-1;
dag2[4,5]<-1;
dag2[9,c(11)]<-1;
dag2[12,6]<-1;
dag2[13,c(5,6)]<-1;
dag2[19,c(14,17)]<-1;
dag2[16,c(13)]<-1;
dag2[20,c(3,16)]<-1;


print(DAGscore(n,myScore, t(dag2)))

######### Final step save the simulated data in workspace

save(covar.s1,means.s1,thedata.s1,covar.s2,means.s2,thedata.s2, covar.s3,means.s3,thedata.s3,
	file="rl_dag_simdata.RData");

