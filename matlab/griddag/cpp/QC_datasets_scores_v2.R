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
setwd("/Users/fraser/myrepos/neuralnet/matlab/griddag/cpp");
library(readr);
write_csv(as.data.frame(thedata),"n10m1000a.csv", col_names = FALSE);


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
if(FALSE){library(bnlearn);
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
}
set.seed(1000);
library(bnlearn);
a<-hc(x=as.data.frame(thedata),score="bge",iss.mu=30,iss.w=30,restart=100000,perturb=100)
print(a);
# [V1][V4][V5][V6][V2|V1:V4][V3|V1:V4][V7|V5][V8|V5:V6:V7][V10|V8][V9|V10] 
dag2=dag0;
dag2[2,c(1,4)]<-1
dag2[3,c(1,4)]<-1
dag2[7,c(5)]<-1
dag2[8,c(5,6,7)]<-1
dag2[10,c(8)]<-1
dag2[9,c(10)]<-1
print(DAGscore(n,myScore, t(dag2)))
#[1] -16397.56

# to beat = -16395.1


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

setwd("/Users/fraser/myrepos/neuralnet/matlab/griddag/cpp");
library(readr);
write_csv(as.data.frame(thedata),"n10m1000b.csv", col_names = FALSE);


myScore<-scoreparameters(n,"bge",thedata,bgepar = list(aw=30,am=30))

dag0<-matrix(data=rep(0,n*n),ncol=n);

print(DAGscore(n,myScore, t(dag0)))
# [1] -17788.25

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

set.seed(1000);
library(bnlearn);
a<-hc(x=as.data.frame(thedata),score="bge",iss.mu=30,iss.w=30,restart=100000,perturb=100)
print(a);

#  [V2][V5][V6][V8][V9][V4|V2:V5:V9][V7|V5][V1|V2:V4][V10|V7][V3|V1:V4] 
dag2=dag0;
dag2[4,c(2,5,9)]<-1
dag2[7,c(5)]<-1
dag2[1,c(2,4)]<-1
dag2[10,c(7)]<-1
dag2[3,c(1,4)]<-1
print(DAGscore(n,myScore, t(dag2)))
#[1] -16924.85


### to beat = -16922 

############################################################################################
############################################################################################
####### NETWORK WITH 20 NODES - version a-------
set.seed(100);
n<-20
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

Sigma[11,c(12:13)]<-1
Sigma[12,c(13:14)]<-1
Sigma[13,c(14)]<-1
Sigma[15,c(17,11)]<-1
Sigma[16,c(13)]<-1
Sigma[17,c(13)]<-1
Sigma[18,c(11)]<-1

Sigma[19,c(14)]<-1
Sigma[20,c(12)]<-1


for(i in 1:n){
  for(j in 1:n){
    if(j>i){Sigma[j,i]<-Sigma[i,j];}
  }
}

Means<-rnorm(n,mean=0,sd=2);

thedata<-mvrnorm(n = 1000, mu=Means,Sigma=Sigma)

setwd("/Users/fraser/myrepos/neuralnet/matlab/griddag/cpp");
library(readr);
write_csv(as.data.frame(thedata),"n20m1000.csv", col_names = FALSE);


myScore<-scoreparameters(n,"bge",thedata,bgepar = list(aw=30,am=30))

dag0<-matrix(data=rep(0,n*n),ncol=n);

print(DAGscore(n,myScore, t(dag0)))
# [1] [1] -35700.37


print(dag1)

set.seed(1000);
library(bnlearn);
a<-hc(x=as.data.frame(thedata),score="bge",iss.mu=30,iss.w=30,restart=100000,perturb=100)
print(a);
# [V1][V4][V6][V8][V9][V10][V11][V15][V16][V18][V19][V20][V2|V1:V4][V3|V1:V4][V17|V15][V14|V17][V12|V11:V14]
# [V13|V11:V14][V7|V12][V5|V7]

dag2=dag0;
dag2[2,c(1,4)]<-1
dag2[3,c(1,4)]<-1
dag2[17,c(15)]<-1
dag2[14,c(17)]<-1
dag2[12,c(11,14)]<-1
dag2[13,c(11,14)]<-1
dag2[7,c(12)]<-1
dag2[5,c(7)]<-1
print(DAGscore(n,myScore, t(dag2)))
# [1] -34017.31

# to beat -34014.5

