rm(list=ls());
setwd("~/myrepos/neuralnet/matlab");
library(BiDAG);
myData<-read.csv("data.txt",header=FALSE);

# cell [i,j]=1 => arc from i to j
# start with null/empty DAG
m<-matrix(data=rep(0,ncol(myData)^2),ncol=ncol(myData));

myScore<-scoreparameters(ncol(m),"bge",myData,bgepar=list(aw=6,am=6))

DAGscore(ncol(m),myScore, m)

m2<-m;
m2[2,3]<-1; # DAG [x1][x2][x3|x2]

myScore<-scoreparameters(ncol(m2),"bge",myData,bgepar=list(aw=6,am=6))

DAGscore(ncol(m2),myScore, m2)
