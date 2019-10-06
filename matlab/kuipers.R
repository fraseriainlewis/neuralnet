library("BiDAG")
setwd("/Users/fraser/myrepos/neuralnet/matlab");
thedata<-read.csv("data_matrix.txt",header=FALSE);
# setup for score calc
myScore<-scoreparameters(3,"bge",thedata,bgepar = list(aw=6,am=6))
# define dags - note - format is each row is child/node, each col parent
# opposite from what BiDAG needs - so use transpose
dag0<-matrix(data=c(0,0,0,
                    0,0,0,
                    0,0,0), ncol=ncol(thedata),byrow=TRUE);
dag1<-matrix(data=c(0,0,1,
                    1,0,0,
                    0,1,0), ncol=ncol(thedata),byrow=TRUE);
dag2<-matrix(data=c(0,0,1,
                    1,0,1,
                    0,1,0), ncol=ncol(thedata),byrow=TRUE);
dag3<-matrix(data=c(0,0,0,
                    1,0,0,
                    0,0,0), ncol=ncol(thedata),byrow=TRUE);

DAGscore(3,myScore, t(dag0))
DAGscore(3,myScore, t(dag1))
DAGscore(3,myScore, t(dag2))
DAGscore(3,myScore, t(dag3))

setwd("/Users/fraser/myrepos/neuralnet/Rabn");
dyn.load("abn.so")
dag.m<-as.integer(dag0);## this creates one long vector - filled by cols from dag.m = same as internal C reprentation so fine.

res<-.Call("checkforcycles",dag.m,3
           #,PACKAGE="abn" ## uncomment to load as package not shlib
)

set.seed(100);
Sigma <- matrix(rep(0,20*20),ncol=20);
for(i in 1:20){Sigma[i,i]<-2;
               if(i<17){Sigma[i,c(i+1,i+2,i+3)]<-1;}
}
Sigma[1,c(2:3)]<-1

Means<-rnorm(20,mean=0,sd=2);

thedata<-mvrnorm(n = 1000, mu=Means,Sigma=Sigma)

myScore<-scoreparameters(20,"bge",thedata,bgepar = list(aw=30,am=30))

dag0<-matrix(data=rep(0,20*20),ncol=20);

DAGscore(20,myScore, t(dag0))

dyn.load("abn.so")
dag.m<-as.integer(dag0);## this creates one long vector - filled by cols from dag.m = same as internal C reprentation so fine.

res<-.Call("checkforcycles",dag.m,20
           #,PACKAGE="abn" ## uncomment to load as package not shlib
)


