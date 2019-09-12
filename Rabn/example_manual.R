dyn.load("abn.so")
source("ex0data.dump.txt");
source("abn-internal.R")
mydat<-ex0.dag.data[,c("b1","b2","b3","g1","b4","p2","p4")];## take a subset of cols

## setup distribution list for each node
mydists<-list(b1="binomial",
              b2="binomial",
              b3="binomial",
              g1="gaussian",
              b4="binomial",
              p2="poisson",
              p4="poisson"
             );
## null model - all independent variables
mydag.empty<-matrix(data=c(
                     0,0,0,0,0,0,0, # 
                     0,0,0,0,0,0,0, #
                     0,0,0,0,0,0,0, # 
                     0,0,0,0,0,0,0, # 
                     0,0,0,0,0,0,0, #
                     0,0,0,0,0,0,0, #
                     0,0,0,0,0,0,0  #
                     ), byrow=TRUE,ncol=7);
colnames(mydag.empty)<-rownames(mydag.empty)<-names(mydat);


mylist<-check.valid.data(data.df=mydat,data.dists=mydists,group.var=NULL);## return a list with entries bin, gaus, pois, ntrials and exposure

## run a series of checks on the DAG passed
check.valid.dag(dag.m=mydag.empty,data.df=mydat,is.ban.matrix=FALSE,group.var=NULL);



## now fit the model to calculate its goodness of fit
myres.c<-fitabn(dag.m=mydag.empty,data.df=mydat,data.dists=mydists,centre=TRUE,
                compute.fixed=FALSE);